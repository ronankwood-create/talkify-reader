#!/usr/bin/env python3
"""
Talkify - read documents aloud with the text on screen, highlighted as spoken.

    python talkify.py                    # opens the reader in your browser
    python talkify.py --engine stub      # no models needed, for UI work
    python talkify.py book.pdf           # open straight into a file

Stdlib HTTP server on purpose: one less thing to install, and localhost does
not need anything fancier.

The state this used to keep in one global dict now lives in four seams, each
with its own module and its own reason to exist:

    session.SessionManager      what is open and where you are in it (volatile)
    storage.LibraryRepository   documents on disk, plus the SQLite index
    storage.SettingsStore       config.json, written atomically
    coordinator.SynthCoordinator    all synthesis, bounded, priority-ordered

Nothing about the HTTP routes changed in that move. They are the contract the
player depends on and they stay exactly as they were.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import shutil
import sys
import threading
import time
import uuid
import webbrowser
import collections
import concurrent.futures
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs, unquote

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import ai
import chapters
import commute
import coordinator
import engines
import export
import extract
import figures
import jobs
import practice
import readaloud
import session
import storage

WEB = os.path.join(HERE, "web")

# Where a document's extracted text survives a restart. Only the text - the
# audio cache is not saved here, so a reloaded document re-synthesizes in the
# background same as a fresh upload would. By design, this deliberately
# does not use the future name.mp3+name.json export pairing - that's a
# decision for when the mobile-sync feature actually gets built.
# Overridable so smoke.py can point it at a throwaway directory instead of
# polluting the real library with test documents on every run.
LIBRARY_DIR = os.environ.get("TALKIFY_LIBRARY_DIR") or os.path.join(HERE, "library")

# User-local settings (default voice, later: API keys). Never committed - see
# .gitignore. Absent by default; every reader falls back to engine defaults.
CONFIG_PATH = os.environ.get("TALKIFY_CONFIG_PATH") or os.path.join(HERE, "config.json")

# Rendered figure crops and their manifests. A sidecar directory, never the
# document JSON: that file is rewritten on every position heartbeat, and
# binary assets have no business riding along with it. See figures.py.
FIGURE_CACHE_DIR = os.path.join(LIBRARY_DIR, "figure-cache")

# doc_id -> the manifest figures.build_index() published for it. The disk
# cache is what survives a restart; this is just so /api/figures does not
# re-read a JSON file on every poll.
FIGURE_MANIFESTS = {}


class _PdfCache:
    """Page counts and rendered pages, remembered between requests.

    Not the PyMuPDF handles, deliberately. An earlier version of this kept the
    Document open, assuming reopening a 291MB scan was the expensive part of
    serving a page. Measured, it is not: opening the file again costs about
    1.5ms against roughly 150ms to render one page at 150 DPI, so a kept-open
    handle buys nothing anyone could notice - and on Windows it takes a lock on
    the user's own book, so while Talkify was running they could not rename,
    move or delete the PDF they were reading. That is exactly the class of
    Windows papercut this project is built to avoid, traded for nothing.

    What is worth keeping is the answers. A page count cannot change while the
    file on disk does not, and /api/doc asked for one on every switch into a
    book; a rendered page is 150ms that need not happen twice, and the player
    prefetches the pages either side of every turn in split view, so paging
    back and forth re-asks for the same images constantly.

    Both are keyed on (path, size, mtime_ns) - the same stat-not-hash identity
    figures.fingerprint() already settled on for the same reason, that a
    content hash of a 291MB scan is not something to do on a page turn. The
    lock covers only the dictionaries, never a render, so two pages still
    render concurrently exactly as they did when every request opened its own
    handle.
    """

    # A page of text is tens of kilobytes; a page of a 200-DPI book scan is
    # three megabytes of PNG. Sixteen of those is fifty megabytes held for as
    # long as the document is open, which is why the count alone is not a
    # bound worth having. Whichever limit binds first wins.
    def __init__(self, max_pages=16, max_bytes=24 << 20):
        self._lock = threading.Lock()
        self._counts = {}               # path -> (key, page_count)
        self._pages = OrderedDict()     # (path, key, index, dpi) -> png bytes
        self.max_pages = max_pages
        self.max_bytes = max_bytes
        self._bytes = 0

    @staticmethod
    def _key(path):
        st = os.stat(path)
        return (st.st_size, st.st_mtime_ns)

    def page_count(self, path):
        """How many pages the PDF has, or 0 if it cannot be read at all.

        Quiet on every failure - a missing or unreadable file just means the
        split view and the page counter are not offered, exactly as before."""
        try:
            key = self._key(path)
        except OSError:
            return 0
        cached = self._counts.get(path)
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            import pymupdf
            with pymupdf.open(path) as doc:
                count = len(doc)
        except Exception:
            return 0
        with self._lock:
            self._counts[path] = (key, count)
        return count

    def render(self, path, index, dpi):
        """One page as PNG bytes. A page turned back to, or one the player has
        already prefetched, comes out of memory.

        None means the page is not there - no such file, or an index outside
        the document. A page that exists but will not render raises, so the
        route can still answer 500 with what went wrong rather than reporting
        a rendering failure as a page that does not exist."""
        try:
            key = self._key(path)
        except OSError:
            return None
        cache_key = (path, key, index, dpi)
        with self._lock:
            png = self._pages.get(cache_key)
            if png is not None:
                self._pages.move_to_end(cache_key)
                return png
        import pymupdf
        with pymupdf.open(path) as doc:
            if not (0 <= index < len(doc)):
                return None
            png = doc[index].get_pixmap(dpi=dpi).tobytes("png")
        with self._lock:
            previous = self._pages.pop(cache_key, None)
            if previous is not None:
                self._bytes -= len(previous)
            self._pages[cache_key] = png
            self._bytes += len(png)
            self._pages.move_to_end(cache_key)
            while self._pages and (len(self._pages) > self.max_pages
                                   or self._bytes > self.max_bytes):
                # Including the page just rendered, if it alone is over the
                # budget - a 300-DPI page of a large scan can be. The caller
                # still gets its PNG; it just will not be there next time,
                # which is the same bargain AudioCache strikes for one
                # oversized block.
                _key, evicted = self._pages.popitem(last=False)
                self._bytes -= len(evicted)
        return png

    def forget(self, path):
        """Drop everything remembered about one PDF - what deleting the
        document that pointed at it should do."""
        with self._lock:
            self._counts.pop(path, None)
            for k in [k for k in self._pages if k[0] == path]:
                self._bytes -= len(self._pages[k])
                del self._pages[k]

    def close(self):
        with self._lock:
            self._counts.clear()
            self._pages.clear()
            self._bytes = 0


PDFS = _PdfCache()


def _cache_budget():
    """How much synthesized audio may sit in RAM at once. A scanned textbook
    is thousands of blocks and WAV is uncompressed, so this is the difference
    between a long session and an out-of-memory kill."""
    raw = os.environ.get("TALKIFY_CACHE_BYTES")
    try:
        return int(raw) if raw else coordinator.DEFAULT_CACHE_BYTES
    except ValueError:
        return coordinator.DEFAULT_CACHE_BYTES


# The speech engine, chosen once in main(). The coordinator reaches it through
# _engine_synth rather than holding a reference, so building the engine after
# the coordinator exists is fine.
ENGINE = None

SETTINGS = storage.SettingsStore(CONFIG_PATH)
LIBRARY = storage.LibraryRepository(LIBRARY_DIR)
SESSIONS = session.SessionManager()
AUDIO_CACHE = coordinator.AudioCache(max_bytes=_cache_budget())

# PDFs only, since OCR on a real scanned book can take minutes and the browser
# needs page counts to show, not a spinner that looks the same stuck or fifteen
# minutes from done. Text/epub/txt open fast enough to stay synchronous - see
# the PDF-only branch in do_POST and _upload.
OPEN_JOBS = jobs.JobTracker("open")
EXPORT_JOBS = jobs.JobTracker("export")
PREVIEWS = coordinator.PreviewCache()


# Files under web/ are small, few, and read on every page load. Cached by
# (size, mtime) so an edit is picked up the moment it lands - the browser is
# told to revalidate rather than to keep them, so a change can never be
# masked by a stale copy in the tab.
_STATIC = {}
_STATIC_LOCK = threading.Lock()


def _static_file(path):
    """(body, content-type, etag) for a file under web/, or None."""
    try:
        st = os.stat(path)
    except OSError:
        return None
    key = (st.st_size, st.st_mtime_ns)
    with _STATIC_LOCK:
        held = _STATIC.get(path)
        if held is not None and held[0] == key:
            return held[1]
    try:
        with open(path, "rb") as f:
            body = f.read()
    except OSError:
        return None
    entry = (body,
             mimetypes.guess_type(path)[0] or "application/octet-stream",
             '"%s"' % hashlib.sha1(body).hexdigest()[:16])
    with _STATIC_LOCK:
        _STATIC[path] = (key, entry)
    return entry


def _audio_identity():
    """What the running engine says makes its audio, or "" if it cannot say."""
    identify = getattr(ENGINE, "audio_identity", None)
    if identify is None:
        return ""
    try:
        return identify() or ""
    except Exception:
        return ""


def _refresh_speech_settings():
    """Pick up an edited pronunciation map at the synthesis boundary.

    config.json is hand-editable and read through, so an entry added while
    Talkify is running should take effect on the next block rather than at
    the next restart. engines.refresh_speech_settings fingerprints the map, so
    the unchanged case - which is every block but the one after an edit -
    costs a settings read and a string compare. A settings file that will not
    parse reads as {} exactly as it always did, which here means no
    overrides and ordinary narration."""
    try:
        engines.refresh_speech_settings(SETTINGS.get("pronunciations"))
    except Exception:
        pass


def _note_audio_identity():
    """Both synthesis paths, before the engine runs. The cache empties itself
    when the thing producing the audio changes underneath it; the coordinator
    holds prefetched blocks outside that cache, and they were made by the same
    old producer, so they go at the same moment. Called from a dispatcher
    thread with no coordinator lock held - see coordinator._make."""
    if AUDIO_CACHE.note_identity(_audio_identity()):
        COORD.drop_prefetch()


def _engine_synth(text, voice):
    if ENGINE is None:
        raise RuntimeError("no speech engine configured")
    _refresh_speech_settings()
    _note_audio_identity()
    return ENGINE.synth(text, voice=voice)


class _ImmediateHandle:
    """What an engine with no submit() of its own looks like to the
    coordinator: the synthesis is already finished by the time this exists, so
    there is nothing left for cancel() to stop."""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def result(self, timeout=None):
        return self._value

    def cancel(self):
        pass


def _engine_submit(text, voice, meta):
    """The coordinator's optional pipelined path. Kokoro offers submit(), so a
    seek can cancel a block that is already inside the worker; every other
    engine falls back to the ordinary synchronous call and behaves exactly as
    it did before."""
    if ENGINE is None:
        raise RuntimeError("no speech engine configured")
    _refresh_speech_settings()
    _note_audio_identity()
    submit = getattr(ENGINE, "submit", None)
    if submit is None:
        return _ImmediateHandle(ENGINE.synth(text, voice=voice))
    return submit(text, voice=voice, meta=meta)


def _preview_line(label):
    """One short sentence per voice. Short on purpose: it is what makes
    pre-warming the whole gallery affordable, and a gallery is for judging a
    voice's timbre, not for listening to a paragraph."""
    return f"Welcome to Talkify. This is the {label.split(' (')[0]} voice."


def _make_preview(voice, label):
    return _engine_synth(_preview_line(label), voice)[:2]


_preview_warm_started = threading.Event()


def _prewarm_previews():
    """Synthesize every voice's sample clip in the background so the gallery
    plays instantly instead of stalling on the engine the first time each card
    is clicked.

    Additive and degrades to nothing: it runs in a daemon
    thread, it swallows per-voice failures (a voice that errors is simply not
    cached and gets retried on demand), and nothing in the reader waits on it.
    Sequential with a short gap between voices because Kokoro is one
    subprocess behind one lock - warming ten voices flat out would contend
    with the block a reader is waiting to hear, which is the one thing this
    must never do.
    """
    if _preview_warm_started.is_set():
        return
    _preview_warm_started.set()

    def run():
        time.sleep(1.5)                       # let any opening document get ahead first
        for v in engines.voices_for(ENGINE.name if ENGINE else None):
            if PREVIEWS.has(v["id"]):
                continue
            try:
                PREVIEWS.get_or_make(v["id"], lambda v=v: _make_preview(v["id"], v["label"]))
            except Exception:
                pass                          # retried on demand; never surfaced
            time.sleep(0.05)

    threading.Thread(target=run, name="preview-warmer", daemon=True).start()


def _note_block_seconds(doc_id, index, voice, audio, mime, words):
    """Called by the coordinator after every successful block. Recording the
    runtime here rather than in the request handler is what keeps the progress
    percentage honest for blocks the background warmer made, which is most of
    them."""
    doc = SESSIONS.get(doc_id)
    if doc is not None:
        doc.note_block_seconds(index, words[-1].e if words else 0.0)


COORD = coordinator.SynthCoordinator(_engine_synth, cache=AUDIO_CACHE,
                                     on_synth=_note_block_seconds,
                                     submit_fn=_engine_submit)
COORD.set_text_provider(SESSIONS.text_at)


# How long a mid-paragraph position heartbeat may go unwritten. Every event
# that matters (a block boundary, a seek, pause, stop, closing the tab, a clean
# server stop) writes immediately regardless - see _position - so this only
# bounds how much of the offset *within* one paragraph an *unclean* kill can
# cost: a crash or a power cut, not anything the user does deliberately.
#
# It is a straight trade. The player beats every three seconds, and writing on
# each one rewrote the whole document - megabytes, for a textbook - twenty
# times a minute for as long as anything was playing. Ten seconds cuts that
# threefold while keeping the worst case well inside a single paragraph, which
# is the granularity a reader would notice anyway. Set
# TALKIFY_POSITION_SAVE_INTERVAL=0 to write every beat, as before.
def _position_save_interval():
    raw = os.environ.get("TALKIFY_POSITION_SAVE_INTERVAL")
    try:
        return max(0.0, float(raw)) if raw else 10.0
    except ValueError:
        return 10.0


POSITION_SAVE_INTERVAL = _position_save_interval()

_last_saved = {}                       # doc_id -> monotonic time of last save
_last_saved_lock = threading.Lock()


def _save_doc(doc_id):
    """Persist everything this document currently is.

    The snapshot is taken *inside* the repository's per-document lock, and
    that is the whole point of routing it through update_record rather than
    calling save() with a record built out here. Two threads each snapshotting
    and then saving - the figure indexer settling while the reader switches
    voice, which is a completely ordinary pair of things to happen at once -
    can otherwise interleave as snapshot, snapshot, write, write, and the disk
    keeps the older thread's idea of the voice. The reader sees the switch
    work, and then finds it undone on the next launch.

    Inside the lock, a later snapshot can only be a superset of an earlier
    one, so the last write is always the most complete."""
    doc = SESSIONS.get(doc_id)
    if doc is None:
        return
    with _last_saved_lock:
        _last_saved[doc_id] = time.monotonic()

    def snapshot(record):
        record.clear()
        record.update(doc.record())

    LIBRARY.update_record(doc_id, snapshot)


def _save_doc_fields(doc_id, *fields):
    """Persist only these fields of a document, read fresh under its lock.

    For a writer that owns one part of the record and has no business
    rewriting the rest: the voice route changes `voice`, the chapter and
    figure indexers change their own index fields. Narrower than a full
    snapshot, and it cannot carry a stale copy of somebody else's field even
    in principle."""
    doc = SESSIONS.get(doc_id)
    if doc is None:
        return
    with _last_saved_lock:
        _last_saved[doc_id] = time.monotonic()

    def patch(record):
        current = doc.record()
        # Content first, and always from the live document. A partial save
        # builds on what was last written, and while a scan is still running
        # that is a shorter book than the one in memory. Persisting the
        # narrow field is the point; persisting yesterday's block list with
        # it is how append-only content quietly stops being append-only.
        for name in ("blocks", "block_pages", "content_revision", "ingestion"):
            if name in current:
                record[name] = current[name]
            else:
                record.pop(name, None)
        for name in fields:
            if name in current:
                record[name] = current[name]
            else:
                record.pop(name, None)

    LIBRARY.update_record(doc_id, patch)


def _due_for_save(doc_id):
    """Whether this document's last write is old enough that a drift-only
    heartbeat should go to disk after all."""
    with _last_saved_lock:
        return time.monotonic() - _last_saved.get(doc_id, 0.0) >= POSITION_SAVE_INTERVAL


def _warm(doc_id, start=0):
    """Aim the background warmer at a document so playback finds most of it
    already synthesized. Starts from `start` (the current/resume block) and
    works outward - forward first since that's what's about to be played,
    backward last - rather than always restarting from block 0.

    Unlike the thread this replaces, calling it again does not start a second
    walk racing the first: the coordinator holds one plan per document and
    re-aiming just replaces it. See coordinator.py."""
    doc = SESSIONS.get(doc_id)
    if doc is None:
        return
    COORD.warm(doc_id, len(doc.blocks), start=start,
               voice=doc.session.snapshot()["voice"])


class _SerialAnalysis:
    """One low-priority worker per kind of background analysis, not one thread
    per document.

    Reloading a library of thirty PDFs used to start thirty detection threads
    at once, all of them opening PDFs and walking pages while the reader is
    waiting on Kokoro for the paragraph they actually asked for. A 980-page scan
    explicitly must not steal CPU from active narration, so
    these queue up and run one at a time.

    Two lanes, not one: chapter detection can end up waiting on a local
    language model for a couple of minutes, and figure indexing must not sit
    behind that.
    """

    def __init__(self):
        self._queues = {}
        self._lock = threading.Lock()

    def submit(self, lane, fn, *args):
        with self._lock:
            q = self._queues.get(lane)
            if q is None:
                import queue as _queue
                q = self._queues[lane] = _queue.Queue()
                threading.Thread(target=self._run, args=(q,),
                                 name=f"analysis-{lane}", daemon=True).start()
        q.put((fn, args))

    @staticmethod
    def _run(q):
        while True:
            fn, args = q.get()
            try:
                fn(*args)
            except Exception:
                pass          # degrade to nothing; never take the reader down


_ANALYSIS = _SerialAnalysis()


def new_doc(blocks, title, doc_id=None, saved=None, source=None, block_pages=None,
            ingestion=None):
    """Register a document, save its text to the library, and kick off
    background synthesis for it immediately - it doesn't wait for anyone to
    press play. Passing doc_id re-registers a document under its previous id
    (used when reloading the library on startup). `saved` carries anything
    already known about it from a previous run (voice, position, ...); a
    freshly opened document gets sensible defaults instead. `source` is
    {"path", "options"} for a freshly opened PDF - kicks off background
    chapter detection; None for anything else (pasted text, txt, epub, or a
    document reloaded from a previous run without a recorded source)."""
    reopened = doc_id                      # None for anything genuinely new
    doc_id = doc_id or uuid.uuid4().hex[:12]
    saved = saved or {}
    source = source if source is not None else saved.get("source")
    doc = SESSIONS.new_document(
        doc_id, title, blocks,
        source=source,
        practice=practice.find(blocks),   # cheap pure-text scan, recomputed fresh every load
        chapters=saved.get("chapters", []),
        block_seconds={int(k): v for k, v in (saved.get("block_seconds") or {}).items()},
        voice=saved.get("voice") or SETTINGS.get("default_voice"),
        position=session.Position.from_dict(saved.get("position")),
        last_opened=saved.get("last_opened"),
        block_pages=block_pages if block_pages is not None else saved.get("block_pages"),
        content_revision=saved.get("content_revision", 1),
        ingestion=ingestion if ingestion is not None else saved.get("ingestion"))
    SESSIONS.add(doc)
    if doc.chapters:
        doc.set_chapter_detection(saved.get("chapter_detection")
                                  or {"state": "ready", "method": None})
    elif source:
        doc.set_chapter_detection({"state": "pending", "method": None})
    if source:
        # Keep whatever a previous run recorded (notably the fingerprint that
        # names its cache directory) while the indexer re-checks it, so a
        # reopened book can serve its figures before the thread has run.
        doc.set_figure_index({**(saved.get("figure_index") or {}),
                              "state": "pending"})
    if reopened is None:
        _save_doc(doc_id)
    else:
        # A document that came back from the library has nothing new to write:
        # everything in its record was just read from that same file, and the
        # two states set above are live UI signals rather than anything worth
        # persisting. Saving anyway meant every launch rewrote the whole
        # library - thirty books, potentially tens of megabytes, to say
        # exactly what the files already said. The first thing that actually
        # changes (a position, a detected chapter, a figure index settling,
        # opening it at all) writes it, as it always did.
        #
        # The SQLite index is a different matter: skipping the save skipped the
        # mirror with it, so a library whose library.db had been deleted came
        # back up with an empty index and stayed that way until each document
        # was opened. This puts back the guarantee that startup leaves the
        # index covering the library, and costs a primary-key lookup per
        # document when it already does.
        LIBRARY.ensure_indexed(doc_id, doc.record())
    _warm(doc_id, start=doc.session.current_block())
    if doc.ingesting():
        # Both indexes take one fixed snapshot of blocks and pages, so
        # starting them while the book is still growing would publish offsets
        # into text that has since moved. _settle() starts them once the
        # document stops changing - and on a partial book, never, which leaves
        # them quietly absent rather than pointing at half a document.
        doc.set_chapter_detection({"state": "pending", "method": None})
    else:
        if source and not doc.chapters:
            _ANALYSIS.submit("chapters", _detect_chapters, doc_id)
        if source:
            _ANALYSIS.submit("figures", _index_figures, doc_id)
    return doc


def _index_figures(doc_id):
    """Build the figure index for a PDF, in the background, after it is
    already open and readable.

    Everything about this is written to the project's rule that an enhancement
    which fails must be indistinguishable from one that was never built: it
    runs on a daemon thread, it publishes only if the document is still the
    one it started on, and every failure - no PyMuPDF, no text layer, no
    captions, no images, an unwritable cache - lands on the same quiet
    "none". No dialog, no empty panel, no delay to playback."""
    doc = SESSIONS.get(doc_id)
    if doc is None or not doc.source:
        return
    path = doc.source.get("path")
    options = doc.source.get("options") or {}
    if not path or not os.path.exists(path):
        if SESSIONS.get(doc_id) is doc:
            doc.set_figure_index({"state": "none"})
        return
    snap = doc.content_snapshot()
    blocks, block_pages = list(snap.blocks), snap.block_pages
    try:
        fp = figures.fingerprint(path, options)
    except OSError:
        doc.set_figure_index({"state": "none"})
        return
    # The document must still be the one this started on - reopening a book
    # replaces it, and publishing against the old one would bind figures to
    # word offsets in text nobody is reading any more.
    stale = lambda: SESSIONS.get(doc_id) is not doc
    manifest = figures.build_index(path, blocks, block_pages, FIGURE_CACHE_DIR,
                                   options=options, cancelled=stale)
    # The index names word offsets inside specific blocks, so it is only valid
    # for the exact text it was built from. A revision that moved underneath it
    # means more of the book arrived while it ran; the next run indexes that.
    if stale() or doc.content_snapshot().revision != snap.revision:
        return
    FIGURE_MANIFESTS[doc_id] = manifest
    doc.set_figure_index({"state": manifest.get("state", "none"),
                          "revision": manifest.get("revision"),
                          "fingerprint": fp,
                          "count": len(manifest.get("items") or [])})
    _save_doc_fields(doc_id, "figure_index")


def _selected_pages(path, options, page_count=None):
    """The PDF pages this document was opened with, sorted, or None when it
    was opened whole.

    Derived from the selection the user actually made, not from where text
    happened to be found: a whole book whose first two pages are blank is
    still a whole book, and its page view should still be able to show them.
    """
    spec = (options or {}).get("pages")
    if not spec or not path:
        return None
    total = page_count if page_count is not None else PDFS.page_count(path)
    if not total:
        return None
    try:
        indexes = sorted({i for i in readaloud.parse_pages(spec, total)
                          if 0 <= i < total})
    except Exception:
        return None
    return indexes or None


def _page_runs(pages):
    """A sorted page list as inclusive [start, end] runs.

    What the browser is told. "20-100" is one run and "20, 400-410" is two -
    a span would have made the second one three hundred and ninety-one pages
    the page view would happily render, which is the bug this whole change is
    about."""
    if not pages:
        return None
    runs = [[pages[0], pages[0]]]
    for page in pages[1:]:
        if page == runs[-1][1] + 1:
            runs[-1][1] = page
        else:
            runs.append([page, page])
    return runs


def _doc_selected_pages(doc):
    """The same, for a live document."""
    source = getattr(doc, "source", None) or {}
    return _selected_pages(source.get("path"), source.get("options"))


def _doc_page_runs(doc):
    return _page_runs(_doc_selected_pages(doc))


def _chapter_scan_pages(doc):
    """Which pages chapter detection may read.

    The selection first: it is exact, it is what the page route enforces, and
    it exists from the moment the document is opened. Block provenance is the
    fallback for a document opened before selections were recorded - and only
    then, because provenance covers the pages that produced text, which while
    a scan is still running is fewer than the reader asked for."""
    chosen = _doc_selected_pages(doc)
    if chosen:
        return chosen
    return _loaded_page_range_pages(doc)


def _loaded_page_range_pages(doc):
    """Every page the document's blocks actually came from, or None."""
    window = _loaded_page_range(doc)
    return list(range(window[0], window[1] + 1)) if window else None


def _loaded_page_range(doc):
    """The inclusive span of PDF pages this document actually holds, or None
    for a document that is the whole file.

    Opening "pages 20-100" of a 980-page book opens a book of eighty-one
    pages, and every background pass has to agree about that. It was measured
    not agreeing: the running-head scan OCR-probed 428 pages of that file for
    chapter marks that could only ever have applied to eighty-one of them, and
    the machine was unusable while it did. Taken from the block provenance
    rather than the requested option string, so it describes what was read
    rather than what was asked for, and so it grows correctly while a scan is
    still appending."""
    try:
        pages = doc.content_snapshot().block_pages
    except Exception:
        return None
    numbers = [p for p in (pages or ()) if isinstance(p, int) and p >= 0]
    return (min(numbers), max(numbers)) if numbers else None


def _detect_chapters(doc_id):
    """Off the critical path, as this
    feature: runs once in the background after a PDF is already open and
    readable, and any failure - no running heads, no contents page, Ollama
    unreachable, a verification check failing, the PDF having moved - just
    leaves doc.chapters empty rather than surfacing anything to the reader.

    Since Phase 5 the first thing chapters.detect() tries is the running-head
    scan, which is local, needs no language model, and is the method
    that actually worked in testing."""
    doc = SESSIONS.get(doc_id)

    def settle(method=None):
        """Whatever happened, say that nothing more is coming. The player
        stops polling on this rather than on a two-minute timer."""
        d = SESSIONS.get(doc_id)
        if d is not None and d is doc:
            info = {"state": "ready" if d.chapters else "none",
                    "method": method if d.chapters else None}
            # printed_page is paired with the published start (see
            # chapters._detect_running_heads), so that is the pair to subtract.
            offsets = [c["page_pdf"] - c["printed_page"] for c in d.chapters
                       if c.get("printed_page") is not None
                       and c.get("page_pdf") is not None]
            if offsets:
                info["pdf_minus_printed_offset"] = offsets[0]
            d.set_chapter_detection(info)
            _save_doc_fields(doc_id, "chapters", "chapter_detection")

    if doc is None or not doc.source:
        return
    path = doc.source["path"]
    if not os.path.exists(path):
        settle()
        return
    try:
        stale = lambda: SESSIONS.get(doc_id) is not doc
        raw = chapters.detect(path, cancelled=stale,
                              pages=_chapter_scan_pages(doc))
        if not raw:
            settle()
            return
        # Which PDF page each block came from. The extraction that opened this
        # document already worked that out and handed it over, so re-running
        # the whole pipeline here - minutes on a scanned textbook, for an
        # answer already in hand - is pure waste. Only a document that came in
        # without page provenance (an older saved library, or a PDF that fell
        # back to the plain text path) has to pay for a second pass, and the
        # block comparison that guards the mapping is exactly as strict either
        # way: same length and same text, or bail rather than mis-point a
        # chapter.
        snap = doc.content_snapshot()
        block_pages = snap.block_pages
        if block_pages is None or len(block_pages) != len(snap.blocks):
            blocks2, block_pages = extract.from_pdf_with_pages(
                path, **doc.source.get("options", {}))
            if (SESSIONS.get(doc_id) is not doc
                    or doc.content_snapshot().revision != snap.revision
                    or tuple(blocks2) != snap.blocks):
                return
            doc.set_block_pages(block_pages)
        elif SESSIONS.get(doc_id) is not doc:
            return
        mapped = []
        for ch in raw:
            bi = next((i for i, p in enumerate(block_pages) if p >= ch["page"]), None)
            if bi is None:
                continue
            # `title` and `block` are the whole contract the player has ever
            # needed; the rest is provenance (which method found this, how
            # sure it is, which PDF page it points at) that the export dialog
            # and the split view can use and older clients simply ignore.
            mapped.append({**{k: v for k, v in ch.items() if k != "page"},
                           "title": ch["title"], "block": bi,
                           "page_pdf": ch["page"]})
        deduped = []
        for c in mapped:
            if not deduped or c["block"] > deduped[-1]["block"]:
                deduped.append(c)
        if not deduped:
            settle()
            return
        if SESSIONS.get(doc_id) is doc:
            doc.set_chapters(deduped)
            settle(method=deduped[0].get("method"))
    except Exception:
        settle()
        return


def load_library():
    """Bring back every document saved from a previous run, oldest first."""
    for doc_id, saved in LIBRARY.load_all():
        try:
            doc = new_doc(saved["blocks"], saved["title"], doc_id=doc_id, saved=saved)
        except (KeyError, TypeError):
            continue
        if doc.ingesting():
            _resume_ingestion(doc)


def _resume_ingestion(doc):
    """Pick a half-read scan up where the last run left it.

    A clean stop settles ingestion; an unclean one leaves it "running", which
    on the next start means exactly this. The saved next_ordinal is never
    ahead of the blocks it describes - both are written in one record - so
    resuming can at worst re-read a page or two, and the OCR cache makes that
    nearly free.

    A source that has gone away settles the document as partial and keeps
    every block it already had. Losing the file must not lose the reading.
    """
    record = doc.ingestion
    path = (doc.source or {}).get("path")
    options = (doc.source or {}).get("options") or {}
    plan = None
    if path and os.path.exists(path):
        try:
            plan = readaloud.plan_ingestion(path, pages=options.get("pages"),
                                            math=options.get("math", "skip"))
        except Exception:
            plan = None
    if plan is None or plan.source_fingerprint != record.get("source_fingerprint"):
        doc.settle_ingestion({**record, "state": "partial",
                              "note": "the source file is gone or has changed",
                              "updated_at": session.now_iso()})
        _save_doc(doc.id)
        return
    carry = record.get("carry")
    if isinstance(carry, (list, tuple)) and len(carry) == 2:
        carry = (str(carry[0]), int(carry[1]))
    else:
        carry = None
    task = IngestionTask(record.get("task_id") or ("ocr-" + uuid.uuid4().hex[:8]),
                         doc, plan,
                         ordinal=int(record.get("next_ordinal") or 0),
                         carry=carry,
                         failed=record.get("failed_pages") or [],
                         succeeded=int(record.get("pages_succeeded") or 0))
    if task.task_id != record.get("task_id"):
        # Every publication checks this id, so a record that lost it needs the
        # document brought into step before anything can be appended.
        doc.settle_ingestion({**record, "task_id": task.task_id})
    INGESTION.start(task)


def synth_block(doc_id, i, voice=None, priority=coordinator.PRIORITY_INTERACTIVE,
                cache_result=True):
    """Synthesize block i of a document, once. Concurrent requests for the
    same block - a live play request racing the background warmer, say - wait
    for the in-flight synthesis instead of each running TTS+Whisper. The
    coordinator owns that deduplication now; this is just the lookup of which
    text and which voice.

    `voice` overrides the document's current one. Playback never passes it -
    what you hear is whatever the session says - but an export does, so a book
    can be exported in a voice the reader is not currently set to."""
    doc = SESSIONS.get(doc_id)
    if doc is None:
        raise KeyError("no such document")
    text = doc.text_at(i)
    if text is None:
        raise KeyError("no such block")
    target_voice = voice or doc.session.snapshot()["voice"]
    return COORD.synth(doc_id, i, text, target_voice, priority=priority,
                       cache_result=cache_result)


def _export_dir():
    """Where exported MP3s land. Overridable in config.json (e.g. pointed at
    a local Google Drive sync folder). Defaults
    to a folder next to the app so it works with no configuration at all."""
    return (os.environ.get("TALKIFY_EXPORT_DIR") or SETTINGS.get("export_dir")
            or os.path.join(HERE, "exports"))



# A generous cap on how much of a text selection gets sent to /api/explain
# as context - well above a paragraph (320 chars, extract.MAX_BLOCK) since a
# selection can legitimately span several, but still far short of what would
# risk Ollama's silent prompt truncation (see ai.py).
MAX_SELECTION_CHARS = 4000

_BAD_FILENAME_CHARS = '<>:"/\\|?*'


def _safe_filename(title):
    cleaned = "".join(c for c in title if c not in _BAD_FILENAME_CHARS).strip(" .")
    return cleaned[:120] or "untitled"


def _block_stream(doc_id, job_id, start, end, voice=None, cache_result=True):
    """(mime, generator of (audio_bytes, [word dicts])) for blocks
    [start, end) of a document, all in one voice.

    The generator hands the export encoder one block at a time and keeps no
    history: a textbook's decoded WAV runs to gigabytes, and holding all of
    it before encoding a single frame is exactly what this replaced. The
    first block is synthesized up front because the encoder has to know the
    mime (and, through it, the sample rate) before it opens the output file;
    it is handed back through the generator rather than synthesized twice.

    `cache_result` decides whether what these calls make is admitted to the
    cache, and the caller decides it once for the whole job. An export in the
    voice the reader is set to makes exactly the audio the reader's own warmer
    would go on to make, so dropping it bought a second synthesis of the whole
    book the moment the export finished. An export in a voice deliberately
    chosen for the file makes audio nobody is about to play, and admitting it
    would evict the reader's voice to hold it - see the note in _run_export."""
    first = None
    mime = ENGINE.mime if ENGINE else "audio/wav"
    if start < end:
        audio, mime, words = synth_block(doc_id, start, voice=voice,
                                         priority=coordinator.PRIORITY_EXPORT,
                                         cache_result=cache_result)
        EXPORT_JOBS.update(job_id, done=start + 1)
        first = (audio, [asdict(w) for w in words])

    def blocks():
        nonlocal first
        if first is not None:
            block, first = first, None      # let the audio go as soon as it's encoded
            yield block
        for i in range(start + 1, end):
            audio, _, words = synth_block(doc_id, i, voice=voice,
                                          priority=coordinator.PRIORITY_EXPORT,
                                          cache_result=cache_result)
            EXPORT_JOBS.update(job_id, done=i + 1)
            yield audio, [asdict(w) for w in words]

    return mime, blocks()


# ------------------------------------------------------- the commute package
#
# An export is not only files to download - it is the folder a phone receives,
# and the one member of it that keeps changing is the position sidecar.
# Everything here exists so that sidecar can be kept current later, from the
# playback heartbeat, without re-reading a book-length timings file or
# guessing which folder the book went into. See commute.py.

def _manifest_content_sha(pkg_dir, slug):
    try:
        with open(os.path.join(pkg_dir, f"{slug}.package.json"), encoding="utf-8") as f:
            return json.load(f).get("document_content_sha256")
    except (OSError, ValueError):
        return None


def _package_dir(out_dir, slug, content_sha):
    """`<slug>--<hash8>`, extended to twelve characters only on a real
    collision - two different books whose titles slug the same way *and*
    whose content hashes share a prefix.

    Re-exporting the same content reuses its folder, which is what makes a
    second export an update of the package rather than a second copy of the
    book. A folder with no readable manifest is a crashed export, not a
    collision, and is reused too."""
    short = os.path.join(out_dir, f"{slug}--{content_sha[:8]}")
    if not os.path.isdir(short):
        return short
    existing = _manifest_content_sha(short, slug)
    if existing is None or existing == content_sha:
        return short
    return os.path.join(out_dir, f"{slug}--{content_sha[:12]}")


def _record_export(out_dir, content_sha, *, mode, directory, slug, rendition,
                   checksum, total_duration_ms, block_offset, block_starts,
                   voice, chapters):
    """Write this export into the desktop's private index, keyed by what the
    book says rather than what it is called - a rename must not orphan a
    package, and two files of the same text share one."""
    entry = {"mode": mode, "dir": os.path.abspath(directory), "slug": slug,
             "rendition_id": rendition,
             "package_id": commute.package_id_for(rendition),
             "checksum": checksum, "total_duration_ms": total_duration_ms,
             "block_offset": block_offset, "block_starts_ms": block_starts,
             "voice": voice, "chapters": chapters}
    commute.record_package(out_dir, content_sha, entry)
    return entry


def _position_document(entry, position):
    """A spec-valid sidecar body for one reading position."""
    block, word = int(position.get("block", 0)), int(position.get("word", 0))
    elapsed = commute.elapsed_for(entry, block, position.get("within", 0.0))
    if elapsed is None:
        # The reader is somewhere this package does not contain: ahead of it
        # in a book that has grown, or in the front matter before the first
        # chapter, which chapter mode does not export. The package's own
        # beginning is the only honest answer. Inventing an offset into audio
        # that does not cover this block is how a resume lands somewhere else
        # entirely and reads as corruption.
        elapsed, block, word = 0, int(entry.get("block_offset", 0)), 0
    chapter_index, chapter_elapsed = commute.locate(
        entry.get("chapters") or [], elapsed)
    return commute.position_document(
        entry["rendition_id"], entry["checksum"], elapsed, block, word,
        entry["total_duration_ms"], chapter_index, chapter_elapsed)


_content_sha_cache = {}                # doc_id -> (content revision, sha256)
_content_sha_lock = threading.Lock()
_sidecar_warned = set()                # doc_ids whose first failure was reported


def _content_sha(doc_id, content):
    """A document's content hash, cached against its content revision.

    The heartbeat arrives every few seconds and a textbook is megabytes of
    text; hashing all of it each time would cost more than the write it
    guards. Content is append-only, so the revision counter is a complete
    cache key - text that has already been published never changes."""
    with _content_sha_lock:
        cached = _content_sha_cache.get(doc_id)
        if cached is not None and cached[0] == content.revision:
            return cached[1]
    sha = commute.content_sha256(content.blocks)
    with _content_sha_lock:
        _content_sha_cache[doc_id] = (content.revision, sha)
    return sha


_export_index_cache = {}               # (path, mtime_ns, size) -> packages
_export_index_lock = threading.Lock()


def _export_index(out_dir=None):
    """The desktop's package index, re-read only when the file has changed.

    The library listing asks this once per document and a library is thirty
    books, so parsing the same JSON thirty times per poll is work the reader
    pays for and never sees. Keyed on the file's own stamp rather than on a
    timer: an export writes the index through an atomic replace, so a stale
    answer is impossible rather than merely unlikely."""
    out_dir = out_dir or _export_dir()
    path = os.path.join(out_dir, commute.INDEX_NAME)
    try:
        st = os.stat(path)
        stamp = (path, st.st_mtime_ns, st.st_size)
    except OSError:
        stamp = (path, None, None)
    with _export_index_lock:
        if stamp in _export_index_cache:
            return _export_index_cache[stamp]
    packages = commute.read_index(out_dir)
    with _export_index_lock:
        _export_index_cache.clear()
        _export_index_cache[stamp] = packages
    return packages


def _export_entry(doc, content=None):
    """The index entry for exactly this document's text, or None.

    Keyed by content hash, so a book that has grown since it was exported is
    a different rendition and finds nothing - the same rule the position
    sidecar follows."""
    try:
        return _export_index().get(
            _content_sha(doc.id, content or doc.content_snapshot()))
    except Exception:
        return None


def _is_exported(doc, content=None):
    """Whether a finished export of this text is still on disk.

    The index is the desktop's memory of exports; the files are the truth. A
    folder the reader has since emptied must not leave the library offering a
    download nothing can serve, so an entry is only believed while the audio
    or the manifest it names is still there. Both, because the two export
    modes write different things: one file for a whole book, a manifest and a
    folder of chapters for a package."""
    entry = _export_entry(doc, content)
    if not entry:
        return False
    pkg_dir, slug = entry.get("dir") or "", entry.get("slug") or ""
    try:
        return (os.path.exists(os.path.join(pkg_dir, slug + ".mp3"))
                or os.path.exists(os.path.join(pkg_dir, slug + ".package.json")))
    except (OSError, ValueError):
        return False


def _exported_file(doc_id, which=None):
    """The MP3 on disk for a document's current export, or None.

    A chaptered package is many files, and `which` picks one by ordinal -
    which is the order they sort in, since commute.chapter_stem numbers them
    for exactly that. The directory is only listed for a package, never for a
    single-file export: that one lives in the shared exports folder alongside
    every other book, and picking a file out of it by position would serve
    somebody else's."""
    doc = SESSIONS.get(doc_id) if doc_id else None
    if doc is None:
        return None
    entry = _export_entry(doc)
    if not entry:
        return None
    pkg_dir, slug = entry.get("dir") or "", entry.get("slug") or ""
    single = os.path.join(pkg_dir, slug + ".mp3")
    if os.path.exists(single):
        return single
    if entry.get("mode") != "chapters":
        return None
    try:
        names = sorted(n for n in os.listdir(pkg_dir)
                       if n.lower().endswith(".mp3"))
        idx = int(which) if which not in (None, "") else 0
    except (OSError, ValueError, TypeError):
        return None
    if not (0 <= idx < len(names)):
        return None
    return os.path.join(pkg_dir, names[idx])


def _update_export_position(doc_id, doc, block, word, within):
    """Keep an exported package's position sidecar current as the user reads.

    Only if this document has been exported at all, and only if what was
    exported is still the text being read: a book that has grown since hashes
    differently, that is a different rendition, and the spec is explicit that
    a position may not be inherited across one. Both misses are silence.
    Nothing here can fail the heartbeat - that request is what drives
    resume-on-reopen, and losing the reader's place over a file a phone may
    never read would be a poor trade. The first failure per document is
    printed so a broken export folder is not completely invisible."""
    try:
        out_dir = _export_dir()
        entry = commute.read_index(out_dir).get(
            _content_sha(doc_id, doc.content_snapshot()))
        if not entry or not os.path.isdir(entry.get("dir") or ""):
            return
        commute.write_position(entry["dir"], entry["slug"], _position_document(
            entry, {"block": block, "word": word, "within": within}))
    except Exception as e:
        if doc_id not in _sidecar_warned:
            _sidecar_warned.add(doc_id)
            print(f"  commute position sidecar not updated: "
                  f"{type(e).__name__}: {e}", file=sys.stderr)


def _run_export(doc_id, job_id, mode="full", voice=None):
    """Runs on its own thread - re-encoding potentially hundreds of blocks is
    too slow to hold an HTTP request open for. Walks every block through
    synth_block() first (a no-op for anything the background warmer already
    cached), so exporting a document you haven't finished listening to still
    produces the complete file(s).

    Those walks queue at PRIORITY_EXPORT, which sits below live playback and
    above every kind of warming: starting an export never makes the paragraph
    you are listening to arrive late, and the warmer topping its window back
    up after every block can no longer stall the export indefinitely.

    `voice` is what the whole file gets read in, defaulting to the document's
    current one. The warmer is re-aimed at block 0 in that same voice, so the
    work it does while the export runs is work the export is about to need
    instead of blocks in a voice this file will not use.

    Blocks stream through export.export_stream() into a `.part` file that is
    atomically renamed on success, so RAM stays flat however long the book is
    and a failure leaves no half-written MP3 behind.

    mode="chapters" splits the output at each detected chapter's block
    instead of producing one contiguous file - each chapter getting its own
    audio file and its own timings starting at 0.0s. Only valid when the
    document actually has chapters (see chapters.py); the caller validates
    that before starting the job. That mode writes a commute package: one
    folder holding the chapter files, a manifest tying them to a single
    document-wide timeline, and a position sidecar - the unit that is copied
    to a phone. See commute.py.

    Both modes end by writing `<name>.position.json` and recording the export
    in the desktop's private index, so the position heartbeat can keep that
    sidecar current without re-reading a book-length timings file."""
    doc = SESSIONS.get(doc_id)
    if doc is None:
        EXPORT_JOBS.update(job_id, status="error", error="document no longer exists")
        return
    # One snapshot, one n. Ingestion is settled by the time an export is
    # allowed to start, but taking the length twice is how a book that grew
    # between the two reads produces a file whose timings stop short of its
    # audio. The same snapshot is what the content hash is taken over, so the
    # rendition names the text that was actually encoded.
    content = doc.content_snapshot()
    n = len(content.blocks)
    reader_voice = doc.session.snapshot()["voice"]
    voice = voice or reader_voice
    # Whether this job's audio belongs in the cache at all. Decided once, here,
    # from the voice the reader is actually set to rather than from whether a
    # voice was passed: asking for the reader's own voice explicitly is still
    # an export of what the reader is about to hear.
    should_cache = (voice == reader_voice)
    EXPORT_JOBS.update(job_id, total=n, voice=voice)
    doc_chapters = doc.chapters
    # This is an export prefetch, not ordinary warming, and its arguments say
    # so: it queues at export priority so the engine never stands idle between
    # blocks, and it starts at the first block this export will actually write
    # and never wraps behind it.
    #
    # Whether it admits what it makes to the cache depends on the voice. In
    # the reader's own voice this walk synthesizes the audio the reader's
    # warmer would go on to make anyway; holding it out of the LRU meant a book
    # was synthesized twice, once for the file and once again for playback. So
    # an export of the document you are reading is warming rather than work
    # done beside it, and the walk it leaves behind is what lets the finally
    # block below skip re-warming.
    #
    # In a voice chosen just for the file it is the opposite trade. Nothing is
    # going to play that audio, and a one-pass scan of a whole book would
    # evict the blocks the reader's voice actually needs to hold a rendition
    # nobody asked to hear. That export stays out of the cache entirely, which
    # leaves the reader's voice as warm as it was and lets the finally block
    # see it still needs warming.
    first_block = (doc_chapters[0]["block"]
                   if mode == "chapters" and doc_chapters else 0)
    COORD.warm(doc_id, n, start=first_block, voice=voice,
               priority=coordinator.PRIORITY_EXPORT, cache_result=should_cache,
               wrap=False)
    try:
        out_dir = _export_dir()
        os.makedirs(out_dir, exist_ok=True)
        base = _safe_filename(doc.title)
        content_sha = commute.content_sha256(content.blocks)
        rendition = commute.rendition_id(content_sha, voice)
        position = doc.session.snapshot()["position"]

        if mode == "chapters" and doc_chapters:
            bounds = [c["block"] for c in doc_chapters] + [n]
            slug = commute.slugify(doc.title)
            stems = commute.chapter_slugs([c["title"] for c in doc_chapters])
            pkg_dir = _package_dir(out_dir, slug, content_sha)
            os.makedirs(pkg_dir, exist_ok=True)
            files, manifest_chapters = [], []
            block_starts, elapsed_ms = [], 0
            for idx, ch in enumerate(doc_chapters):
                start, end = bounds[idx], bounds[idx + 1]
                stem = commute.chapter_stem(idx + 1, stems[idx])
                mp3_path = os.path.join(pkg_dir, f"{stem}.mp3")
                json_path = os.path.join(pkg_dir, f"{stem}.json")
                mime, blocks = _block_stream(doc_id, job_id, start, end,
                                             voice=voice,
                                             cache_result=should_cache)
                result = export.export_stream(
                    blocks, mp3_path, json_path, mime,
                    chapter_index=idx, block_start_index=start,
                    id3=commute.chapter_id3(doc.title, ch["title"], idx + 1,
                                            len(doc_chapters), stem))
                duration_ms = round(result["duration"] * 1000)
                # Where each block starts on the *document-wide* timeline. The
                # chapter's own audio starts at 0.0s, so this is the only place
                # the two timelines are related, and it is recorded once here
                # rather than recomputed by every reader of the sidecar.
                local = 0.0
                for block_duration in result["block_durations"]:
                    block_starts.append(elapsed_ms + round(local * 1000))
                    local += block_duration
                manifest_chapters.append(commute.manifest_chapter(
                    idx, idx + 1, ch["title"], stem, elapsed_ms, duration_ms,
                    start, end, result["audio_sha256"], result["timings_sha256"]))
                elapsed_ms += duration_ms
                files.append({"title": ch["title"], "name": f"{stem}.mp3",
                              "path": mp3_path, "json_path": json_path})
            block_starts.append(elapsed_ms)
            manifest = commute.build_manifest(doc.title, rendition, content_sha,
                                              elapsed_ms, manifest_chapters)
            # Written last, on purpose. A whole staging directory cannot be
            # renamed over an existing one on Windows, so the spec's fallback
            # is what this does: the manifest is the last file to appear, and
            # a folder without one is not yet a package.
            manifest_path = os.path.join(pkg_dir, f"{slug}.package.json")
            commute.write_json(manifest_path, manifest)
            entry = _record_export(
                out_dir, content_sha, mode="chapters", directory=pkg_dir,
                slug=slug, rendition=rendition,
                checksum=manifest["package_checksum"]["value"],
                total_duration_ms=elapsed_ms, block_offset=bounds[0],
                block_starts=block_starts, voice=voice,
                chapters=[{"chapter_index": c["chapter_index"],
                           "global_start_ms": c["global_start_ms"],
                           "duration_ms": c["duration_ms"]}
                          for c in manifest_chapters])
            commute.write_position(pkg_dir, slug, _position_document(entry, position))
            EXPORT_JOBS.update(job_id, status="done", files=files, dir=pkg_dir,
                               package=manifest_path, slug=slug)
        else:
            mp3_path = os.path.join(out_dir, f"{base}.mp3")
            json_path = os.path.join(out_dir, f"{base}.json")
            mime, blocks = _block_stream(doc_id, job_id, 0, n, voice=voice,
                                         cache_result=should_cache)
            result = export.export_stream(blocks, mp3_path, json_path, mime)
            total_ms = round(result["duration"] * 1000)
            block_starts, local = [], 0.0
            for block_duration in result["block_durations"]:
                block_starts.append(round(local * 1000))
                local += block_duration
            block_starts.append(total_ms)
            # A single-file export writes no manifest - spec section 3.1 is
            # three files - but its sidecar still has to carry a package
            # checksum, so one is taken over the manifest this export *would*
            # have had: one chapter, the whole book, these two files. It is a
            # reference to immutable contents either way, which is the whole
            # job of that field.
            entry = _record_export(
                out_dir, content_sha, mode="full", directory=out_dir,
                slug=base, rendition=rendition,
                checksum=commute.package_checksum(
                    rendition, content_sha, total_ms,
                    [commute.manifest_chapter(
                        0, 1, doc.title, base, 0, total_ms, 0, n,
                        result["audio_sha256"], result["timings_sha256"])]),
                total_duration_ms=total_ms, block_offset=0,
                block_starts=block_starts, voice=voice, chapters=[])
            commute.write_position(out_dir, base, _position_document(entry, position))
            EXPORT_JOBS.update(job_id, status="done", path=mp3_path,
                               json_path=json_path, dir=out_dir, slug=base)
    except Exception as e:
        EXPORT_JOBS.update(job_id, status="error", error=f"{type(e).__name__}: {e}")
    finally:
        # The export owned this document's warm plan for however many hours it
        # ran. Hand it back: release the tail of the prefetch window nobody is
        # going to collect now, and re-aim the warmer at where the reader
        # actually is, in the voice the reader is actually set to - which for
        # an export in a second voice is not the one just used. Without this a
        # document is left with no plan at all until it is next opened, so the
        # first paragraph played after an export waits on a cold engine.
        COORD.drop_prefetch(doc_id)
        reader = SESSIONS.get(doc_id)
        if reader is not None:
            # An export that ran in the voice the reader is set to has just
            # cached the whole document, and re-aiming the warmer at it would
            # walk a plan whose every block is a hit. Counting the blocks
            # cached in that voice rather than in any voice is the point: an
            # export in a second voice leaves the reader's own voice exactly
            # as cold as it was, and that document does still need warming.
            voice_now = reader.session.snapshot()["voice"]
            warmed = COORD.voice_counts(doc_id).get(voice_now, 0)
            if warmed < len(reader.blocks):
                _warm(doc_id, start=reader.session.current_block())


# ------------------------------------------------------ progressive opening
#
# A 900-page scan takes minutes to OCR. Waiting for all of it before showing
# anything is the shape of mistake to avoid: the reader gets
# nothing at all while a heuristic-heavy pipeline grinds. So a scanned book is
# opened from its first readable page and grown from behind, and every part of
# that degrades to the old behaviour if it cannot start.
#
# The one rule everything here serves: content is append-only. Once a block has
# been handed to a browser, neither its text nor its index may change - audio,
# timings, chapter marks, the saved position and the DOM all name blocks by
# index.

# How many selected pages to read before deciding a document is readable. Two,
# because one page of a scanned textbook is quite often a title plate.
INGEST_PROBE_PAGES = 2
# Pages whose lines are kept as evidence for what counts as a running head.
# Bounded because a running head is only ever recurring within the section it
# belongs to, and because this is a 900-page book.
FURNITURE_SAMPLE = 48
# A cap on how many failed page numbers a document record will carry, so a
# corrupt 900-page scan cannot bloat the library file.
MAX_FAILED_PAGES = 100
# How often a growing document is written to disk. Every page would be an
# honest but wasteful choice on a 900-page book; the OCR cache means a crash
# costs re-reading a couple of pages, not re-OCRing them.
INGEST_SAVE_INTERVAL = 2.0


class IngestionTask:
    """One document being read off a scanner in the background.

    `document` is the object, not the id: a document deleted and reopened
    under the same id is a different object, and that identity check is what
    stops a task from publishing into a book nobody asked it to read.
    """

    __slots__ = ("task_id", "doc_id", "document", "plan", "cancel", "ordinal",
                 "carry", "observed", "failed", "succeeded", "source", "saved_at")

    def __init__(self, task_id, document, plan, ordinal=0, carry=None,
                 observed=(), failed=(), succeeded=0, source=None):
        # source(path, indices, cancelled=...) -> (page, lines, error) in order.
        # Injectable so the whole publish/settle/resume machinery can be driven
        # from a fake page iterator - smoke.py must not need Tesseract, and the
        # parts that fail quietly here are the ordering and identity checks,
        # not the OCR.
        self.source = source or readaloud.iter_page_lines
        self.task_id = task_id
        self.doc_id = document.id
        self.document = document
        self.plan = plan
        self.cancel = threading.Event()
        self.ordinal = ordinal              # index into plan.selected_pages
        self.carry = carry                  # a paragraph held back for the next page
        self.observed = collections.deque(observed, maxlen=FURNITURE_SAMPLE)
        self.failed = list(failed)
        self.succeeded = succeeded
        self.saved_at = 0.0

    def live(self):
        """Whether this task still owns this document. Checked before every
        publication: a deleted, reopened or re-ingested document rejects the
        task that used to be reading it."""
        if self.cancel.is_set():
            return False
        if SESSIONS.get(self.doc_id) is not self.document:
            return False
        return self.document.ingestion.get("task_id") == self.task_id


class IngestionManager:
    """At most one live ingestion per document, and a bounded number overall.

    One at a time by default. OCR is CPU-bound and this runs behind somebody
    listening to page one, and a background scan must
    not take the processor that narration needs.
    """

    def __init__(self, workers=1):
        self._lock = threading.Lock()
        self._tasks = {}
        self._futures = set()
        self._pool = ThreadPoolExecutor(max_workers=workers,
                                        thread_name_prefix="ingest")
        self._stopped = False

    def start(self, task):
        with self._lock:
            if self._stopped:
                return False
            previous = self._tasks.get(task.doc_id)
            self._tasks[task.doc_id] = task
        if previous is not None:
            _revoke(previous)
        self._futures.add(self._pool.submit(self._run, task))
        return True

    def cancel(self, doc_id):
        """Never joins. This is called from an HTTP thread deleting a
        document, and waiting there for a page of OCR to finish would hold the
        request open for seconds."""
        with self._lock:
            task = self._tasks.pop(doc_id, None)
        if task is not None:
            _revoke(task)

    def active(self, doc_id):
        with self._lock:
            return doc_id in self._tasks

    def _run(self, task):
        try:
            _ingest(task)
        except Exception as e:
            # Whatever happened, the prefix stays readable. That is the whole
            # contract: a failure here reduces how much of the book exists,
            # never how much of it works.
            try:
                _settle(task, "partial", note=f"{type(e).__name__}: {e}")
            except Exception:
                pass
        finally:
            with self._lock:
                if self._tasks.get(task.doc_id) is task:
                    self._tasks.pop(task.doc_id, None)

    def shutdown(self, timeout=5.0):
        with self._lock:
            self._stopped = True
            tasks = list(self._tasks.values())
            self._tasks.clear()
            futures = list(self._futures)
        for task in tasks:
            _revoke(task)
        self._pool.shutdown(wait=False, cancel_futures=True)
        # A page already inside Tesseract cannot be interrupted, so give it a
        # bounded moment to notice and stop. Bounded because a stuck page must
        # not be able to hold the whole application open; whatever it had is
        # already saved, and the OCR cache makes picking it up again cheap.
        if futures:
            concurrent.futures.wait(futures, timeout=timeout)


def _revoke(task):
    """Take a task's claim on its document away, under the session lock.

    Setting the cancel flag alone leaves a window: a publish already past its
    own check appends anyway, and then declines to warm what it just appended,
    which leaves blocks with no audio and a replacement task about to write
    the same pages again. Clearing the document's task id closes it, because
    append_content re-checks that id while holding the lock it appends under.
    """
    task.cancel.set()
    doc = SESSIONS.get(task.doc_id)
    if doc is None or doc is not task.document:
        return
    record = doc.ingestion
    if record.get("task_id") != task.task_id:
        return
    try:
        doc.settle_ingestion({**record, "task_id": None})
    except session.ContentError:
        pass


INGESTION = IngestionManager()


def _ingestion_state(task, state, **fields):
    """The ingestion record as it stands, with `fields` applied."""
    plan = task.plan
    record = {"task_id": task.task_id, "state": state,
              "pages_total": len(plan.selected_pages),
              "pages_processed": task.ordinal,
              "pages_succeeded": task.succeeded,
              "next_ordinal": task.ordinal,
              "failed_pages": task.failed[:MAX_FAILED_PAGES],
              "failed_page_count": len(task.failed),
              "source_fingerprint": plan.source_fingerprint,
              "updated_at": session.now_iso()}
    if task.ordinal:
        record["available_through_page"] = plan.selected_pages[task.ordinal - 1]
    if task.carry is not None:
        # A paragraph held back to join the next page lives only in this
        # thread's memory. Saved with the ordinal that describes it, so an
        # unclean stop cannot drop the last paragraph of a page and leave the
        # resume starting cleanly after it.
        record["carry"] = [task.carry[0], task.carry[1]]
    record.update(fields)
    return record


def _publish(task, pairs, state="running"):
    """Append one page's blocks, then persist and warm outside the lock.

    Returns the number of blocks the document now has, or None if the task no
    longer owns the document - which is the answer to "should I keep reading".
    """
    doc = task.document
    if not task.live():
        return None
    blocks, pages = extract.blockify_pages(pairs)
    ingestion = _ingestion_state(task, state)
    try:
        snapshot = doc.content_snapshot()
        before = len(snapshot.blocks)
        if blocks:
            snapshot = doc.append_content(snapshot.revision, blocks, pages,
                                          ingestion=ingestion,
                                          task_id=task.task_id)
        else:
            # A page of pure figure, a blank verso, a page Tesseract could not
            # read: progress moves, content does not.
            snapshot = doc.settle_ingestion(ingestion, task_id=task.task_id)
    except session.ContentError:
        # Two very different things arrive here. Losing the claim on the
        # document - cancelled, deleted, handed to another task - is the
        # ordinary way a task ends, and it stops quietly. Content the document
        # would not accept is a bug, and it has to surface: swallowing it
        # would leave the document stuck "running" and the browser polling a
        # scan that is never going to finish.
        if task.live():
            raise
        return None
    if blocks:
        # Outside the session lock, in the documented order: session, then
        # coordinator. The coordinator reads text back through the locked
        # text_at(), so a queued new index either sees a whole block or none.
        COORD.extend(doc.id, before, len(snapshot.blocks),
                     doc.session.snapshot()["voice"])
    return len(snapshot.blocks)


def _settle(task, state, note=None):
    """Publish a terminal ingestion state and let the additive analyses run.

    Chapter detection and figure indexing both take a fixed snapshot of blocks
    and pages, so starting them while the book is still growing would publish
    offsets into text that has since moved. They wait for this.
    """
    doc = task.document
    if not task.live():
        # A task that has been cancelled or replaced does not get to have the
        # last word on a document somebody else is now reading - least of all
        # to mark it partial and switch its indexes off.
        return
    record = _ingestion_state(task, state)
    if note:
        record["note"] = note[:200]
    try:
        doc.settle_ingestion(record, task_id=task.task_id)
    except session.ContentError:
        return
    # The book's own problem sets were found in the prefix; the rest of them
    # only exist now. A pure-text scan, and a set that straddles the boundary
    # between two appends is only recognisable once both halves are there -
    # which is exactly now.
    try:
        doc.practice = practice.find(list(doc.content_snapshot().blocks))
    except Exception:
        pass                     # additive, and silent when it finds nothing
    _save_doc(task.doc_id)
    if state == "complete" and doc.source:
        if not doc.chapters:
            _ANALYSIS.submit("chapters", _detect_chapters, task.doc_id)
        _ANALYSIS.submit("figures", _index_figures, task.doc_id)
    elif doc.source:
        # A partial book keeps its indexes quietly absent rather than pointing
        # at half a document. Nothing is shown, nothing is broken.
        doc.set_chapter_detection({"state": "none", "method": None})
        doc.set_figure_index({"state": "none"})
        _save_doc(task.doc_id)


def _furniture(task):
    """What counts as a running head, from the pages read so far.

    New evidence only ever affects pages not yet committed. A head that
    survived onto page one because page one was the only page anyone had seen
    stays there; removing it later would mean renumbering every block after
    it, which is the one thing that must never happen. An occasional spoken
    "Chapter 3" at the very start is a far smaller price.
    """
    return readaloud.find_furniture(list(task.observed), 0.25,
                                    min_pages=len(task.observed))


def _convert_page(task, page, lines):
    """One page's lines to publishable (paragraph, page) pairs, carrying an
    unfinished paragraph forward to join whatever comes next."""
    result = readaloud.paragraphs_for_page(lines, page, _furniture(task),
                                           task.plan.math, task.plan.drop_captions)
    ready, task.carry = readaloud.merge_paragraphs(
        [(text, page) for text in result.paragraphs], task.carry, flush=False)
    return ready


def _read_pages(task, indices, publish=True):
    """Read `indices` in order, publishing each page's text as it arrives.

    With publish=False the pages are read first and converted afterwards, and
    the difference matters: a running head is only recognisable once it has
    been seen repeat, so converting page one before page two has been read
    keeps the head. That is the right trade in the background - waiting for
    the next page before showing this one would defeat the point - but the
    opening pages are read as a batch anyway, so they may as well learn from
    each other first.

    Returns (pairs, stopped). `pairs` is filled only when publish is False,
    which is how the opening prefix is collected before there is a document to
    append to.
    """
    plan = task.plan
    collected, gathered = [], []
    for page, lines, error in task.source(
            plan.path, indices, cancelled=task.cancel.is_set):
        task.ordinal += 1
        if error is not None or lines is None:
            if error != "cancelled":
                task.failed.append(page)
            if task.cancel.is_set():
                return collected, True
            # Page k failing must never hide page k+1. Flush whatever was
            # being carried across the gap - it has nothing to join to now,
            # and a carry left in `collected` on the publishing path would be
            # returned to a caller that discards it, losing a paragraph.
            if task.carry is not None:
                if publish:
                    if _publish(task, [task.carry]) is None:
                        return collected, True
                else:
                    collected.append(task.carry)
                task.carry = None
            continue
        if task.cancel.is_set():
            # Tesseract cannot be interrupted mid-page, so a cancel that
            # landed while this page was being read arrives with the page.
            # Publishing it anyway would put text into a document nobody is
            # reading any more.
            return collected, True
        task.succeeded += 1
        task.observed.append(lines)
        if not publish:
            gathered.append((page, lines))
            continue
        if readaloud.source_fingerprint(plan.path) != plan.source_fingerprint:
            # Checked per page, not per batch: a page takes a second or two to
            # read, and the file can be replaced inside that window. An
            # os.stat is nothing beside the OCR it guards.
            return collected, True
        if _publish(task, _convert_page(task, page, lines)) is None:
            return collected, True
        # Debounced rather than per page: a 900-page book would otherwise
        # rewrite its whole record hundreds of times, and the OCR cache means
        # a crash costs re-reading a page or two, not re-OCRing them. The
        # ordinal and the blocks it describes are written in one record, so
        # what is on disk can never claim more progress than it has text for.
        now = time.monotonic()
        if now - task.saved_at >= INGEST_SAVE_INTERVAL:
            task.saved_at = now
            _save_doc(task.doc_id)
    for page, lines in gathered:
        collected.extend(_convert_page(task, page, lines))
    return collected, task.cancel.is_set()


def _ingest(task):
    """Read the rest of the book, a page at a time, behind the reader."""
    plan = task.plan
    remaining = plan.selected_pages[task.ordinal:]
    for start in range(0, len(remaining), FURNITURE_SAMPLE):
        chunk = remaining[start:start + FURNITURE_SAMPLE]
        if not task.live():
            return
        if readaloud.source_fingerprint(plan.path) != plan.source_fingerprint:
            # The file was replaced or edited underneath us. Never append text
            # from a different document to this one. _read_pages re-checks this
            # per page as well; this is the one that reports it.
            _settle(task, "partial", note="the source file changed")
            return
        _pairs, stopped = _read_pages(task, chunk)
        if stopped:
            return
    if task.carry is not None and task.live():
        _publish(task, [task.carry])
        task.carry = None
    if task.live():
        _settle(task, "complete" if not task.failed else "partial")


# A placeholder so an IngestionTask can be built before the document exists -
# the prefix has to be read first, and the task is what reads it.
class _PendingDoc:
    id = None
    ingestion = {}

    @staticmethod
    def content_snapshot():
        raise session.ContentError("no document yet")


_PENDING_DOC = _PendingDoc()


def _open_progressive(job_id, path, title, options, plan, source=None):
    """Open a scanned book from its first readable pages and grow it behind.

    Returns the document, or None if no prefix could be read - in which case
    the caller falls back to converting the whole book, exactly as before.
    Nothing here is allowed to be a precondition for opening the file.
    """
    task = IngestionTask("ocr-" + uuid.uuid4().hex[:8], _PENDING_DOC, plan,
                         source=source)
    pairs = []
    total = len(plan.selected_pages)
    while task.ordinal < total and not pairs:
        # Two pages, then one at a time until something readable turns up.
        # A scanned textbook's first page is quite often a title plate, and a
        # blank one must not mean "this book has no text in it".
        head = plan.selected_pages[task.ordinal:task.ordinal + INGEST_PROBE_PAGES]
        if not head:
            break
        pairs, _stopped = _read_pages(task, head, publish=False)
        OPEN_JOBS.update(job_id, done=task.ordinal, total=total)
        if pairs:
            break
        if task.carry is not None:
            # Holding the carry back would leave nothing to read at all. A
            # first page the reader can hear outranks a perfect join across
            # its bottom edge.
            pairs, task.carry = [task.carry], None
    if not pairs:
        return None

    blocks, block_pages = extract.blockify_pages(pairs)
    if not blocks:
        return None
    source = {"path": path, "options": options}
    state = "running" if task.ordinal < total else "complete"
    doc = new_doc(blocks, title, source=source, block_pages=block_pages,
                  ingestion=_ingestion_state(task, state))
    task.document = doc
    task.doc_id = doc.id
    if state == "running":
        INGESTION.start(task)
    return doc




def _run_open(job_id, path, title, options):
    """Runs on its own thread - a scanned PDF's OCR pass is exactly the kind
    of thing that shouldn't hold an HTTP request open (see extract.from_pdf's
    progress callback). Mirrors _run_export's job-status pattern so the client
    polls the same way it already does for exports."""
    options = {k: v for k, v in options.items() if k != "progress"}
    block_pages = None
    if path.lower().endswith(".pdf"):
        # A fully scanned book is readable from its first page long before it
        # is readable in full. Everything about this is additive: if the plan
        # cannot be made, or no prefix can be read, it falls through to the
        # whole-document conversion below and behaves exactly as it always did.
        try:
            plan = readaloud.plan_ingestion(path, pages=options.get("pages"),
                                            math=options.get("math", "skip"))
        except Exception:
            plan = None
        if plan is not None and plan.progressive:
            try:
                doc = _open_progressive(job_id, path, title, options, plan)
            except Exception:
                doc = None
            if doc is not None:
                snapshot = doc.content_snapshot()
                OPEN_JOBS.update(
                    job_id, status="done",
                    doc={"id": doc.id, "title": title,
                         "blocks": list(snapshot.blocks),
                         "practice": doc.practice,
                         "block_pages": list(snapshot.block_pages or []),
                         "content_revision": snapshot.revision,
                         "ingestion": snapshot.ingestion,
                         "page_runs": _doc_page_runs(doc),
                         "has_pages": True,
                         "page_count": PDFS.page_count(path)})
                return
    try:
        def progress(done, total):
            OPEN_JOBS.update(job_id, done=done, total=total)
        if path.lower().endswith(".pdf"):
            try:
                blocks, block_pages = extract.from_pdf_with_pages(path, **options)
            except Exception:
                blocks = extract.load(path, progress=progress, **options)
        else:
            blocks = extract.load(path, progress=progress, **options)
    except SystemExit as e:
        OPEN_JOBS.update(job_id, status="error", error=str(e))
        return
    except Exception as e:
        OPEN_JOBS.update(job_id, status="error", error=f"{type(e).__name__}: {e}")
        return

    if not blocks:
        OPEN_JOBS.update(job_id, status="error",
                         error="No readable text found. If this is a "
                               "scanned PDF, OCR may be needed.")
        return

    source = {"path": path, "options": options} if path.lower().endswith(".pdf") else None
    page_count = PDFS.page_count(path) if source else 0
    doc = new_doc(blocks, title, source=source, block_pages=block_pages)
    OPEN_JOBS.update(job_id, status="done",
                     doc={"id": doc.id, "title": title, "blocks": blocks,
                          "practice": doc.practice, "block_pages": block_pages,
                          "content_revision": doc.content_snapshot().revision,
                          "ingestion": doc.ingestion,
                          "page_runs": _page_runs(_selected_pages(
                              path, options, page_count)) if source else None,
                          "has_pages": bool(source), "page_count": page_count})


def _asset_key(doc_id, i, voice):
    """Names one synthesized clip: this document, this block, this voice.
    Both halves of a block (its timings and its audio) carry it, so the
    player can refuse to pair a clip with timings from a different voice -
    the one contract that must never drift."""
    return f"{doc_id}:{i}:{voice or ''}"


def _requested_voice(q):
    """An explicit voice on a playback request. The player names the voice it
    is loading a slot for rather than leaving the server to use whatever the
    document happens to be set to when the request lands - with two slots in
    flight at once, those are not always the same thing. Anything not in the
    running engine's catalogue is ignored, not an error."""
    voice = q.get("voice", [None])[0]
    if not voice:
        return None
    if voice not in engines.voice_ids_for(ENGINE.name if ENGINE else None):
        return None
    return voice


def _requested_priority(q):
    """How urgent this playback request is.

    The player asks for three blocks around the one it is playing: the one it
    needs now, the standby slot that makes the next paragraph start without a
    gap, and one more of lookahead. All three used to arrive as
    PRIORITY_INTERACTIVE, so a seek - the request the user is most visibly
    waiting on - queued behind two blocks of speculation nobody was listening
    to yet. `prefetch=1` says "wanted soon, but nobody is waiting", which is
    the honest description and puts it in its own tier.

    Absent or malformed means interactive: an older cached copy of app.js, or
    anything else that does not know about the flag, keeps exactly the
    behaviour it had."""
    return (coordinator.PRIORITY_PREFETCH
            if (q.get("prefetch") or [""])[0] in ("1", "true", "yes")
            else coordinator.PRIORITY_INTERACTIVE)


def _figure_manifest(doc_id, doc):
    """The public figure manifest for a document, from memory if the indexer
    has published one this run, from the sidecar cache if it published one in
    a previous run, and otherwise the quiet empty shape."""
    manifest = FIGURE_MANIFESTS.get(doc_id)
    if manifest is None:
        fp = (doc.figure_index or {}).get("fingerprint")
        if fp:
            manifest = figures.load_manifest(FIGURE_CACHE_DIR, fp)
            if manifest is not None:
                FIGURE_MANIFESTS[doc_id] = manifest
    if manifest is None:
        state = (doc.figure_index or {}).get("state") or "none"
        return {"state": "pending" if state == "pending" else "none",
                "revision": None, "items": [], "refs": {}}
    return manifest


def _public_ingestion(content):
    """What the browser is told about a document still being read.

    Deliberately a subset: task ids, source fingerprints and next_ordinal are
    bookkeeping for resuming a scan, not anything a reader needs, and the
    smaller shape is one less thing to keep in step. A document that was never
    progressively read reports a plain "complete", so the client has one thing
    to check rather than two.
    """
    record = content.ingestion or {}
    state = record.get("state", "complete")
    out = {"state": state, "revision": content.ingestion_revision}
    for key in ("pages_total", "pages_processed", "pages_succeeded",
                "available_through_page", "failed_page_count", "note"):
        if key in record:
            out[key] = record[key]
    out["failed_pages"] = list(record.get("failed_pages") or [])
    return out


def _doc_summary(doc):
    snap = doc.session.snapshot()
    content = doc.content_snapshot()
    exported = _is_exported(doc, content)
    out = {"id": doc.id, "title": doc.title, "blocks": len(content.blocks),
           "ingestion": _public_ingestion(content),
           "ready": AUDIO_CACHE.ready(doc.id), "voice": snap["voice"],
           "cached_voices": {(v or ""): c
                             for v, c in COORD.voice_counts(doc.id).items()},
           "last_opened": snap["last_opened"], "position": snap["position"],
           "chapters": len(doc.chapters),
           "exported": exported,
           # Which voice is already on disk, so the library can offer to
           # replace it by name. Session state cannot answer this after a
           # reload; the index entry can. "" when nothing is recorded, and
           # the client renders nothing rather than guessing.
           "exported_voice": ((_export_entry(doc, content) or {}).get("voice")
                              or "") if exported else ""}
    out.update(doc.progress())
    return out


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass                                  # keep the console for real output

    # ------------------------------------------------------------ helpers
    def _send(self, code, body=b"", ctype="application/json", extra=None,
              cache="no-store"):
        if isinstance(body, str):
            body = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", cache)
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj, code=200):
        self._send(code, json.dumps(obj), "application/json")

    def _is_local_client(self):
        client_ip = self.client_address[0]
        return client_ip in ("127.0.0.1", "::1", "localhost")

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, X-Requested-With, Range")
        self.end_headers()

    # ------------------------------------------------------------- routes
    def do_GET(self):
        url = urlparse(self.path)
        path, q = url.path, parse_qs(url.query)

        # The desktop API is loopback-only; nothing is served over the LAN.
        if not self._is_local_client():
            return self._send(403, b"Access denied: desktop API is loopback-only",
                              "text/plain")

        if path in ("/", "/index.html"):
            return self._file(os.path.join(WEB, "index.html"))
        if path in ("/triage", "/triage.html"):
            return self._file(os.path.join(WEB, "triage.html"))
        if path.startswith("/web/"):
            return self._file(os.path.join(WEB, os.path.basename(path)))

        if path == "/api/docs":
            out = [_doc_summary(d) for d in SESSIONS.all()]
            return self._json({"docs": out,
                                "engine": ENGINE.name if ENGINE else None})

        if path == "/api/recent":
            limit = int(q.get("limit", ["3"])[0])
            # Order first, summarize second. A summary walks the document's
            # per-block runtimes to work out how far through it you are, and
            # building one for every book in the library to then throw all but
            # three away was the whole cost of a shelf that shows three.
            docs = sorted(SESSIONS.all(),
                          key=lambda d: d.session.snapshot()["last_opened"] or "",
                          reverse=True)
            return self._json({"docs": [_doc_summary(d) for d in docs[:limit]]})

        if path == "/api/voices":
            engine_name = ENGINE.name if ENGINE else None
            # First look at the gallery is also the latest moment worth
            # starting the sample-clip warmer; idempotent, so the startup call
            # and this one cannot produce two warmers.
            _prewarm_previews()
            return self._json({"engine": engine_name,
                                "voices": engines.voices_for(engine_name),
                                "default": SETTINGS.get("default_voice"),
                                # Which sample clips are already made. Not
                                # used by the player - it is how smoke.py can
                                # tell pre-warming happened rather than time a
                                # request and hope.
                                "previews_ready": PREVIEWS.ready()})

        if path == "/api/doc":
            doc_id = q.get("doc", [None])[0]
            doc = SESSIONS.get(doc_id)
            if doc is None:
                return self._json({"error": "no such document"}, 404)
            doc.session.touch()
            _save_doc(doc_id)
            snap = doc.session.snapshot()
            # This is the document the reader has open, so its warm window
            # outranks every other book the library is warming in the
            # background. Set it before warming, so the first window already
            # goes in at the active tier.
            COORD.set_active_document(doc_id)
            _warm(doc_id, start=snap["position"]["block"])   # reprioritize toward the resume point
            source_path = doc.source.get("path") if doc.source else None
            page_count = PDFS.page_count(source_path) if source_path else 0
            has_pages = page_count > 0
            # Which voices this document already has audio for, and how much
            # of it. Trying voices mid-read leaves a book split across several,
            # and the export dialog uses this to say which one exports fastest.
            # A block cached under the engine default (voice None) is reported
            # under "", the same value the picker's default option carries.
            cached_voices = {(v or ""): c
                             for v, c in COORD.voice_counts(doc_id).items()}
            content = doc.content_snapshot()
            return self._json({"id": doc_id, "title": doc.title,
                                "blocks": list(content.blocks),
                                "voice": snap["voice"], "position": snap["position"],
                                "cached_voices": cached_voices,
                                "practice": doc.practice, "chapters": doc.chapters,
                                "chapter_detection": doc.chapter_detection,
                                "figure_index": doc.figure_index,
                                "generation": snap["generation"],
                                "block_pages": (list(content.block_pages)
                                                if content.block_pages is not None else None),
                                "content_revision": content.revision,
                                "ingestion": _public_ingestion(content),
                                "has_pages": has_pages,
                                "page_runs": _doc_page_runs(doc),
                                "page_count": page_count})

        if path == "/api/doc/blocks":
            # The suffix of a document that is still being read off a scanner.
            # Idempotent by design: it answers "everything from block N
            # onward", not "what changed since you last asked", so a poll that
            # is retried, duplicated or arrives out of order returns the same
            # thing and no update can fall between two of them.
            doc_id = q.get("doc", [None])[0]
            doc = SESSIONS.get(doc_id)
            if doc is None:
                return self._json({"error": "no such document",
                                    "code": "not_found"}, 404)
            raw = q.get("from", [None])[0]
            try:
                start = int(raw)
                if start < 0:
                    raise ValueError(raw)
            except (TypeError, ValueError):
                return self._json({"error": "invalid from",
                                    "code": "invalid_cursor"}, 400)
            content = doc.content_snapshot()
            if start > len(content.blocks):
                # The browser is holding blocks this document does not have.
                # That means it is looking at a different document under the
                # same id - reopened, or replaced - and the honest answer is
                # to start again rather than to splice a suffix onto text that
                # was never there.
                return self._json({"error": "content cursor is ahead of the document",
                                    "code": "content_reset", "reload": True}, 409)
            pages = content.block_pages
            return self._json({"doc": doc_id, "from": start,
                                "revision": content.revision,
                                "blocks": list(content.blocks[start:]),
                                "block_pages": (list(pages[start:])
                                                if pages is not None else None),
                                "practice": doc.practice,
                                "ingestion": _public_ingestion(content)})

        if path == "/api/page":
            doc_id = q.get("doc", [None])[0]
            if not doc_id:
                return self._send(400, b"missing doc param", "text/plain")
            doc = SESSIONS.get(doc_id)
            if doc is None or not doc.source or not doc.source.get("path"):
                return self._send(404, b"no such document or not a pdf", "text/plain")
            pdf_path = doc.source["path"]
            if not os.path.exists(pdf_path):
                return self._send(404, b"pdf file missing on disk", "text/plain")
            try:
                page_idx = int(q.get("page", [0])[0])
                dpi = int(q.get("dpi", [150])[0])
            except (ValueError, TypeError):
                return self._send(400, b"invalid page or dpi parameter", "text/plain")
            dpi = max(72, min(300, dpi))
            # A page outside the selection is not part of this document.
            # Enforced here as well as in the player because the browser
            # caches app.js: a tab still running yesterday's copy must not be
            # able to walk a 980-page scan a page-render at a time.
            chosen = _doc_selected_pages(doc)
            if chosen is not None and page_idx not in set(chosen):
                return self._send(404, b"page is outside this document",
                                  "text/plain")
            try:
                png_bytes = PDFS.render(pdf_path, page_idx, dpi)
            except Exception as e:
                return self._send(500, f"page render failed: {e}".encode(), "text/plain")
            if png_bytes is None:
                return self._send(404, b"page index out of range", "text/plain")
            etag = '"%s"' % hashlib.sha1(png_bytes).hexdigest()[:16]
            if self.headers.get("If-None-Match") == etag:
                return self._not_modified(etag, "private, max-age=86400")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png_bytes)))
            self.send_header("ETag", etag)
            # Private, not public: this is a page of the user's own book.
            self.send_header("Cache-Control", "private, max-age=86400")
            self.end_headers()
            self.wfile.write(png_bytes)
            return

        if path == "/api/chapters":
            doc_id = q.get("doc", [None])[0]
            doc = SESSIONS.get(doc_id)
            if doc is None:
                return self._json({"error": "no such document"}, 404)
            return self._json({"chapters": doc.chapters,
                               "chapter_detection": doc.chapter_detection})

        if path == "/api/figures":
            # Quiet by design: an unknown document, a document with no PDF
            # behind it, or an index that found nothing all answer the same
            # shape. The player draws no panel and shows no error for "none".
            doc_id = q.get("doc", [None])[0]
            doc = SESSIONS.get(doc_id)
            if doc is None:
                return self._json({"error": "no such document"}, 404)
            return self._json(_figure_manifest(doc_id, doc))

        if path == "/api/figure":
            doc_id = q.get("doc", [None])[0]
            fig_id = q.get("id", [""])[0]
            doc = SESSIONS.get(doc_id)
            if doc is None:
                return self._send(404, b"not found", "text/plain")
            # Membership in the manifest is the *only* thing that makes an id
            # servable. No path, page, xref or filename from the browser ever
            # selects a file here, so a traversal-shaped id is not a special
            # case - it simply isn't in the manifest, and 404s like any other
            # id that names nothing.
            manifest = _figure_manifest(doc_id, doc)
            item = next((it for it in manifest.get("items") or []
                         if it.get("id") == fig_id), None)
            fp = (doc.figure_index or {}).get("fingerprint")
            if item is None or not fp:
                return self._send(404, b"not found", "text/plain")
            try:
                with open(figures.asset_path(FIGURE_CACHE_DIR, fp, fig_id), "rb") as f:
                    png = f.read()
            except OSError:
                return self._send(404, b"not found", "text/plain")
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(png)))
            self.send_header("ETag", f'"{fig_id}"')
            # Content-addressed id: the bytes behind it can never change, so
            # the browser may keep it as long as it likes. Private, because
            # this is a page of the user's own book.
            self.send_header("Cache-Control", "private, max-age=31536000, immutable")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(png)
            return

        if path == "/api/block":
            doc_id = q.get("doc", [None])[0]
            i = int(q.get("i", ["0"])[0])
            doc = SESSIONS.get(doc_id)
            if doc is None or not (0 <= i < len(doc.blocks)):
                return self._json({"error": "no such document or block"}, 404)
            voice = _requested_voice(q)
            actual_voice = voice if voice is not None else doc.session.snapshot()["voice"]
            try:
                _, _, words = synth_block(doc_id, i, voice=actual_voice,
                                          priority=_requested_priority(q))
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 500)
            snap = doc.session.snapshot()
            # Timings and audio arrive over two separate responses, and the
            # player joins them. These fields are how it can prove the two
            # halves came from the same synthesis rather than assuming it:
            # a voice switch between the two requests changes asset_key.
            return self._json({"i": i, "words": [asdict(w) for w in words],
                               "voice": actual_voice,
                               "asset_key": _asset_key(doc_id, i, actual_voice),
                               "session_generation": snap["generation"]})

        if path == "/api/audio":
            doc_id = q.get("doc", [None])[0]
            i = int(q.get("i", ["0"])[0])
            doc = SESSIONS.get(doc_id)
            if doc is None or not (0 <= i < len(doc.blocks)):
                return self._send(404, b"", "text/plain")
            voice = _requested_voice(q)
            actual_voice = voice if voice is not None else doc.session.snapshot()["voice"]
            try:
                audio, mime, _ = synth_block(doc_id, i, voice=actual_voice,
                                             priority=_requested_priority(q))
            except Exception as e:
                return self._send(500, str(e), "text/plain")
            return self._send(200, audio, mime, extra={
                "X-Talkify-Doc": doc_id,
                "X-Talkify-Block": str(i),
                "X-Talkify-Voice": actual_voice or "",
                "X-Talkify-Asset-Key": _asset_key(doc_id, i, actual_voice)})

        if path == "/api/export/status":
            job_id = q.get("job", [None])[0]
            job = EXPORT_JOBS.get(job_id)
            if job is None:
                return self._json({"error": "no such export job"}, 404)
            return self._json({k: v for k, v in job.items() if k != "json_path"})

        if path == "/api/open/status":
            job_id = q.get("job", [None])[0]
            job = OPEN_JOBS.get(job_id)
            if job is None:
                return self._json({"error": "no such open job"}, 404)
            return self._json(job)

        if path == "/api/export/download":
            job_id = q.get("job", [None])[0]
            job = EXPORT_JOBS.get(job_id)
            if job is not None and job.get("status") == "done":
                if "files" in job:
                    try:
                        idx = int(q.get("file", ["0"])[0])
                    except ValueError:
                        return self._send(404, b"no such file", "text/plain")
                    if not (0 <= idx < len(job["files"])):
                        return self._send(404, b"no such file", "text/plain")
                    file_path = job["files"][idx]["path"]
                else:
                    file_path = job["path"]
            else:
                # No job id, or one this process no longer remembers. The
                # library offers a download for anything already exported and
                # that outlives the job which made it - a restart, or just a
                # day later - so the package index serves the same files by
                # document instead.
                file_path = _exported_file(q.get("doc", [None])[0],
                                           q.get("file", [None])[0])
                if file_path is None:
                    return self._send(404, b"not ready", "text/plain")
            if not os.path.exists(file_path):
                return self._send(404, b"no such file", "text/plain")
            # Copied out in chunks rather than read whole: a book-length MP3
            # is the one response big enough that buffering it would undo the
            # streaming export it came from.
            self.send_response(200)
            self.send_header("Content-Type", "audio/mpeg")
            self.send_header("Content-Length", str(os.path.getsize(file_path)))
            self.send_header("Content-Disposition",
                              f'attachment; filename="{os.path.basename(file_path)}"')
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            with open(file_path, "rb") as f:
                shutil.copyfileobj(f, self.wfile, 64 * 1024)
            return

        if path == "/api/ai/status":
            # Which model would answer, and whether it is on this machine.
            # The player hides every AI affordance when this says nothing is
            # available, and names the provider in the panel when something
            # is - the reader should always know when a passage is leaving
            # the machine.
            return self._json(ai.provider_info())

        if path == "/api/diagnose":
            # A debug/triage tool, not part of the reading path - see
            # web/triage.html. Deliberately synchronous: it's meant for a
            # handful of pages at a time to audit extraction quality, not
            # a whole book, so it doesn't need the job/poll machinery
            # /api/open uses for OCR that can take minutes.
            file_path = os.path.expanduser(q.get("path", [""])[0])
            if not file_path or not os.path.exists(file_path):
                return self._json({"error": f"no such file: {file_path}"}, 400)
            pages = q.get("pages", [None])[0]
            try:
                result = readaloud.diagnose_pages(file_path, pages=pages)
            except Exception as e:
                return self._json({"error": f"{type(e).__name__}: {e}"}, 400)
            return self._json({"pages": result})

        if path == "/api/voice/preview":
            voice = q.get("voice", [None])[0]
            engine_name = ENGINE.name if ENGINE else None
            if not voice or voice not in engines.voice_ids_for(engine_name):
                return self._send(404, b"unknown voice", "text/plain")
            catalogue = engines.voices_for(engine_name)
            label = next(v["label"] for v in catalogue if v["id"] == voice)
            try:
                (audio, mime), cached = PREVIEWS.get_with_status(
                    voice, lambda: _make_preview(voice, label))
            except Exception as e:
                return self._send(500, str(e), "text/plain")
            # Whether this one came out of memory or had to be synthesized.
            # The browser ignores it; it is how smoke.py can tell that
            # pre-warming actually happened rather than timing a request and
            # hoping.
            return self._send(200, audio, mime,
                              extra={"X-Preview-Cache": "hit" if cached else "miss"})

        return self._send(404, b"not found", "text/plain")

    def do_POST(self):
        url = urlparse(self.path)
        if not self._is_local_client():
            return self._send(403, b"Access denied: desktop API is loopback-only",
                              "text/plain")

        if url.path == "/api/upload":
            return self._upload()
        if url.path == "/api/rename":
            return self._rename()
        if url.path == "/api/voice":
            return self._voice()
        if url.path == "/api/position":
            return self._position()
        if url.path == "/api/delete":
            return self._delete()
        if url.path == "/api/export":
            return self._export()
        if url.path == "/api/explain":
            return self._explain()
        if url.path != "/api/open":
            return self._send(404, b"not found", "text/plain")

        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")

        if payload.get("text"):
            blocks = extract.split_blocks(payload["text"])
            title = payload.get("title") or "Pasted text"
            doc = new_doc(blocks, title, source=None)
            return self._json({"id": doc.id, "title": title, "blocks": blocks,
                                "practice": doc.practice})

        path = os.path.expanduser(payload.get("path", ""))
        if not os.path.exists(path):
            return self._json({"error": f"no such file: {path}"}, 400)
        options = payload.get("options") or {}
        title = os.path.basename(path)

        # A PDF may need OCR - potentially minutes for a real scanned book -
        # so it gets a job the client polls, same as export already does.
        # Anything else (txt, epub) is fast enough to stay synchronous.
        if path.lower().endswith(".pdf"):
            job_id = OPEN_JOBS.create(status="running", done=0, total=0)
            threading.Thread(target=_run_open, args=(job_id, path, title, options),
                              daemon=True).start()
            return self._json({"job": job_id})

        try:
            blocks = extract.load(path, **options)
        except SystemExit as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        if not blocks:
            return self._json({"error": "No readable text found."}, 400)

        doc = new_doc(blocks, title, source=None)
        return self._json({"id": doc.id, "title": title, "blocks": blocks,
                            "practice": doc.practice})

    def _upload(self):
        """Take the file's bytes straight from the picker, so nobody has to
        type a Windows path."""
        import tempfile
        name = unquote(self.headers.get("X-Filename", "upload.txt"))
        pages = unquote(self.headers.get("X-Pages", "")).strip()
        n = int(self.headers.get("Content-Length", 0))
        if n <= 0:
            return self._json({"error": "empty upload"}, 400)
        if n > 400 * 1024 * 1024:
            return self._json({"error": "file too large"}, 400)

        tmpdir = os.path.join(tempfile.gettempdir(), "talkify")
        os.makedirs(tmpdir, exist_ok=True)
        path = os.path.join(tmpdir, os.path.basename(name))
        # Straight from the socket to the file, a megabyte at a time. The
        # previous version built the whole upload in memory with `data +=
        # chunk`, which reallocates and copies everything received so far on
        # every chunk: a 300MB scanned textbook - the exact case this project
        # is written around - meant tens of gigabytes of memcpy before a byte
        # was written, and held the file twice over in RAM while it did.
        with open(path, "wb") as f:
            left = n
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        title = os.path.basename(name)

        # Same reasoning as /api/open: a scanned PDF's OCR pass can take
        # minutes, so it gets a pollable job instead of holding this request.
        if path.lower().endswith(".pdf"):
            options = {"pages": pages} if pages else {}
            job_id = OPEN_JOBS.create(status="running", done=0, total=0)
            threading.Thread(target=_run_open, args=(job_id, path, title, options),
                              daemon=True).start()
            return self._json({"job": job_id})

        try:
            blocks = extract.load(path)
        except SystemExit as e:
            return self._json({"error": str(e)}, 400)
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}"}, 400)

        if not blocks:
            return self._json({"error": "No readable text found in that file."}, 400)

        doc = new_doc(blocks, title, source=None)
        return self._json({"id": doc.id, "title": title, "blocks": blocks,
                            "practice": doc.practice})

    def _rename(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        doc_id = payload.get("doc")
        title = (payload.get("title") or "").strip()
        if not title:
            return self._json({"error": "title can't be empty"}, 400)
        doc = SESSIONS.get(doc_id)
        if doc is None:
            return self._json({"error": "no such document"}, 404)
        doc.rename(title)
        _save_doc(doc_id)
        return self._json({"id": doc_id, "title": title})

    def _voice(self):
        """Switch a document's active voice mid-read, or set the config
        default for future documents. Switching invalidates the cache for
        blocks after the one currently playing (the "unread" ones) and
        restarts the background warmer from there under a new generation, so
        the stale walk in the old voice stops instead of racing the new one."""
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        voice = payload.get("voice")
        engine_name = ENGINE.name if ENGINE else None
        if not voice or voice not in engines.voice_ids_for(engine_name):
            return self._json({"error": "unknown voice"}, 400)

        if payload.get("default"):
            SETTINGS.set("default_voice", voice)

        doc_id = payload.get("doc")
        if doc_id:
            doc = SESSIONS.get(doc_id)
            if doc is None:
                return self._json({"error": "no such document"}, 404)
            cur = doc.session.current_block()
            if payload.get("block") is not None:
                try:
                    cur = int(payload["block"])
                except (TypeError, ValueError):
                    return self._json({"error": "block must be an integer"}, 400)
                cur = max(0, min(cur, max(len(doc.blocks) - 1, 0)))
            # Whether the browser is mid-sentence right now decides what
            # happens to the block it is on, and it is the one thing the
            # server cannot know: the <audio> element is the only thing that
            # knows if it is running. Absent (an old client, or a switch made
            # with nothing open), assume paused - the conservative answer,
            # since it only costs one re-synthesis.
            playing = bool(payload.get("playing"))
            if playing:
                # The browser already holds this block's audio and is speaking
                # it in the old voice. Re-making it would delay cur + 1 behind
                # a clip nobody will ever hear, and the stall would land
                # exactly at the chunk boundary the switch is trying to make
                # seamless. Keep it, and start the new voice at the next block.
                keep_through, warm_from = cur, cur + 1
            else:
                # Paused on cur: pressing Play must speak the new voice
                # immediately, so cur is the first thing to drop and the first
                # thing to re-make.
                keep_through, warm_from = cur - 1, cur
            # Switching a document's voice is something you only do to the
            # one you are reading, so it names the active document too.
            COORD.set_active_document(doc_id)

            # Atomic compare-and-set: check expected_generation against current
            # generation under the session lock, and bump coordinator + session
            # generation in one uninterrupted operation.
            expected = payload.get("expected_generation")
            ok, new_or_cur_gen = doc.session.check_and_set_voice(
                voice,
                expected_generation=expected,
                generation_fn=lambda: COORD.bump(doc_id),
            )
            if not ok:
                return self._json({"error": "stale voice change",
                                   "generation": new_or_cur_gen}, 409)

            AUDIO_CACHE.drop_after(doc_id, keep_through)
            # The voice, and only the voice. A switch has no opinion about
            # where the reader is or what has been indexed, and writing a
            # whole record to say one word is how a save starts carrying
            # somebody else's stale field.
            _save_doc_fields(doc_id, "voice")
            # Past blocks are not dropped and are not re-warmed in the new
            # voice: seeking backwards stays instant, and what you already
            # heard keeps the voice you heard it in.
            _warm(doc_id, start=min(warm_from, max(len(doc.blocks) - 1, 0)))
            return self._json({"ok": True, "voice": voice,
                               "generation": new_or_cur_gen,
                               "kept_through": keep_through, "warm_from": warm_from})
        return self._json({"ok": True, "voice": voice})

    def _position(self):
        """Heartbeat from the player: where it is, and how far into the
        current block's audio. Drives both resume-on-reopen and the
        runtime-based progress percentage - see Document.progress().

        A jump of more than one block is a seek (clicking a word further down,
        or a chapter link), and that re-aims the warmer under a new generation
        so work queued around the old position is dropped rather than
        competing with what is about to play. Advancing block by block is not
        a seek and deliberately doesn't bump anything - the warm window ahead
        of the reader is exactly what should survive normal playback."""
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        doc_id = payload.get("doc")
        doc = SESSIONS.get(doc_id)
        if doc is None:
            return self._json({"error": "no such document"}, 404)
        # The heartbeat is the surest signal of all: it only ever comes from
        # the document being played, so it re-asserts the active tier even if
        # the reader has been left open in a second tab.
        COORD.set_active_document(doc_id)
        block = int(payload.get("block", 0))
        previous = doc.session.current_block()
        doc.session.set_position(block, int(payload.get("word", 0)),
                                 float(payload.get("within", 0.0)))
        # Anything that actually changed play state - a paragraph boundary, a
        # seek, pause, stop, leaving the reader, closing the tab - is saved the
        # moment it happens. What is not is the pure drift heartbeat that
        # arrives every three seconds while a paragraph plays: it rewrites the
        # whole document (blocks and all - see storage.LibraryRepository) to
        # move `within` a few seconds, and on a textbook that is megabytes to
        # disk every three seconds for the life of the session. Those are
        # coalesced onto a floor instead, so a resume is still accurate to a
        # few seconds and every boundary is still exact.
        if (payload.get("settled") or block != previous
                or _due_for_save(doc_id)):
            _save_doc(doc_id)
            # The same moments, for the same reasons, keep an exported
            # package's position sidecar current: a paragraph boundary, a
            # seek, a pause, or the ten-second floor - which is exactly the
            # "at most every ten seconds while playing" the commute spec
            # asks of a writer. Silent when this document was never exported.
            _update_export_position(doc_id, doc, block,
                                    int(payload.get("word", 0)),
                                    float(payload.get("within", 0.0)))
        if abs(block - previous) > 1:
            doc.session.set_generation(COORD.bump(doc_id))
            _warm(doc_id, start=block)
        # The generation rides back on every heartbeat, so the player's idea
        # of it stays current without a second request - a seek bumps it here
        # and the very response that acknowledges the seek carries the new
        # number. Without that, the next voice switch would be refused as
        # stale for a reason the user could not possibly have caused.
        return self._json({"ok": True,
                           "generation": doc.session.snapshot()["generation"]})

    def _export(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        doc_id = payload.get("doc")
        mode = payload.get("mode") or "full"
        doc = SESSIONS.get(doc_id)
        if doc is None:
            return self._json({"error": "no such document"}, 404)
        if mode == "chapters" and not doc.chapters:
            return self._json({"error": "no chapters detected for this document"}, 400)
        content = doc.content_snapshot()
        if content.ingestion.get("state") == "running":
            # Exporting now would write however much of the book has been
            # scanned so far under the whole book's filename, and nothing
            # about the resulting file would say so. Refuse and say why; the
            # reader can keep reading in the meantime.
            return self._json({"error": "document is still being read",
                                "code": "ingestion_running", "retryable": True,
                                "ingestion": _public_ingestion(content)}, 409)
        # An empty/absent voice means "whatever the reader is set to", which is
        # what every client sent before the picker existed.
        target_voice = payload.get("voice") or doc.session.snapshot()["voice"]
        job_id = EXPORT_JOBS.create(status="running", done=0,
                                    total=len(content.blocks), voice=target_voice)
        threading.Thread(target=_run_export, args=(doc_id, job_id, mode),
                         kwargs={"voice": target_voice}, daemon=True).start()
        return self._json({"job": job_id})

    def _explain(self):
        # Everything below answers with JSON, including the failures: this is
        # the one endpoint whose backend is a third party over the network,
        # so it has the most ways to go wrong, and the panel has to be able
        # to print what happened rather than an empty box.
        n = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._json({"error": "malformed request"}, 400)
        if not isinstance(payload, dict):
            return self._json({"error": "malformed request"}, 400)
        doc_id = payload.get("doc")
        i = payload.get("block")
        doc = SESSIONS.get(doc_id)
        if doc is None or not isinstance(i, int) or not (0 <= i < len(doc.blocks)):
            return self._json({"error": "no such document or block"}, 400)
        info = ai.provider_info()
        if not info["available"]:
            return self._json({"error": "no explain backend available"}, 503)

        selection = (payload.get("selection") or "").strip()
        question = (payload.get("question") or "").strip()
        warning = None
        if selection:
            # Ground in the actual selected text, not just the paragraph it
            # started in - a selection can span several blocks (the reader
            # only tracks *which* block the selection toolbar opened from,
            # for UI purposes), and quoting it inside the question instead
            # of sending it as the context meant the model only ever saw
            # whatever fraction of it happened to sit in that one paragraph.
            context = selection
            if len(context) > MAX_SELECTION_CHARS:
                context = context[:MAX_SELECTION_CHARS]
                warning = "Only the first part of your selection was sent - it was too long."
            if not question:
                question = "Explain this passage."
        else:
            context = doc.blocks[i]
            if not question:
                question = "Explain this passage."
        try:
            answer, ask_warning = ai.explain(question, context)
            warning = warning or ask_warning
        except Exception as e:
            return self._json({"error": f"{type(e).__name__}: {e}",
                                "provider": info["provider"]}, 502)
        return self._json({"answer": answer, "warning": warning,
                            "provider": info["provider"], "label": info["label"],
                            "local": info["local"]})

    def _delete(self):
        n = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(n) or b"{}")
        doc_id = payload.get("doc")
        removed = SESSIONS.remove(doc_id)
        if removed is None:
            return self._json({"error": "no such document"}, 404)
        # Removed from SESSIONS first, so any OCR page still in flight already
        # fails its identity check by the time it tries to publish. The cancel
        # is what stops the *next* page from being read at all; it never joins,
        # because waiting here for a page of Tesseract would hold this request
        # open for seconds.
        INGESTION.cancel(doc_id)
        COORD.forget(doc_id)          # its queued warm work and its cached audio
        if removed.source and removed.source.get("path"):
            PDFS.forget(removed.source["path"])
        FIGURE_MANIFESTS.pop(doc_id, None)
        with _last_saved_lock:
            _last_saved.pop(doc_id, None)
        LIBRARY.delete(doc_id)
        return self._json({"ok": True})

    def _not_modified(self, etag, cache_control):
        self.send_response(304)
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", cache_control)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def _file(self, path):
        """Serve one file out of web/, from memory when it has not changed.

        `no-cache` rather than `no-store`: the browser must still ask before
        using what it holds, so an edited app.js can never be served stale -
        but when it has not been edited the answer is a 304 with no body
        instead of 96KB down the socket and a re-parse in the tab. Every load
        of the reader used to re-read and resend app.js, style.css and
        index.html in full.
        """
        entry = _static_file(path)
        if entry is None:
            return self._send(404, b"not found", "text/plain")
        body, ctype, etag = entry
        if self.headers.get("If-None-Match") == etag:
            return self._not_modified(etag, "no-cache")
        self._send(200, body, ctype, extra={"ETag": etag}, cache="no-cache")


def main():
    global ENGINE
    ap = argparse.ArgumentParser(description="Read documents aloud, highlighted as spoken.")
    ap.add_argument("file", nargs="?", help="PDF, EPUB or text file to open")
    ap.add_argument("--engine", default="kokoro", choices=list(engines.ENGINES),
                    help="kokoro (offline, default), edge (network), stub (no models)")
    ap.add_argument("--voice", help="engine-specific voice id")
    ap.add_argument("--rate", type=float, default=1.0)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="host interface to bind. The API refuses non-loopback "
                         "clients, so a wider bind exposes a port that serves nothing.")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--pipeline", action="store_true",
                    help="overlap Kokoro and Whisper: keep two blocks in the "
                         "worker so one is being spoken while the other is "
                         "being timed. Off by default because it was measured "
                         "not to pay - see bench_synth.py, and run it before "
                         "turning this on.")
    a = ap.parse_args()

    kw = {"rate": a.rate}
    if a.voice:
        kw["voice"] = a.voice
    # The pronunciation dictionary before the engine, so the warm-up block is
    # spoken with the same table as everything after it.
    _refresh_speech_settings()
    ENGINE = engines.build(a.engine, **kw)
    if hasattr(ENGINE, "submit") and a.pipeline:
        # Opt-in, and measured before being offered at all: on this hardware
        # Kokoro takes 7.9s of a 9.1s block and Whisper 1.2s, so overlapping
        # them can save at most 13% even in theory - and in practice the two
        # runtimes contend for the same cores and it saves nothing. See
        # bench_synth.py. The rest of the pipelined machinery still earns its
        # place at one dispatcher: a seek can now cancel a block that is
        # already inside the worker instead of waiting eight seconds for audio
        # nobody will hear.
        COORD.scale(coordinator.DEFAULT_MAX_INFLIGHT)
    if hasattr(ENGINE, "warm"):
        ENGINE.warm()

    load_library()
    _prewarm_previews()          # gallery samples, in the background, best-effort

    if a.file:
        path = os.path.expanduser(a.file)
        source = {"path": path, "options": {}} if path.lower().endswith(".pdf") else None
        progress = lambda done, total: print(f"  OCR {done}/{total}", end="\r", file=sys.stderr)
        block_pages = None
        if source:
            try:
                blocks, block_pages = extract.from_pdf_with_pages(path)
            except Exception:
                blocks = extract.load(path, progress=progress)
        else:
            blocks = extract.load(path, progress=progress)
        new_doc(blocks, os.path.basename(path), source=source, block_pages=block_pages)

    url = f"http://127.0.0.1:{a.port}/"
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Talkify  engine={a.engine}  {url}")
    print("Ctrl-C to stop.")
    if not a.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        # Ingestion first: it is the only thing still adding to documents, and
        # everything below wants a document that has stopped changing. The OCR
        # cache means whatever it had started is cheap to pick up again.
        INGESTION.shutdown()
        COORD.shutdown()
        if hasattr(ENGINE, "shutdown"):
            ENGINE.shutdown()
        PDFS.close()
        # A drift heartbeat may have been coalesced (see _position); this is
        # where it gets written, so a clean stop never costs anyone their
        # place even mid-paragraph.
        for d in SESSIONS.all():
            try:
                _save_doc(d.id)
            except Exception:
                pass
        LIBRARY.close()


if __name__ == "__main__":
    main()
