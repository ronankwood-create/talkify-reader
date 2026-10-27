"""
The volatile half of Talkify's state: what is open, and where you are in it.

This is the other half of retiring the global STATE dict. Everything here dies
with the process and is rebuilt from the library on the next start; anything
that has to survive lives in storage.py.

    Position          block / word / within-block seconds
    PlaybackSession   one document's live playback state - voice, position,
                      generation counter
    ContentSnapshot   text, page provenance and ingestion status, read together
    Document          a document's content plus its session
    SessionManager    which documents are open, and in what order

Content is append-only. A document may still be being read off a scanner
while somebody is listening to its first page, so blocks arrive over time -
but once a block has been handed to a browser, neither its text nor its index
may ever change. Audio, timings, chapter marks, the saved position and the
DOM all name blocks by index; renumbering them is not a redraw, it is a
silent corruption of every one of those. Hence append_content(), which
extends and never replaces, and ContentSnapshot, which is how anything that
needs both the text and its page numbers reads them without landing on either
side of an append.

One lock guards the whole layer rather than one per document. These are all
short critical sections around dict and int updates - the expensive things
(synthesis, OCR, disk) happen outside it - and a single lock removes the
question of what order to take two of them in, which is exactly the class of
bug this refactor exists to close.

Note what is *not* here: the per-block synthesis locks that used to hang off
each document. The coordinator owns deduplication now (one _Job per block,
whoever asks second waits on the first), so a second lock keyed the same way
would only be a way for the two to disagree.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timezone


def now_iso():
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class ContentSnapshot:
    """A document's text, its page provenance and how much of it exists yet -
    all read in one go, under one lock.

    Reading `blocks` and `block_pages` as two separate properties was safe
    while a document's text never changed after it opened. Progressive
    ingestion appends while the reader is reading, so two reads can land on
    either side of an append and disagree about the length. Everything that
    needs both takes one of these instead.

    `revision` counts appends. `ingestion_revision` counts every change to the
    ingestion status, including the many that add pages but no text - a page
    of pure figures, a page that failed - so a poller can tell "nothing new"
    from "nothing at all is happening".
    """
    blocks: tuple
    block_pages: tuple | None
    revision: int
    ingestion: dict
    ingestion_revision: int

    def __len__(self):
        return len(self.blocks)


# Kept in step with extract.MAX_BLOCK, which is the source of truth; this
# layer cannot import it without dragging PDF extraction into the session
# model, so smoke.py asserts the two agree instead.
MAX_BLOCK_CHARS = 320

# What a document that is not being progressively read looks like. Text,
# EPUBs and any PDF opened the ordinary way are complete the moment they
# exist, and every ingestion question about them answers itself.
COMPLETE = {"state": "complete"}


class ContentError(ValueError):
    """An append that would have broken the append-only contract.

    Every one of these means a bug rather than a bad document: the point of
    raising is that a block already on someone's screen must never change its
    text or its index, so a task that has lost track of what it is appending
    to is stopped rather than allowed to guess.
    """


@dataclass
class Position:
    block: int = 0
    word: int = 0
    within: float = 0.0             # seconds into the current block's audio

    def as_dict(self):
        return {"block": self.block, "word": self.word, "within": self.within}

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        try:
            within = float(d.get("within", 0.0))
            # NaN and infinity survive float() and then poison every arithmetic
            # they touch - progress() ends up calling round() on a NaN, which
            # raises, which is a 500 on a route the reader hits every few
            # seconds. A position that is not a number is no position at all.
            if within != within or within in (float("inf"), float("-inf")):
                within = 0.0
            return cls(int(d.get("block", 0)), int(d.get("word", 0)), within)
        except (TypeError, ValueError, OverflowError):
            return cls()


class PlaybackSession:
    """One document's live playback state.

    `generation` is the same counter the coordinator keys staleness off. It
    goes up whenever what should be playing next changes out from under work
    that is already queued - a voice switch today, a seek later.
    """

    def __init__(self, doc_id, lock, voice=None, position=None, last_opened=None):
        self.doc_id = doc_id
        self._lock = lock
        self.voice = voice
        self.position = position or Position()
        self.last_opened = last_opened or now_iso()
        self.generation = 0

    def snapshot(self):
        with self._lock:
            return {"voice": self.voice, "position": self.position.as_dict(),
                    "last_opened": self.last_opened, "generation": self.generation}

    def set_position(self, block, word, within):
        with self._lock:
            self.position = Position(int(block), int(word), float(within))
            return self.position

    def current_block(self):
        with self._lock:
            return self.position.block

    def set_voice(self, voice, generation=None):
        """Switch this document's voice. `generation` is the number the
        coordinator just issued for the same event - the two counters have to
        agree, and the coordinator's is the one that actually cancels queued
        work, so it is the one that wins. Left out (in a test, say), this
        increments its own."""
        with self._lock:
            self.voice = voice
            self.generation = self.generation + 1 if generation is None else generation
            return self.generation

    def check_and_set_voice(self, voice, expected_generation=None, generation_fn=None):
        """Atomically check expected_generation and update the voice.

        generation_fn, if provided, is called under the lock to obtain the
        next generation (e.g. COORD.bump(doc_id)).
        Returns (ok, current_or_new_generation).
        """
        with self._lock:
            if expected_generation is not None and int(expected_generation) != self.generation:
                return False, self.generation
            gen = generation_fn() if generation_fn is not None else (self.generation + 1)
            self.voice = voice
            self.generation = gen
            return True, gen

    def set_generation(self, generation):
        """Adopt a generation the coordinator issued for a seek."""
        with self._lock:
            self.generation = generation
            return self.generation

    def touch(self):
        with self._lock:
            self.last_opened = now_iso()
            return self.last_opened


class Document:
    """An open document: its text, what we know about it, and its session."""

    def __init__(self, doc_id, title, blocks, lock, source=None, practice=None,
                 chapters=None, block_seconds=None, voice=None, position=None,
                 last_opened=None, block_pages=None, chapter_detection=None,
                 figure_index=None, content_revision=1, ingestion=None):
        self.id = doc_id
        # Tuples, not lists, and replaced wholesale on every append. That is
        # what makes "the same object is the same text" true - which the
        # library's serialisation cache relies on (storage._content_for keys
        # on identity) and which a list appended to in place would quietly
        # break.
        self._blocks = tuple(blocks)
        self.source = source
        self.practice = practice or []
        self._lock = lock
        self._title = title
        self._chapters = list(chapters or [])
        # Quiet status for the two background indexes, so the player can stop
        # polling when there is nothing coming rather than on a timer. Both
        # start "pending" only if something is actually going to run; a
        # document with no PDF behind it is "none" from the moment it opens.
        self._chapter_detection = dict(chapter_detection or
                                       {"state": "none", "method": None})
        self._figure_index = dict(figure_index or {"state": "none"})
        self._block_pages = tuple(block_pages) if block_pages is not None else None
        self._content_revision = max(1, int(content_revision or 1))
        self._ingestion = dict(ingestion) if ingestion else dict(COMPLETE)
        self._ingestion_revision = 1
        # block index -> synthesized runtime in seconds. Filled in as the
        # warmer works through the document; drives the progress percentage.
        self._block_seconds = dict(block_seconds or {})
        self.session = PlaybackSession(doc_id, lock, voice=voice, position=position,
                                       last_opened=last_opened)

    # ---------------------------------------------------------- properties
    @property
    def title(self):
        with self._lock:
            return self._title

    def rename(self, title):
        with self._lock:
            self._title = title

    @property
    def chapters(self):
        with self._lock:
            return list(self._chapters)

    def set_chapters(self, chapters):
        with self._lock:
            self._chapters = list(chapters)

    @property
    def chapter_detection(self):
        with self._lock:
            return dict(self._chapter_detection)

    def set_chapter_detection(self, info):
        with self._lock:
            self._chapter_detection = dict(info)

    @property
    def figure_index(self):
        with self._lock:
            return dict(self._figure_index)

    def set_figure_index(self, info):
        with self._lock:
            self._figure_index = dict(info)

    @property
    def blocks(self):
        """The document's text. Immutable, and safe to hold on to: an append
        replaces this tuple rather than growing it, so a caller reading an
        old one keeps reading consistent text rather than watching it move
        under them."""
        with self._lock:
            return self._blocks

    @property
    def block_pages(self):
        with self._lock:
            return list(self._block_pages) if self._block_pages is not None else None

    def set_block_pages(self, pages):
        """Fill in page provenance a document was opened without.

        Only ever used to supply what was missing, never to change what a
        block says - and it must match the text it describes, or the mapping
        it feeds (chapter starts, figure references) points at the wrong
        paragraphs.
        """
        with self._lock:
            if pages is None:
                raise ContentError("page provenance is filled in, never cleared")
            if len(pages) != len(self._blocks):
                raise ContentError(
                    f"{len(pages)} page numbers for {len(self._blocks)} blocks")
            self._block_pages = tuple(pages)

    @property
    def block_seconds(self):
        with self._lock:
            return dict(self._block_seconds)

    def note_block_seconds(self, index, seconds):
        with self._lock:
            self._block_seconds[index] = seconds

    def text_at(self, index):
        """The coordinator's warm requests come through here. Under the lock,
        so a queued index either sees a complete committed block or no block -
        it can never see half an append.

        Taking the lock here is only safe because the coordinator calls this
        with none of its own locks held (see coordinator._process). The
        declared order is session lock, then coordinator lock - a voice switch
        already enters the coordinator while holding this one - so a
        coordinator that reached for this while holding its own would close
        the cycle and hang the whole server."""
        with self._lock:
            if 0 <= index < len(self._blocks):
                return self._blocks[index]
            return None

    # -------------------------------------------------------------- content
    def content_snapshot(self):
        """Text, page provenance and ingestion status, read together.

        Anything that needs more than one of those - the open response, the
        suffix poll, export, chapter mapping, figure indexing - reads through
        here, so no operation can ever see blocks from one revision and page
        numbers from the next.
        """
        with self._lock:
            return ContentSnapshot(self._blocks, self._block_pages,
                                   self._content_revision, dict(self._ingestion),
                                   self._ingestion_revision)

    @property
    def ingestion(self):
        with self._lock:
            return dict(self._ingestion)

    def append_content(self, expected_revision, blocks, pages, ingestion=None,
                       task_id=None, max_block=MAX_BLOCK_CHARS):
        """Add blocks to the end of the document. Never anywhere else.

        `expected_revision` is what makes this safe against two tasks racing:
        an appender that has not seen the previous append is refused rather
        than allowed to write on top of it. `task_id`, when given, must match
        the ingestion task the document currently belongs to - which is how a
        task belonging to a document that has since been deleted and reopened
        discovers that its work is no longer wanted.

        Returns the snapshot that includes this append, so the caller can
        persist and warm from exactly what it just published without taking
        the lock again.
        """
        pages = tuple(pages) if pages is not None else None
        blocks = tuple(blocks)
        if not blocks:
            raise ContentError("append_content with nothing to append; "
                               "use settle_ingestion for a status-only change")
        with self._lock:
            if expected_revision != self._content_revision:
                raise ContentError(
                    f"expected revision {expected_revision}, have "
                    f"{self._content_revision}")
            if task_id is not None and task_id != self._ingestion.get("task_id"):
                raise ContentError(
                    f"task {task_id} is not this document's ingestion task")
            for text in blocks:
                if not isinstance(text, str) or not text.strip():
                    raise ContentError("blocks must be non-empty text")
                if max_block and len(text) > max_block:
                    raise ContentError(
                        f"block of {len(text)} characters exceeds {max_block}")
            if (self._block_pages is None) != (pages is None):
                raise ContentError(
                    "page provenance must be present for every block or none")
            if pages is not None:
                if len(pages) != len(blocks):
                    raise ContentError(
                        f"{len(pages)} page numbers for {len(blocks)} blocks")
                # Greater than or equal to the frontier, not strictly greater.
                # A paragraph carried across a page boundary keeps the page it
                # started on, so an append for page N+1 legitimately leads with
                # a block belonging to page N - the page the frontier is
                # already sitting on.
                frontier = self._block_pages[-1] if self._block_pages else -1
                last = frontier
                for page in pages:
                    if not isinstance(page, int) or isinstance(page, bool):
                        raise ContentError(f"page {page!r} is not an index")
                    if page < last:
                        raise ContentError(
                            f"page {page} arrives after page {last}; content is "
                            "published in reading order or not at all")
                    last = page
            self._blocks = self._blocks + blocks
            if pages is not None:
                self._block_pages = self._block_pages + pages
            self._content_revision += 1
            if ingestion is not None:
                self._ingestion = dict(ingestion)
            self._ingestion_revision += 1
            return ContentSnapshot(self._blocks, self._block_pages,
                                   self._content_revision, dict(self._ingestion),
                                   self._ingestion_revision)

    def settle_ingestion(self, ingestion, expected_revision=None, task_id=None):
        """Record ingestion progress that produced no text.

        Most pages of a scanned book do produce blocks, but plenty do not - a
        full-page figure, a blank verso, a page Tesseract could not read - and
        the reader still wants to see the count moving. This moves the status
        without touching the content revision, so a poller can tell a page
        that yielded nothing from nothing happening at all.
        """
        with self._lock:
            if expected_revision is not None and expected_revision != self._content_revision:
                raise ContentError(
                    f"expected revision {expected_revision}, have "
                    f"{self._content_revision}")
            if task_id is not None and task_id != self._ingestion.get("task_id"):
                raise ContentError(
                    f"task {task_id} is not this document's ingestion task")
            self._ingestion = dict(ingestion)
            self._ingestion_revision += 1
            return ContentSnapshot(self._blocks, self._block_pages,
                                   self._content_revision, dict(self._ingestion),
                                   self._ingestion_revision)

    def ingesting(self):
        """Whether more text is still coming. Export refuses while this is
        true rather than quietly writing a prefix under the whole book's
        name, and the two background indexes wait for it."""
        with self._lock:
            return self._ingestion.get("state") == "running"


    # ------------------------------------------------------------- storage
    def record(self):
        """Exactly the shape the library has always written to disk."""
        with self._lock:
            rec = {"title": self._title, "blocks": self._blocks,
                   "voice": self.session.voice,
                   "position": self.session.position.as_dict(),
                   "last_opened": self.session.last_opened,
                   "block_seconds": {str(k): v for k, v in self._block_seconds.items()},
                   "source": self.source, "chapters": list(self._chapters),
                   "chapter_detection": dict(self._chapter_detection),
                   "figure_index": dict(self._figure_index)}
            if self._block_pages is not None:
                rec["block_pages"] = list(self._block_pages)
            if self._ingestion.get("state") != "complete":
                # Only written when there is something to say. A library of
                # ordinary documents keeps exactly the record shape it always
                # had, and an older Talkify reading a newer library simply
                # ignores a key it does not know.
                rec["content_revision"] = self._content_revision
                rec["ingestion"] = dict(self._ingestion)
            return rec

    def progress(self):
        """Percentage through the document by actual synthesized-audio
        runtime, not word or block count - a 4-word block and a 60-word block
        should not count the same. Falls back to a block-count ratio until
        every block has been synthesized at least once, since total runtime
        isn't known before that (this only affects the number shown before the
        background warmer catches up; it never blocks anything)."""
        with self._lock:
            n = len(self._blocks)
            bs = self._block_seconds
            total = sum(bs.get(i, 0.0) for i in range(n)) if len(bs) >= n and n else None
            cur = min(self.session.position.block, max(n - 1, 0))
            within = self.session.position.within
            elapsed = sum(bs.get(i, 0.0) for i in range(cur)) + within
        if total:
            pct = max(0, min(100, round(elapsed / total * 100)))
        else:
            pct = max(0, min(100, round((cur / n) * 100))) if n else 0
        return {"elapsed_seconds": round(elapsed, 1),
                "total_seconds": round(total, 1) if total else None,
                "percent": pct}


class SessionManager:
    """Which documents are open, in the order they were opened.

    Opening a second document never evicts the first - each keeps its own
    session and its own place in the audio cache, so synthesis for one
    document never blocks progress on another.
    """

    def __init__(self):
        self.lock = threading.RLock()
        self._docs = {}
        self._order = []

    def new_document(self, doc_id, title, blocks, **kw):
        """Build a Document wired to this manager's lock. Not yet added."""
        return Document(doc_id, title, blocks, self.lock, **kw)

    def add(self, document):
        with self.lock:
            self._docs[document.id] = document
            if document.id not in self._order:
                self._order.append(document.id)
        return document

    def get(self, doc_id):
        with self.lock:
            return self._docs.get(doc_id)

    def remove(self, doc_id):
        with self.lock:
            doc = self._docs.pop(doc_id, None)
            if doc_id in self._order:
                self._order.remove(doc_id)
            return doc

    def all(self):
        with self.lock:
            return [self._docs[d] for d in self._order if d in self._docs]

    def text_at(self, doc_id, index):
        """The coordinator's text provider: warm requests look their text up
        when they run, not when they are queued."""
        doc = self.get(doc_id)
        return doc.text_at(index) if doc is not None else None
