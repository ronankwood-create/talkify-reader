"""
Durable state for Talkify: the library on disk, and the user's settings.

Two things live here, both of which have to survive a crash mid-write:

    SettingsStore       config.json - default voice, export dir, later API keys
    LibraryRepository   one {doc_id}.json per document, plus a SQLite index

Every write goes through _atomic_write(): a temp file in the *same* directory,
flushed, fsync'd, then os.replace()d over the target. os.replace is atomic on
both Windows and POSIX, so a reader either sees the whole old file or the whole
new one - never the truncated middle of a write. That matters more than it
sounds: the position heartbeat rewrites a document's JSON every few seconds
while it plays, so an ordinary open(..., "w") is a live window onto a corrupt
library every time the process dies at the wrong moment.

Why both JSON files and SQLite:

    The JSON files are the contract. They are what load_library() reads at
    startup, what a user can inspect or hand-edit, and what the smoke test
    asserts on. Nothing about reading a document depends on the database.

    SQLite (library.db) is the
    durable index - recents, progress, per-block offsets - and is where the
    split-view and search work in later phases will hang off. It is written
    strictly as a mirror, inside try/except: if the database is locked, on a
    read-only volume, or corrupt, the count in .sqlite_errors goes up and the
    reader carries on with no idea anything happened. Same degrade-to-nothing
    rule as every other enhancement - a fragile store must never
    be on the path between the user and their text.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import sqlite3
import sys
import threading
import time
import uuid
from datetime import datetime

# The library schema. IF NOT EXISTS so
# reopening an existing library is a no-op rather than an error.
SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    content_sha256 TEXT NOT NULL,
    title TEXT NOT NULL,
    author TEXT,
    mime_type TEXT NOT NULL,
    language TEXT,
    page_count INTEGER NOT NULL DEFAULT 0 CHECK (page_count >= 0),
    word_count INTEGER NOT NULL DEFAULT 0 CHECK (word_count >= 0),
    canonical_char_count INTEGER NOT NULL DEFAULT 0 CHECK (canonical_char_count >= 0),
    extraction_status TEXT NOT NULL DEFAULT 'pending',
    extraction_revision TEXT,
    imported_at_ms INTEGER NOT NULL,
    updated_at_ms INTEGER NOT NULL,
    last_opened_at_ms INTEGER NOT NULL,
    archived_at_ms INTEGER
);

CREATE INDEX IF NOT EXISTS documents_recent_idx
    ON documents(last_opened_at_ms DESC, id) WHERE archived_at_ms IS NULL;

CREATE TABLE IF NOT EXISTS document_pages (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    display_width_pt REAL NOT NULL CHECK (display_width_pt > 0),
    display_height_pt REAL NOT NULL CHECK (display_height_pt > 0),
    source_rotation INTEGER NOT NULL DEFAULT 0,
    render_revision TEXT NOT NULL,
    PRIMARY KEY (document_id, page_index)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS document_blocks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page_index INTEGER NOT NULL,
    reading_order INTEGER NOT NULL CHECK (reading_order >= 0),
    kind TEXT NOT NULL,
    source_text TEXT NOT NULL,
    reflow_text TEXT NOT NULL,
    canonical_start INTEGER NOT NULL CHECK (canonical_start >= 0),
    canonical_end INTEGER NOT NULL CHECK (canonical_end >= canonical_start),
    x0 REAL NOT NULL,
    y0 REAL NOT NULL,
    x1 REAL NOT NULL CHECK (x1 >= x0),
    y1 REAL NOT NULL CHECK (y1 >= y0),
    UNIQUE (document_id, reading_order),
    FOREIGN KEY (document_id, page_index)
        REFERENCES document_pages(document_id, page_index) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS document_blocks_page_idx
    ON document_blocks(document_id, page_index, reading_order);

CREATE TABLE IF NOT EXISTS block_spans (
    id TEXT PRIMARY KEY,
    block_id TEXT NOT NULL REFERENCES document_blocks(id) ON DELETE CASCADE,
    span_order INTEGER NOT NULL CHECK (span_order >= 0),
    spoken_text TEXT NOT NULL,
    source_start INTEGER NOT NULL CHECK (source_start >= 0),
    source_end INTEGER NOT NULL CHECK (source_end >= source_start),
    reflow_start INTEGER NOT NULL CHECK (reflow_start >= 0),
    reflow_end INTEGER NOT NULL CHECK (reflow_end >= reflow_start),
    canonical_start INTEGER NOT NULL CHECK (canonical_start >= 0),
    canonical_end INTEGER NOT NULL CHECK (canonical_end >= canonical_start),
    UNIQUE (block_id, span_order)
);

CREATE TABLE IF NOT EXISTS span_boxes (
    span_id TEXT NOT NULL REFERENCES block_spans(id) ON DELETE CASCADE,
    box_order INTEGER NOT NULL CHECK (box_order >= 0),
    page_index INTEGER NOT NULL CHECK (page_index >= 0),
    x0 REAL NOT NULL,
    y0 REAL NOT NULL,
    x1 REAL NOT NULL CHECK (x1 >= x0),
    y1 REAL NOT NULL CHECK (y1 >= y0),
    PRIMARY KEY (span_id, box_order)
) WITHOUT ROWID;

CREATE TABLE IF NOT EXISTS playback_timelines (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    profile_hash TEXT NOT NULL,
    extraction_revision TEXT NOT NULL,
    sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
    total_samples INTEGER NOT NULL CHECK (total_samples >= 0),
    created_at_ms INTEGER NOT NULL,
    UNIQUE (document_id, profile_hash, extraction_revision)
);

CREATE TABLE IF NOT EXISTS timeline_cues (
    timeline_id TEXT NOT NULL REFERENCES playback_timelines(id) ON DELETE CASCADE,
    cue_order INTEGER NOT NULL CHECK (cue_order >= 0),
    span_id TEXT NOT NULL REFERENCES block_spans(id) ON DELETE CASCADE,
    start_sample INTEGER NOT NULL CHECK (start_sample >= 0),
    end_sample INTEGER NOT NULL CHECK (end_sample > start_sample),
    PRIMARY KEY (timeline_id, cue_order),
    UNIQUE (timeline_id, span_id)
) WITHOUT ROWID;

CREATE INDEX IF NOT EXISTS timeline_cues_lookup_idx
    ON timeline_cues(timeline_id, start_sample, end_sample);

CREATE TABLE IF NOT EXISTS playback_progress (
    document_id TEXT PRIMARY KEY REFERENCES documents(id) ON DELETE CASCADE,
    timeline_id TEXT REFERENCES playback_timelines(id) ON DELETE SET NULL,
    current_span_id TEXT REFERENCES block_spans(id) ON DELETE SET NULL,
    canonical_offset INTEGER NOT NULL DEFAULT 0 CHECK (canonical_offset >= 0),
    position_sample INTEGER NOT NULL DEFAULT 0 CHECK (position_sample >= 0),
    completed INTEGER NOT NULL DEFAULT 0 CHECK (completed IN (0, 1)),
    revision INTEGER NOT NULL DEFAULT 0,
    updated_at_ms INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS playback_progress_updated_idx
    ON playback_progress(updated_at_ms DESC);
"""

# Audio positions are stored as sample counts, not seconds, so that
# timings never drift through float rounding. Kokoro's output rate.
SAMPLE_RATE = 24_000


def _atomic_write(path, data, mode="w"):
    """Replace `path` with `data` in one step, or not at all.

    The temp file must sit in the same directory as the target: os.replace is
    only atomic within a filesystem, and %TEMP% is regularly on a different
    volume from a user's Downloads folder on Windows.
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    os.makedirs(directory, exist_ok=True)
    tmp = os.path.join(directory, f".{os.path.basename(path)}.{uuid.uuid4().hex}.part")
    kwargs = {"encoding": "utf-8"} if mode == "w" else {}
    try:
        with open(tmp, mode, **kwargs) as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        # Windows refuses to replace a file that any other handle currently
        # has open, even for reading - so os.replace raises PermissionError
        # where POSIX would just do it. The readers here are all short (one
        # small JSON file), and the alternative is a failed save every time a
        # backup tool, an editor or another request happens to be looking at
        # the file. Retry briefly, then give up honestly.
        for attempt in range(10):
            try:
                os.replace(tmp, path)
                break
            except PermissionError:
                if attempt == 9:
                    raise
                time.sleep(0.01 * (attempt + 1))
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def _epoch_ms(iso_or_none):
    """ISO-8601 (what the rest of the app passes around) to the integer epoch
    milliseconds the schema wants. An unparseable value becomes 'now' rather
    than an exception - a bad timestamp must not cost someone their library."""
    try:
        return int(datetime.fromisoformat(iso_or_none).timestamp() * 1000)
    except (TypeError, ValueError):
        return int(time.time() * 1000)


class _Content:
    """Everything the index needs to know about one document's block list.

    Derived once, when that list is first seen, then reused for as long as it
    lives. The digest is folded block by block instead of over one joined
    string, so a 1.6MB textbook is never materialized a second time just to be
    hashed, and the word count is the sum of the per-block counts - exactly
    what splitting the joined text gives, since blocks are joined by
    whitespace and no token can straddle a join.
    """

    __slots__ = ("blocks", "digest", "word_count", "char_count", "_offsets")

    def __init__(self, blocks):
        self.blocks = blocks
        h = hashlib.sha256()
        sep = ("\n\n").encode("utf-8")
        words = 0
        offsets = []
        offset = 0
        for i, text in enumerate(blocks):
            if i:
                h.update(sep)
                offset += 2
            h.update(text.encode("utf-8"))
            words += len(text.split())
            offsets.append(offset)
            offset += len(text)
        self.digest = h.hexdigest()
        self.word_count = words
        self.char_count = offset
        self._offsets = offsets

    def offset_of(self, block_index):
        """Where block `block_index` starts in the joined text - the schema's
        canonical_offset. A position past the end (saved against a document
        that has since been re-extracted shorter) clamps rather than raising."""
        if block_index <= 0 or not self._offsets:
            return 0
        if block_index >= len(self._offsets):
            return self.char_count
        return self._offsets[block_index]


class SettingsStore:
    """config.json, read on demand and written atomically.

    Read-through rather than cached on purpose: this file is small, it is
    hand-editable, and reading it fresh means an edit made in a text editor
    takes effect without restarting the server. A missing or corrupt file
    reads as {} - every caller already falls back to an engine default, so a
    mangled config costs a preference, never the ability to read a book.
    """

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._cache = None              # ((size, mtime_ns), parsed dict)

    def all(self):
        """The settings, re-parsed only when the file on disk has changed.

        Still read-through in the way that matters: the cache is keyed on the
        file's size and modification time, so an edit made in a text editor
        takes effect on the next call exactly as before. What it stops is
        opening and parsing the same few hundred bytes once per document at
        startup and once per request that wants a default voice.
        """
        try:
            st = os.stat(self.path)
            key = (st.st_size, st.st_mtime_ns)
        except OSError:
            self._cache = None
            return {}
        cached = self._cache
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            with open(self.path, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        data = data if isinstance(data, dict) else {}
        self._cache = (key, data)
        return data

    def get(self, key, default=None):
        return self.all().get(key, default)

    def set(self, key, value):
        """Read-modify-write under a lock, so two settings changed at the same
        moment can't lose one of them."""
        with self._lock:
            cfg = dict(self.all())
            cfg[key] = value
            _atomic_write(self.path, json.dumps(cfg))
            self._cache = None          # re-stat on the next read, never guess
        return cfg


class LibraryRepository:
    """Documents on disk: a JSON file each, mirrored into a SQLite index.

    The JSON file is authoritative. save() writes it first and lets any error
    propagate, exactly as the previous inline _save_doc() did; only after it
    lands does the SQLite mirror run, and that one swallows its own failures.
    """

    def __init__(self, directory, db_name="library.db"):
        self.directory = directory
        self.db_path = os.path.join(directory, db_name)
        self.sqlite_errors = 0
        self.last_sqlite_error = None
        self._lock = threading.RLock()
        self._db = None
        # doc_id -> the _Content derived from the block list last saved under
        # it, and the JSON text that list serialized to. Both exist for one
        # reason: the position heartbeat calls save() every three seconds for
        # as long as a document plays, and everything derived from `blocks` -
        # the digest, the word count, the serialized array - is identical on
        # every one of those calls. On a 5000-block textbook re-deriving it
        # each time cost ~28ms; reusing it costs a pointer comparison.
        self._content = {}
        self._blocks_json = {}
        # One lock per document, not one for the library: a figure index
        # settling on a 900-page textbook must not stall a position heartbeat
        # for a different book. Held only around read-modify-write of one
        # record - never around OCR, synthesis, an AI call, HTTP or rendering.
        self._record_locks = {}
        self._record_locks_guard = threading.Lock()
        # doc_id -> the record this repository last wrote there. The base an
        # update mutates, so a writer that only touches `voice` cannot roll
        # back a `figure_index` written a moment earlier by somebody else.
        self._records = {}
        self._connect()

    def _record_lock(self, doc_id):
        with self._record_locks_guard:
            lock = self._record_locks.get(doc_id)
            if lock is None:
                lock = self._record_locks[doc_id] = threading.RLock()
            return lock

    # The record's own large members. They are replaced wholesale by whoever
    # owns them (session.Document builds a new list on every append) and never
    # mutated in place, so copying them per save would be pure cost on a
    # 5000-block textbook.
    _BULK_FIELDS = ("blocks", "block_pages")

    @classmethod
    def _detached(cls, record):
        """A copy a mutator can safely change without reaching what is stored.

        A plain dict() is not enough: `position` and `figure_index` are nested
        objects, so a mutator writing record["position"]["block"] would be
        editing the repository's own idea of what is on disk, outside the
        lock, and a later unrelated save would persist it. One level down is
        as deep as this record goes."""
        out = {}
        for key, value in record.items():
            if key in cls._BULK_FIELDS:
                out[key] = value
            elif isinstance(value, dict):
                out[key] = dict(value)
            elif isinstance(value, list):
                out[key] = list(value)
            else:
                out[key] = value
        return out

    def latest_record(self, doc_id):
        """A fresh, mutable copy of what is currently persisted.

        Served from the last successful write when there was one, and read off
        disk otherwise - which is the reopened-library case. A file that is
        missing or will not parse reads as {}, exactly as load_all() has
        always treated it: a corrupt record costs its contents, never the
        ability to save the next thing."""
        held = self._records.get(doc_id)
        if held is not None:
            return self._detached(held)
        try:
            with open(self.path_for(doc_id), encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return {}
        return self._detached(data) if isinstance(data, dict) else {}

    def update_record(self, doc_id, mutator):
        """Read-modify-write one document under its own lock.

        Every durable change to a document goes through here - the voice, the
        reading position, a rename, a chapter list, a figure index. The point
        is not the write, which was already atomic; it is that the *read* and
        the write are one step. Two threads that each snapshot a document and
        then save it can otherwise land in the order snapshot-A, snapshot-B,
        write-B, write-A - and the disk ends up holding A's older idea of the
        voice, silently undoing a switch the reader watched happen.

        `mutator` is handed the current record and mutates it in place. It
        must be quick and must not do I/O of its own: it runs with this
        document's lock held.
        """
        with self._record_lock(doc_id):
            record = self.latest_record(doc_id)
            mutator(record)
            self._write(doc_id, record)
            return record

    def _content_for(self, doc_id, blocks):
        """The digest, word count and block offsets of `blocks`, derived once
        per list rather than once per save.

        Keyed on the identity of the list object, which is exactly what a
        Document hands over: Document.record() passes self.blocks straight
        through and nothing ever mutates it in place, so the same object is
        the same text. A different (or first) object recomputes from scratch,
        which is also what happens after a restart - so nothing stale can
        outlive the process that cached it.
        """
        cached = self._content.get(doc_id)
        if cached is not None and cached.blocks is blocks:
            return cached
        content = _Content(blocks)
        self._content[doc_id] = content
        return content

    # ------------------------------------------------------------ sqlite
    def _connect(self):
        try:
            os.makedirs(self.directory, exist_ok=True)
            # isolation_level=None means autocommit, so the PRAGMAs below apply
            # immediately and every statement here is explicit about its own
            # transaction. check_same_thread=False because ThreadingHTTPServer
            # hands each request to a different thread; one connection behind
            # one lock is simpler than a pool and these writes are tiny.
            db = sqlite3.connect(self.db_path, check_same_thread=False,
                                 isolation_level=None, timeout=5.0)
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA journal_mode = WAL")
            db.execute("PRAGMA synchronous = NORMAL")
            db.execute("PRAGMA busy_timeout = 5000")
            db.execute("PRAGMA foreign_keys = ON")
            db.executescript(SCHEMA)
            self._db = db
        except sqlite3.Error as e:
            self._note_error(e)
            self._db = None

    def _note_error(self, e):
        self.sqlite_errors += 1
        self.last_sqlite_error = f"{type(e).__name__}: {e}"
        # Loud enough to find in a terminal, quiet enough not to interrupt
        # anyone reading. A swallowed error costs more than a
        # visible one - but this one must not stop playback either.
        print(f"talkify: library index unavailable ({self.last_sqlite_error})",
              file=sys.stderr)

    def close(self):
        with self._lock:
            if self._db is not None:
                try:
                    self._db.close()
                except sqlite3.Error:
                    pass
                self._db = None

    # -------------------------------------------------------------- json
    def path_for(self, doc_id):
        return os.path.join(self.directory, f"{doc_id}.json")

    def save(self, doc_id, record):
        """Persist one document, replacing what is there.

        `record` is the same dict shape the library has always held on disk:
        title, blocks, voice, position, last_opened, block_seconds, source,
        chapters. Serialised per document like every other write - a caller
        that built this record from live state should be using update_record
        so that the snapshot happens inside the lock too."""
        with self._record_lock(doc_id):
            self._write(doc_id, record)

    def _write(self, doc_id, record):
        """Caller holds this document's record lock."""
        os.makedirs(self.directory, exist_ok=True)
        payload, rest = self._record_json(doc_id, record)
        # Nothing to write when nothing moved - a document saved twice with the
        # same content (a figure index settling to the state it already had, an
        # analysis pass confirming what was there) rewrites the whole file for
        # no reason. Only the small, mutable half of the record is compared;
        # the blocks are matched by identity, which is stricter and free.
        #
        # The note of what is on disk is made *after* the write lands, never
        # before. Marking it first meant a write that failed - and on Windows
        # _atomic_write can genuinely fail, which is why it retries - left this
        # believing the file said something it did not, and the next save of
        # that same content would be skipped as redundant.
        held = self._blocks_json.get(doc_id)
        if rest is None or held is None or held[2] != rest:
            _atomic_write(self.path_for(doc_id), payload)
            held = self._blocks_json.get(doc_id)
            if rest is not None and held is not None:
                self._blocks_json[doc_id] = (held[0], held[1], rest)
        # Only once the bytes are on disk. A failed _atomic_write must leave
        # this believing the *previous* record is what is persisted, or the
        # next update would build on a version that never landed.
        self._records[doc_id] = self._detached(record)
        self._mirror(doc_id, record)

    def _record_json(self, doc_id, record):
        """json.dumps(record), with the blocks array serialized once per
        document instead of once per heartbeat.

        Returns (payload, rest), where `rest` is the serialized form of
        everything except the blocks - the only part that can change from one
        save to the next, and so what save() compares to decide whether a
        rewrite would say anything new. It is None when the record has no
        block list to lift out and the payload is a plain json.dumps.

        The result is a JSON object with exactly the same members and values -
        only the key order differs, and nothing has ever read this file
        positionally. `blocks` is far and away its largest member and the only
        immutable one, so lifting it out of the per-save work is most of the
        saving."""
        blocks = record.get("blocks")
        if not isinstance(blocks, list):
            return json.dumps(record), None
        cached = self._blocks_json.get(doc_id)
        if cached is None or cached[0] is not blocks:
            # A different (or first) block list: re-serialize it, and forget
            # what the last write said, since it was about other text.
            cached = (blocks, json.dumps(blocks), None)
            self._blocks_json[doc_id] = cached
        rest = json.dumps({k: v for k, v in record.items() if k != "blocks"})
        # rest is "{}" when blocks was the only member; otherwise drop its
        # opening brace and splice the remaining members on.
        payload = '{"blocks":' + cached[1] + ("," + rest[1:] if len(rest) > 2 else "}")
        return payload, rest

    def load_all(self):
        """Every saved document, oldest first, as (doc_id, record) pairs.

        Ordered by file mtime rather than by anything in the database: the
        JSON files are the source of truth, and a library that predates the
        index (or one whose index failed to open) must still come back in the
        same order it always did.
        """
        out = []
        paths = sorted(glob.glob(os.path.join(self.directory, "*.json")),
                       key=os.path.getmtime)
        for path in paths:
            doc_id = os.path.splitext(os.path.basename(path))[0]
            try:
                with open(path, encoding="utf-8") as f:
                    record = json.load(f)
            except (OSError, ValueError):
                continue
            if not isinstance(record, dict) or "blocks" not in record:
                continue
            out.append((doc_id, record))
        return out

    def delete(self, doc_id):
        # Under the same lock as every write to this document. Without it a
        # save already past its read can land after the file is removed and
        # put the document back - JSON, SQLite mirror and all - which is not a
        # slow delete, it is a delete that did not happen.
        with self._record_lock(doc_id):
            self._delete_locked(doc_id)

    def _delete_locked(self, doc_id):
        self._content.pop(doc_id, None)
        self._blocks_json.pop(doc_id, None)
        self._records.pop(doc_id, None)
        try:
            os.remove(self.path_for(doc_id))
        except OSError:
            pass
        with self._lock:
            if self._db is None:
                return
            try:
                # ON DELETE CASCADE clears blocks, pages, spans and progress.
                self._db.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            except sqlite3.Error as e:
                self._note_error(e)

    # ------------------------------------------------------------- index
    def ensure_indexed(self, doc_id, record):
        """Mirror this document only if the index does not already hold it.

        For the startup path, which re-registers every saved document without
        writing any of them back. Deriving a record's content summary means
        hashing the whole text, so doing it for a library the index already
        covers would put back the cost that not-rewriting just removed - hence
        the primary-key check first. An unavailable index is not an error here
        any more than anywhere else in this class."""
        with self._lock:
            if self._db is None:
                return False
            try:
                row = self._db.execute(
                    "SELECT 1 FROM documents WHERE id = ?", (doc_id,)).fetchone()
            except sqlite3.Error as e:
                self._note_error(e)
                return False
            if row is not None:
                return False
        self._mirror(doc_id, record)
        return True

    def _mirror(self, doc_id, record):
        with self._lock:
            if self._db is None:
                return
            try:
                self._write_index(doc_id, record)
            except sqlite3.Error as e:
                self._note_error(e)
                try:
                    self._db.execute("ROLLBACK")
                except sqlite3.Error:
                    pass

    def _write_index(self, doc_id, record):
        db = self._db
        blocks = record.get("blocks") or []
        content = self._content_for(doc_id, blocks)
        digest = content.digest
        now_ms = int(time.time() * 1000)
        opened_ms = _epoch_ms(record.get("last_opened"))
        source = record.get("source") or {}
        source_uri = source.get("path") or f"talkify:text/{doc_id}"
        mime = ("application/pdf" if str(source_uri).lower().endswith(".pdf")
                else "text/plain")

        row = db.execute("SELECT content_sha256 FROM documents WHERE id = ?",
                         (doc_id,)).fetchone()
        db.execute("BEGIN IMMEDIATE")
        try:
            db.execute(
                """INSERT INTO documents (
                       id, source_uri, content_sha256, title, author, mime_type,
                       language, page_count, word_count, canonical_char_count,
                       extraction_status, extraction_revision,
                       imported_at_ms, updated_at_ms, last_opened_at_ms, archived_at_ms)
                   VALUES (?, ?, ?, ?, NULL, ?, NULL, 0, ?, ?, 'complete', 'v1',
                           ?, ?, ?, NULL)
                   ON CONFLICT(id) DO UPDATE SET
                       source_uri = excluded.source_uri,
                       content_sha256 = excluded.content_sha256,
                       title = excluded.title,
                       mime_type = excluded.mime_type,
                       word_count = excluded.word_count,
                       canonical_char_count = excluded.canonical_char_count,
                       updated_at_ms = excluded.updated_at_ms,
                       last_opened_at_ms = excluded.last_opened_at_ms""",
                (doc_id, str(source_uri), digest, record.get("title") or "untitled",
                 mime, content.word_count, content.char_count,
                 now_ms, now_ms, opened_ms))

            # Blocks only get rewritten when the text actually changed. The
            # position heartbeat calls save() every few seconds while a
            # document plays; re-inserting two thousand rows each time would
            # turn a progress update into a disk stall.
            if row is None or row["content_sha256"] != digest:
                db.execute(
                    """INSERT INTO document_pages (document_id, page_index,
                           display_width_pt, display_height_pt, source_rotation,
                           render_revision)
                       VALUES (?, 0, 612.0, 792.0, 0, 'v1')
                       ON CONFLICT(document_id, page_index) DO NOTHING""",
                    (doc_id,))
                db.execute("DELETE FROM document_blocks WHERE document_id = ?", (doc_id,))
                offset = 0
                rows = []
                for i, text in enumerate(blocks):
                    end = offset + len(text)
                    # Coordinates are zeroes until Phase 1 carries PyMuPDF's
                    # real block rects through extraction; the columns exist
                    # now so that split-view is a data change, not a migration.
                    rows.append((f"{doc_id}:{i}", doc_id, 0, i, "paragraph",
                                 text, text, offset, end, 0.0, 0.0, 0.0, 0.0))
                    offset = end + 2          # the "\n\n" between blocks
                db.executemany(
                    """INSERT INTO document_blocks (
                           id, document_id, page_index, reading_order, kind,
                           source_text, reflow_text, canonical_start, canonical_end,
                           x0, y0, x1, y1)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)

            position = record.get("position") or {}
            block_index = int(position.get("block", 0) or 0)
            within = float(position.get("within", 0.0) or 0.0)
            canonical = content.offset_of(block_index)
            db.execute(
                """INSERT INTO playback_progress (
                       document_id, timeline_id, current_span_id, canonical_offset,
                       position_sample, completed, revision, updated_at_ms)
                   VALUES (?, NULL, NULL, ?, ?, 0, 1, ?)
                   ON CONFLICT(document_id) DO UPDATE SET
                       canonical_offset = excluded.canonical_offset,
                       position_sample = excluded.position_sample,
                       revision = playback_progress.revision + 1,
                       updated_at_ms = excluded.updated_at_ms""",
                (doc_id, max(0, canonical), max(0, int(within * SAMPLE_RATE)), now_ms))
            db.execute("COMMIT")
        except BaseException:
            # Rolling back can itself fail - if BEGIN never took, there is no
            # transaction to undo. Swallowing that is the point: the caller
            # needs the error that actually happened, not "no transaction is
            # active" reported in its place.
            try:
                db.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise

    def recent(self, limit=3):
        """The library shelf query. Returns [] rather than raising
        if the index is unavailable, so a caller can always render something."""
        with self._lock:
            if self._db is None:
                return []
            try:
                rows = self._db.execute(
                    """SELECT d.id, d.title, d.last_opened_at_ms,
                              p.canonical_offset, p.position_sample, p.completed
                         FROM documents AS d
                         LEFT JOIN playback_progress AS p ON p.document_id = d.id
                        WHERE d.archived_at_ms IS NULL
                        ORDER BY d.last_opened_at_ms DESC, d.id
                        LIMIT ?""", (int(limit),)).fetchall()
            except sqlite3.Error as e:
                self._note_error(e)
                return []
        return [dict(r) for r in rows]

    def block_count(self, doc_id):
        """How many block rows the index holds for a document. Used by the
        smoke test; also the cheapest way to see whether a mirror ran."""
        with self._lock:
            if self._db is None:
                return 0
            try:
                row = self._db.execute(
                    "SELECT COUNT(*) AS n FROM document_blocks WHERE document_id = ?",
                    (doc_id,)).fetchone()
            except sqlite3.Error as e:
                self._note_error(e)
                return 0
        return row["n"] if row else 0

    def tables(self):
        with self._lock:
            if self._db is None:
                return []
            try:
                rows = self._db.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
            except sqlite3.Error as e:
                self._note_error(e)
                return []
        return sorted(r["name"] for r in rows)
