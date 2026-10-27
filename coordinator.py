"""
Bounded synthesis: one worker, a priority queue, and a byte-capped cache.

What this replaces: every call to _warm() used to spawn its own daemon thread
that walked a whole document calling synth(). Open four documents, switch voice
twice and reopen one, and six threads were racing for the same blocks with
nothing bounding either the thread count or the audio they piled into memory. A
980-page textbook's worth of WAV is gigabytes.

Three ideas, all of them small:

  AudioCache          keyed by (doc_id, block_index, voice), with a byte
                      budget. Insert past the budget and the least recently
                      used entries go until it fits again.

  WarmRequest         one unit of work: which block, which voice, which
                      generation, how urgent. Ordered by (priority, sequence),
                      so an interactive request jumps the whole warm backlog
                      and ties break in arrival order.

  SynthCoordinator    dispatcher threads draining that queue. One by default,
                      because most engines are one thing behind one lock and
                      more threads only ever meant more contention. Kokoro is
                      the exception: its worker runs TTS and ASR as two stages,
                      so two dispatchers keep both busy - see max_inflight.

Generation counters are how a seek or a voice switch cancels work that is
already queued. Each document has a counter; switching voice bumps it; a
request pulled off the queue whose generation is older than the document's
current one is dropped without being synthesized. Note that staleness here is
only about *wasted effort*, never about correctness: the cache key includes the
voice, so audio produced under an old generation can never be published as if
it belonged to the new one. That is the invariant the release gate asks for,
and it holds structurally rather than by timing.

Lock order, the one rule to keep: the coordinator's lock may be held while
taking the cache's lock, never the other way round, and neither is ever held
across a call into the speech engine or a completion callback. That is why
_process() does its synthesis outside every lock it owns - and why cancelling
a job's engine handle happens after the lock is released, never under it.

Pipelining, when the engine offers it
-------------------------------------
An engine may expose an optional submit(text, voice, meta) -> handle alongside
the required synth(). Given one, the coordinator runs max_inflight dispatchers
instead of one, so two blocks can be inside the engine at once - which is what
lets Kokoro's TTS stage work on block N+1 while Whisper is still aligning block
N. Two, not eight: more dispatchers only fill the engine's own queues, hold
more audio, and put speculative work in front of requests that have not been
made yet.

The handle is also what makes a seek genuinely cheap. Queued stale work was
always dropped for free; now bump() and forget() can cancel the block that is
already inside the engine as well, so the audio a reader has just seeked away
from stops being paid for instead of finishing quietly.

Prefetch for a scan, which is not the same as cache
---------------------------------------------------
An export is a one-pass forward walk, and the cache is an LRU. Admitting the
walk's own misses is the textbook way to destroy an LRU: blocks 0..N evict the
warmed blocks N+1.. just before the walk reaches them. So export warms with
cache_result=False.

That leaves a second question, which is where a block prefetched under that
rule actually goes. Nowhere was the wrong answer: a prefetched block that
finishes before the exporter asks for it was dropped on the floor and then
synthesized a *second* time when the exporter arrived - measured at exactly
2x Kokoro for a book, which on this machine is hours. Uncollected prefetch is
therefore retained in a small handoff buffer, outside the LRU and bounded both
by bytes and by the warm window itself: _refill counts what is waiting there
as outstanding, so the producer stops rather than piling up audio in front of
a consumer that has stalled. synth() collects from it and frees the slot.

Retained only when nobody was waiting. A pinned job's result went straight to
the caller that asked for it, and keeping a second reference to it would be
memory with no reader.
"""
from __future__ import annotations

import heapq
import itertools
import queue
import threading
from collections import OrderedDict
from dataclasses import dataclass, field

# Lower sorts first. A block someone is waiting to hear beats everything;
# below that, an export - a job the user asked for and is watching a progress
# bar for - beats every kind of speculative warming.
#
# Export used to sit at the bottom of the band, below the warmer, on the
# reasoning that it is a background job. That starved it outright: every
# completed block calls _refill(), which tops the warm window straight back
# up, so an export block needing synthesis sat behind an endless stream of
# warm requests and the progress bar stopped moving for good. Priority 4 keeps
# the property that actually mattered - live playback (0) still preempts the
# export, so listening never stutters - without the starvation.
#
# The warm band is tiered because the library warms *every* document at
# startup. With one priority for all of them the single worker alternated
# between the book you are reading and a 900-page textbook nobody has open,
# in arrival order - so the background book took half the synthesis time the
# reader was waiting on. Blocks ahead of the open document now outrank blocks
# of documents that are merely in the library; PRIORITY_WARM stays as the
# neutral default for callers that never say which document is active.
# The player's own lookahead sits in its own tier between the two. It is not
# interactive - nobody is listening to block i+2 yet - but it is not warming
# either: it is the standby slot that makes the next paragraph start without a
# gap, and it is wanted within one block's playing time. Giving it
# PRIORITY_INTERACTIVE, which is what /api/block and /api/audio did for every
# caller alike, meant a seek to block 400 queued *behind* the two speculative
# blocks the browser had just asked for, at the one moment the user is most
# obviously waiting: measured as a whole extra block of synthesis before the
# paragraph that was clicked on even started.
PRIORITY_INTERACTIVE = 0        # sounding right now / someone waiting on playback
PRIORITY_PREFETCH = 2           # the player's standby slot and its lookahead
PRIORITY_EXPORT = 4             # user-initiated export; beats all warming
PRIORITY_ACTIVE_WARM = 5        # blocks ahead of the actively open document
PRIORITY_WARM = 10              # default warm priority
PRIORITY_BACKGROUND_WARM = 12   # documents in the library, not currently open

# Kokoro emits 24 kHz mono PCM, so 256 MiB retains about 93 minutes of speech.
# The old 128 MiB ceiling held only 46 minutes: a typical chapter-sized PDF
# could finish warming and still show barely half its blocks ready. The cache
# remains strictly bounded (and TALKIFY_CACHE_BYTES can lower it on a small
# machine), but the default now covers the common hour-to-hour-and-a-half
# export without throwing away work the user has explicitly waited for.
DEFAULT_CACHE_BYTES = 256 * 1024 * 1024
# How many blocks of one document may be queued or in flight at once. The
# warmer refills as each finishes, so a 2000-block book still gets fully
# warmed - it just never has 2000 requests pending to do it.
DEFAULT_WARM_WINDOW = 8
# A ceiling across all documents, so opening twenty books at once cannot
# balloon the queue either.
DEFAULT_MAX_PENDING = 256
# How much prefetched-but-uncollected audio may sit outside the cache. The
# warm window bounds this far more tightly in practice - one export window is
# eight blocks, about 6 MB of 24 kHz WAV - and this is the ceiling that holds
# if several documents are being exported at once, or if one is abandoned
# mid-window.
DEFAULT_HANDOFF_BYTES = 32 * 1024 * 1024
# How many failed block indexes one document may remember. A corrupt book
# cannot be allowed to grow this without limit; past the cap the warmer simply
# retries, which costs a failed call and no memory.
MAX_FAILED_BLOCKS = 256
# Dispatchers to run against an engine that offers the pipelined submit(),
# when a caller asks for the overlap. Two, because the pipeline has two stages:
# one block can be in TTS while another is in ASR, and a third would only wait.
# More dispatchers do not make a two-stage worker any wider - they just hold
# more audio and put speculative work in front of requests nobody has made yet.
#
# Not the default. Measured against real models, Kokoro takes 7.9s of a 9.1s
# block and Whisper 1.2s, so the best overlap theoretically available is 13%,
# and contention between the two runtimes ate all of it. See bench_synth.py -
# and re-run it before assuming the answer is the same on other hardware.
DEFAULT_MAX_INFLIGHT = 2


# Sentinels for the per-document block index below. _NO_BLOCKS is an empty
# mapping to look a miss up in without allocating one; _MISSING marks "no
# entry" distinctly from a cached voice of None, which is a real and common
# value (it means "whatever the engine defaults to").
_NO_BLOCKS = {}
_MISSING = object()


class StaleRequest(Exception):
    """Raised into a waiter whose document moved on before its turn came up."""


@dataclass(frozen=True)
class SynthMeta:
    """What the coordinator knows about one request, passed to a pipelined
    engine's submit(). Ordering and logging only - none of it reaches the cache
    key, which is still (doc_id, block_index, voice). An engine is free to read
    just `priority` and ignore the rest; that is what KokoroEngine does."""
    doc_id: str = ""
    block_index: int = -1
    generation: int = 0
    priority: int = 0
    background: bool = False


@dataclass(frozen=True)
class CacheEntry:
    voice: object
    audio: bytes
    mime: str
    words: list


class AudioCache:
    """LRU over synthesized blocks, bounded by total audio bytes.

    Keyed by (doc_id, block_index, voice) as the design calls for, but lookups
    are by block: get(doc, i) returns whatever voice that block is currently
    cached under. That is deliberate and it preserves the existing behaviour of
    a mid-document voice switch - blocks already spoken keep their old audio
    (nothing yanks the paragraph you are listening to), and only the unread
    ones ahead are dropped and re-synthesized in the new voice. One entry per
    block is what makes that describable at all; two voices of the same block
    would just be memory nobody asked for.
    """

    def __init__(self, max_bytes=DEFAULT_CACHE_BYTES):
        self.max_bytes = max_bytes
        self._lock = threading.Lock()
        # Ordered least-recently-used first, so eviction is popitem(last=False)
        # and a hit is move_to_end - both O(1), which a plain list scan is not
        # once a book has a few thousand blocks in here.
        self._entries = OrderedDict()   # (doc_id, index, voice) -> CacheEntry
        # doc_id -> {index: the voice that block is currently held in}. Nested
        # rather than flat ((doc_id, index) -> voice), so every per-document
        # question - how many blocks are ready, which voices they are in, drop
        # them all - is answered out of that document's own bucket instead of
        # a scan of the whole cache. /api/docs asks ready() once per document
        # every three seconds while the library view is open, and the cache
        # holds thousands of entries once a textbook is warm: that was
        # O(library x cache) per poll, and ready() is now O(1).
        self._by_doc = {}               # doc_id -> {index: voice}
        self._bytes = 0
        # What produced the audio in here: transport, backend, model. The key
        # is (doc_id, index, voice) and deliberately stays that way - a fourth
        # component would be carried through every lookup to describe
        # something that changes at most once in a session. Instead, when it
        # does change, everything made by the old one goes.
        self._identity = None
        self.evictions = 0

    # ------------------------------------------------------------ internals
    def _drop(self, key):
        entry = self._entries.pop(key, None)
        if entry is None:
            return
        self._bytes -= len(entry.audio)
        doc_id, index, _voice = key
        held = self._by_doc.get(doc_id)
        if held is not None and held.get(index, _MISSING) == entry.voice:
            del held[index]
            if not held:
                del self._by_doc[doc_id]

    # ---------------------------------------------------------------- api
    def get(self, doc_id, index, voice=None):
        """The cached audio for this block in whatever voice it was made in (or
        a specific requested voice), or None. Counts as a use, so it moves to
        the back of the LRU."""
        with self._lock:
            cached_voice = self._by_doc.get(doc_id, _NO_BLOCKS).get(index, _MISSING)
            if cached_voice is _MISSING:
                return None
            if voice is not None and cached_voice != voice:
                return None
            key = (doc_id, index, cached_voice)
            entry = self._entries.get(key)
            if entry is None:
                return None
            self._entries.move_to_end(key)
            return entry

    def put(self, doc_id, index, voice, audio, mime, words):
        size = len(audio)
        with self._lock:
            # A single block bigger than the whole budget is not cacheable -
            # storing it would put the cache over its cap by definition. The
            # caller still gets its audio; it just won't be there next time.
            if size > self.max_bytes:
                return None
            held = self._by_doc.get(doc_id)
            if held is not None and index in held:
                self._drop((doc_id, index, held[index]))
            key = (doc_id, index, voice)
            entry = CacheEntry(voice, audio, mime, words)
            self._entries[key] = entry
            self._entries.move_to_end(key)
            held = self._by_doc.get(doc_id)
            if held is None:
                held = self._by_doc[doc_id] = {}
            held[index] = voice
            self._bytes += size
            while self._bytes > self.max_bytes and len(self._entries) > 1:
                oldest = next(iter(self._entries))
                self._drop(oldest)
                self.evictions += 1
            return entry

    def ready(self, doc_id):
        """How many of a document's blocks are cached right now."""
        with self._lock:
            return len(self._by_doc.get(doc_id, _NO_BLOCKS))

    def cached_indices(self, doc_id):
        with self._lock:
            return sorted(self._by_doc.get(doc_id, _NO_BLOCKS))

    def voice_counts(self, doc_id):
        """{voice_id: how many of this document's blocks are cached in it}.

        A block is only ever held in one voice at a time (see the class note),
        so trying two voices mid-read leaves a document split across them.
        This is what lets the export dialog say which voice is already mostly
        made - choosing it is the difference between stitching cached audio
        and re-synthesizing the whole book."""
        with self._lock:
            counts = {}
            for v in self._by_doc.get(doc_id, _NO_BLOCKS).values():
                counts[v] = counts.get(v, 0) + 1
            return counts

    def note_identity(self, identity):
        """Called on the synthesis path. Empties the cache if the thing making
        the audio has changed underneath it.

        Reachable in one real way: the optional ONNX backend loads for one
        worker epoch and fails for the next (an asset moved, a session that
        will not start), so the process falls back to PyTorch. Those two do
        not produce the same bytes, and half a book in each is a book that
        changes voice at a paragraph boundary for no reason the reader can
        see.

        Returns True on the call that actually emptied it, so a caller holding
        audio anywhere else - the coordinator's prefetch handoff - can drop
        that too. Anything made by the old producer has to go at the same
        moment, or an export in flight stitches both into one file."""
        if not identity:
            return False                # no worker yet; nothing was made by it
        with self._lock:
            if self._identity == identity:
                return False
            changed = self._identity is not None
            self._identity = identity
            if not changed:
                return False
            self._entries.clear()
            self._by_doc.clear()
            self._bytes = 0
            return True

    def drop_document(self, doc_id):
        with self._lock:
            held = self._by_doc.get(doc_id)
            if not held:
                return
            for index, voice in list(held.items()):
                self._drop((doc_id, index, voice))

    def drop_after(self, doc_id, index):
        """Forget every block past `index` - what a voice switch needs."""
        with self._lock:
            held = self._by_doc.get(doc_id)
            if not held:
                return
            for i, voice in [kv for kv in held.items() if kv[0] > index]:
                self._drop((doc_id, i, voice))

    def stats(self):
        with self._lock:
            return {"bytes": self._bytes, "max_bytes": self.max_bytes,
                    "entries": len(self._entries), "evictions": self.evictions}


@dataclass(order=True)
class WarmRequest:
    """One block to synthesize. Only priority and sequence take part in the
    ordering - everything else is payload, and comparing two job objects would
    raise the moment two requests tied on priority."""
    priority: int
    sequence: int
    doc_id: str = field(compare=False)
    block_index: int = field(compare=False)
    generation: int = field(compare=False)
    voice: object = field(compare=False)
    background: bool = field(default=True, compare=False)
    # True when this request took the default tier rather than one its caller
    # named. Only those may be re-tiered in flight - an export names
    # PRIORITY_EXPORT and must keep it whichever document is open.
    tiered: bool = field(default=False, compare=False)
    job: object = field(default=None, compare=False)


# Stopping the worker has to go through the queue like everything else, and it
# has to be *comparable* with everything else - PriorityQueue heapifies, so a
# bare None sentinel raises TypeError the moment it is compared against a real
# request. Priority -1 also means shutdown does not wait behind the backlog.
_SHUTDOWN = WarmRequest(-1, -1, "", -1, -1, None, background=False, job=None)


class _Job:
    """The one synthesis of one (doc, block, voice), and whoever is waiting.

    This is what replaces the old per-block synth_locks dict: a live play
    request that arrives while the warmer is already making that same block
    attaches to this job and waits, instead of running TTS a second time.
    """
    __slots__ = ("key", "text", "voice", "event", "result", "error",
                 "claimed", "pinned", "priority", "handle", "cancel_requested",
                 "cache_result", "waiters")

    def __init__(self, key, text, voice, priority, cache_result=True):
        self.key = key
        self.text = text
        self.voice = voice
        self.priority = priority
        self.event = threading.Event()
        self.result = None
        self.error = None
        self.claimed = False        # a worker has started on it
        self.pinned = False         # somebody is blocked waiting for it
        self.waiters = 0
        # Set once this job is inside a pipelined engine, and only then. A
        # seek or a voice switch cancels through it - which is the difference
        # between "stop queueing new work" and "stop the work in flight".
        self.handle = None
        self.cancel_requested = False
        # A long export is a sequential scan over a byte-bounded LRU. Keeping
        # every newly synthesized early block would evict the already-warmed
        # later blocks just before the exporter reached them, turning a
        # partly cached book into a 100% miss. Export prefetch therefore uses
        # cache_result=False, and what it makes is handed over rather than
        # cached - see the module docstring.
        #
        # Upgrade-only: live playback attaching to a job may promote it to
        # cached, because the paragraph about to be heard is worth keeping
        # against a rewind or a browser retry. Nothing demotes a job somebody
        # else created, or an export attaching to the reader's warm work would
        # quietly throw that work away.
        self.cache_result = bool(cache_result)


class _WarmPlan:
    """The order a document gets warmed in, and how far through it we are.

    Forward from the resume point first because that is what is about to be
    played, then backwards over what came before - the same walk the old
    threaded _warm() did, just consumed a window at a time instead of all at
    once.

    wrap=False drops that backward pass. An export reads a contiguous range
    once, forward, and never looks behind it: warming what it has already
    passed would spend the engine on audio nobody will collect, and - since
    prefetch is held for its consumer rather than cached - would sit in the
    handoff window blocking the blocks the exporter is actually waiting for.
    """
    __slots__ = ("doc_id", "order", "cursor", "voice", "generation",
                 "priority", "cache_result")

    def __init__(self, doc_id, total, start, voice, generation, priority=None,
                 cache_result=True, wrap=True):
        start = max(0, min(start, max(total - 1, 0)))
        self.doc_id = doc_id
        self.order = list(range(start, total))
        if wrap:
            self.order += list(range(start - 1, -1, -1))
        self.cursor = 0
        self.voice = voice
        self.generation = generation
        self.priority = priority
        self.cache_result = bool(cache_result)


class SynthCoordinator:
    """Single-worker priority queue in front of the speech engine.

    synth_fn(text, voice) -> (audio, mime, words) is the whole dependency, so
    this module never imports engines and the engine never learns there is a
    queue. on_synth(doc_id, index, voice, audio, mime, words) is called after
    each success, outside every lock, for whatever bookkeeping the app wants
    (Talkify records the block's runtime for its progress bar).
    """

    def __init__(self, synth_fn, cache=None, on_synth=None,
                 warm_window=DEFAULT_WARM_WINDOW, max_pending=DEFAULT_MAX_PENDING,
                 submit_fn=None, max_inflight=1,
                 handoff_bytes=DEFAULT_HANDOFF_BYTES):
        self.cache = cache if cache is not None else AudioCache()
        self.warm_window = warm_window
        self.max_pending = max_pending
        self.max_handoff_bytes = max(0, int(handoff_bytes))
        self._synth_fn = synth_fn
        # submit_fn(text, voice, meta) -> handle with .result(timeout)/.cancel().
        # Optional: an engine that does not offer one is driven through
        # synth_fn exactly as before, and max_inflight should stay at 1 for it.
        self._submit_fn = submit_fn
        self.max_inflight = max(1, int(max_inflight))
        self._on_synth = on_synth
        self._text_provider = None
        self._queue = queue.PriorityQueue()
        self._lock = threading.RLock()
        self._jobs = {}             # (doc_id, index, voice) -> _Job
        self._generations = {}      # doc_id -> int
        self._plans = {}            # doc_id -> _WarmPlan
        self._failed = {}           # doc_id -> {block indices that raised}
        self._active_doc_id = None  # the document the reader actually has open
        self._outstanding = {}      # doc_id -> queued-or-running warm requests
        self._pending = 0           # entries sitting in the queue
        # (priority, sequence) of every queued request somebody is blocked on,
        # as a heap. A dispatcher that has just claimed speculative work checks
        # the smallest of these before starting it, because the priority queue
        # can only order what was already in it - see _run. A count is not
        # enough: yielding to a waiter that sorts *behind* us hands the same
        # request back to the same dispatcher forever.
        self._waiting_keys = []
        # Prefetched blocks nobody has collected yet - see the module
        # docstring. Insertion order is the order an export consumes them in.
        self._handoff = OrderedDict()   # (doc_id, index, voice) -> (audio, mime, words)
        self._handoff_bytes = 0
        self._handoff_docs = {}     # doc_id -> entries held for it
        self._sequence = itertools.count()
        self._stopped = False
        self.stats = {"synthesized": 0, "discarded": 0, "failed": 0,
                      "deduped": 0, "refused": 0, "cancelled": 0, "yielded": 0,
                      "retained": 0, "collected": 0, "retiered": 0}
        self._workers = [
            threading.Thread(target=self._run, name=f"synth-coordinator-{n}",
                             daemon=True)
            for n in range(self.max_inflight)]
        for worker in self._workers:
            worker.start()

    def scale(self, max_inflight):
        """Add dispatchers once the engine is known.

        The coordinator is built at import time and the engine is chosen in
        main(), so how many blocks may be in flight cannot be known at
        construction. Growth only, and only ever at startup: reducing would
        mean stopping a thread that is inside the engine.
        """
        want = max(1, int(max_inflight))
        with self._lock:
            if self._stopped or want <= len(self._workers):
                return
            fresh = [threading.Thread(target=self._run, daemon=True,
                                      name=f"synth-coordinator-{n}")
                     for n in range(len(self._workers), want)]
            self._workers.extend(fresh)
            self.max_inflight = want
        for worker in fresh:
            worker.start()

    def set_text_provider(self, provider):
        """provider(doc_id, index) -> str or None. Warm requests fetch their
        text at the moment they run, so the coordinator never holds a second
        copy of the document and a document deleted while its warm window is
        still queued simply finds nothing there."""
        self._text_provider = provider

    # -------------------------------------------------------- active document
    def set_active_document(self, doc_id):
        """Name the document the reader is actually looking at.

        Everything else in the library keeps warming, just at a lower tier, so
        a textbook warming in the background can no longer take half the
        worker away from the book being read. Requests already in the queue
        keep the priority they were given - re-filling here is what gets the
        *next* window of the newly active document in at the better tier, and
        the queue drains fast enough that re-prioritising in place would buy
        nothing worth the bookkeeping.

        Passing None means nothing is open, which is the honest state after
        the active document is deleted.
        """
        with self._lock:
            if self._active_doc_id == doc_id:
                return
            self._active_doc_id = doc_id
            if doc_id is not None and doc_id in self._plans:
                self._refill(doc_id)

    def active_document(self):
        with self._lock:
            return self._active_doc_id

    # ------------------------------------------------------------ generation
    def generation(self, doc_id):
        with self._lock:
            return self._generations.setdefault(doc_id, 0)

    def bump(self, doc_id):
        """A seek or a voice switch. Everything already queued for this
        document under the old number becomes discardable the moment a
        dispatcher reaches it - and anything already inside a pipelined engine
        is cancelled outright rather than left to finish for nobody.

        Pinned jobs are exempt: somebody is blocked on that exact
        (block, voice) right now, and the key names the voice, so the answer
        is still correct for them.
        """
        with self._lock:
            gen = self._generations.get(doc_id, 0) + 1
            self._generations[doc_id] = gen
            self._plans.pop(doc_id, None)
            self._failed.pop(doc_id, None)
            handles = self._stale_handles(doc_id)
        self._cancel(handles)
        return gen

    def _stale_handles(self, doc_id):
        """Caller holds self._lock. Mark this document's unpinned in-flight
        jobs cancelled and hand back their engine handles to cancel outside
        the lock. cancel_requested is also what closes the window where a job
        is inside submit() but has not stored its handle yet."""
        handles = []
        for job in self._jobs.values():
            if job.key[0] != doc_id or job.pinned or job.event.is_set():
                continue
            job.cancel_requested = True
            if job.handle is not None:
                handles.append(job.handle)
        return handles

    @staticmethod
    def _cancel(handles):
        """Never called with a lock held. Cancellation writes to the engine -
        for Kokoro that is a line down a subprocess pipe - and holding the
        coordinator lock across it would put the session lock, the coordinator
        lock and a pipe write in the same critical section."""
        for handle in handles:
            try:
                handle.cancel()
            except Exception:
                pass

    # ------------------------------------------------------------- handoff
    #
    # Audio made speculatively for a consumer that does not want it cached.
    # Everything here is called with self._lock held.
    def _retain(self, key, audio, mime, words):
        """Hold a prefetched block for whoever asked for the walk that made
        it. Newest-first eviction, because the consumer is a forward scan: the
        oldest entry is the one it is about to ask for, and dropping that is
        the re-synthesis this buffer exists to prevent."""
        size = len(audio)
        if size > self.max_handoff_bytes or key in self._handoff:
            return
        self._handoff[key] = (audio, mime, words)
        self._handoff_bytes += size
        self._handoff_docs[key[0]] = self._handoff_docs.get(key[0], 0) + 1
        self.stats["retained"] += 1
        while self._handoff_bytes > self.max_handoff_bytes and self._handoff:
            self._drop_handoff(next(reversed(self._handoff)))

    def _drop_handoff(self, key):
        entry = self._handoff.pop(key, None)
        if entry is None:
            return None
        self._handoff_bytes -= len(entry[0])
        left = self._handoff_docs.get(key[0], 0) - 1
        if left > 0:
            self._handoff_docs[key[0]] = left
        else:
            self._handoff_docs.pop(key[0], None)
        return entry

    def _collect(self, key):
        entry = self._drop_handoff(key)
        if entry is not None:
            self.stats["collected"] += 1
        return entry

    def drop_prefetch(self, doc_id=None):
        """Release anything prefetched for this document that nobody
        collected. The export that asked for the walk has finished or failed,
        so the tail of its window has no reader left.

        No document means all of them, which is what a change of producer
        needs: the cache empties itself on one, and this buffer holds audio
        the same rule applies to."""
        with self._lock:
            for key in [k for k in self._handoff
                        if doc_id is None or k[0] == doc_id]:
                self._drop_handoff(key)

    # -------------------------------------------------------------- warming
    def warm(self, doc_id, total, start=0, voice=None, *, priority=None,
             cache_result=True, wrap=True):
        """(Re)aim the background warmer at a document. Cheap to call often -
        reopening a document re-plans from the new resume point rather than
        starting another walk to race the first one.

        `priority`, `cache_result` and `wrap` are for bounded consumers that
        use the warm window as prefetch. In particular, export keeps the
        engine fed but does not admit its misses to the LRU: a sequential scan
        must not evict warmed blocks that are still ahead of it. What it makes
        early is handed to it later instead - see the module docstring."""
        if total <= 0:
            return
        with self._lock:
            if self._stopped:
                return
            gen = self._generations.setdefault(doc_id, 0)
            self._plans[doc_id] = _WarmPlan(
                doc_id, total, start, voice, gen,
                priority=priority, cache_result=cache_result, wrap=wrap)
            self._refill(doc_id)

    def extend(self, doc_id, old_total, new_total, voice=None):
        """More blocks exist than the current plan knows about.

        Progressive ingestion appends to a document while it is being read, and
        a plan precomputes its whole order, so without this the blocks that
        arrive after the plan was built would never be warmed. Appending to a
        compatible plan keeps the cursor - and therefore the window and the
        dedup - exactly where they were.
        """
        if new_total <= old_total:
            return
        with self._lock:
            if self._stopped:
                return
            gen = self._generations.setdefault(doc_id, 0)
            plan = self._plans.get(doc_id)
            if plan is not None and plan.voice == voice and plan.generation == gen:
                plan.order.extend(range(old_total, new_total))
            else:
                plan = _WarmPlan(doc_id, new_total, old_total, voice, gen)
                plan.order = list(range(old_total, new_total))
                self._plans[doc_id] = plan
            self._refill(doc_id)

    def _refill(self, doc_id):
        """Caller holds self._lock. Top this document's warm window back up.

        Two ceilings, and between them they are the release gate on pending
        work: at most warm_window blocks outstanding for any one document, and
        at most max_pending entries in the queue across all of them.

        Audio already prefetched and not yet collected counts against the
        first of those. That is the back-pressure on a producer whose consumer
        has stalled: without it, an export pausing to encode would let the warm
        window refill on every completion and the handoff buffer would grow
        with no ceiling but its own byte cap.
        """
        plan = self._plans.get(doc_id)
        if plan is None:
            return
        # Which tier this document's warm work joins. Read once per refill:
        # the answer cannot change while we hold the lock.
        tiered = plan.priority is None
        is_active = doc_id == self._active_doc_id
        warm_priority = (plan.priority if not tiered else
                         (PRIORITY_ACTIVE_WARM if is_active
                          else PRIORITY_BACKGROUND_WARM))
        # Tiering orders the queue; it does not decide who gets *into* it, and
        # max_pending is a hard admission cap. load_library() warms every
        # document at startup, so thirty books at a warm window of eight fill
        # 240 of the 256 slots before the reader has opened anything - and
        # each completed background block re-filled its own window straight
        # back into the slot it had just freed, so the open document's window
        # was refused over and over while a queue full of background work
        # drained one block at a time. Reserving one window for whatever is
        # open costs the library nothing measurable and means the book being
        # read can always queue. A caller that named its own priority (an
        # export) is not speculative and is not held back.
        ceiling = (self.max_pending if is_active or not tiered
                   else max(1, self.max_pending - self.warm_window))
        while (self._outstanding.get(doc_id, 0)
               + self._handoff_docs.get(doc_id, 0)) < self.warm_window:
            if self._pending >= ceiling:
                self.stats["refused"] += 1
                return                              # try again as the queue drains
            if plan.cursor >= len(plan.order):
                self._plans.pop(doc_id, None)       # fully warmed, nothing left
                return
            index = plan.order[plan.cursor]
            plan.cursor += 1
            if index in self._failed.get(doc_id, ()):
                # This block raised once already. Speculative work does not get
                # to fail forever; an explicit request for it, a voice switch
                # or a seek clears the mark and lets it be tried again.
                continue
            key = (doc_id, index, plan.voice)
            existing_job = self._jobs.get(key)
            if existing_job is not None and existing_job.claimed and existing_job.event.is_set():
                self._jobs.pop(key, None)
                existing_job = None
            if (existing_job is not None or key in self._handoff
                    or self.cache.get(doc_id, index, voice=plan.voice) is not None):
                self.stats["deduped"] += 1
                continue                            # already made, or being made
            job = _Job(key, None, plan.voice, warm_priority,
                       cache_result=plan.cache_result)
            self._jobs[key] = job
            self._outstanding[doc_id] = self._outstanding.get(doc_id, 0) + 1
            self._enqueue(WarmRequest(warm_priority, next(self._sequence), doc_id,
                                      index, plan.generation, plan.voice,
                                      background=True, tiered=tiered, job=job))

    def _enqueue(self, request):
        """Caller holds self._lock."""
        self._pending += 1
        if not request.background:
            heapq.heappush(self._waiting_keys,
                           (request.priority, request.sequence))
        self._queue.put(request)

    def _pop_waiting(self, request):
        """Caller holds self._lock. This foreground request has been claimed.

        Removing the exact key rather than the smallest one costs a scan of a
        list that is as long as the number of callers blocked right now - a
        handful - and means _outranked() never compares against a request that
        is already being served."""
        key = (request.priority, request.sequence)
        try:
            self._waiting_keys.remove(key)
        except ValueError:
            return
        heapq.heapify(self._waiting_keys)

    def _outranked(self, request):
        """Caller holds self._lock. Is somebody blocked on a request that
        sorts strictly ahead of this speculative one?

        Strictly is what makes yielding terminate. A dispatcher that puts a
        request back has to be sure the next get() returns something else, and
        the only thing that guarantees that is knowing a smaller entry is in
        the queue. Yielding on a tie - two export-priority requests, one
        speculative and one blocked, the blocked one queued later - hands the
        same request straight back and the dispatcher spins on it forever.
        """
        return (bool(self._waiting_keys)
                and self._waiting_keys[0] < (request.priority, request.sequence))

    def _demoted(self, request):
        """Caller holds self._lock. Has this speculative request outlived the
        tier it was given, and if so what should it be?  Returns the new
        priority, or None to run it as it stands.

        set_active_document() re-aims the *next* window at the better tier but
        cannot reorder a heap, so the outgoing document keeps a full window of
        PRIORITY_ACTIVE_WARM requests in the queue. Those tie with the newly
        opened book's window and win on arrival order, so switching documents
        handed the worker back to the book just closed for eight blocks -
        about a minute of real Kokoro, spent entirely on text nobody is
        reading. Re-tiering at the moment a dispatcher claims it costs no scan
        of the queue and needs nothing kept in sync.

        Demotion only. Promotion at claim time would be a no-op by
        construction: a request that has just been popped is already the most
        urgent thing in the queue. It also cannot loop - the request goes back
        carrying the priority it was demoted to, so the next dispatcher to
        reach it finds nothing left to change.
        """
        if not request.tiered:
            return None
        want = (PRIORITY_ACTIVE_WARM if request.doc_id == self._active_doc_id
                else PRIORITY_BACKGROUND_WARM)
        return want if want > request.priority else None

    # ------------------------------------------------------------- blocking
    def synth(self, doc_id, index, text, voice, priority=PRIORITY_INTERACTIVE,
              timeout=None, cache_result=True):
        """Get this block's audio, synthesizing it if nobody has yet.

        Returns (audio, mime, words). Raises whatever the engine raised, so
        the HTTP layer reports failures exactly the way it always has.
        `cache_result=False` is a cache-admission hint for one-pass scans such
        as export; existing hits are still returned. Interactive priority
        always retains a shared result when its caller asks for caching.

        A caller who is blocked waiting is never refused for exceeding
        max_pending - it is already bounded by how many requests the browser
        and the export thread can have in flight, and refusing it would just
        turn a wait into an error. What it does instead is count toward
        _pending, so a burst of live requests makes _refill stop adding warm
        work until the burst clears. That is the back-pressure working in the
        right direction.
        """
        with self._lock:
            if self._stopped:
                raise RuntimeError("coordinator is stopped")

        entry = self.cache.get(doc_id, index, voice=voice)
        if entry is not None:
            return entry.audio, entry.mime, entry.words

        with self._lock:
            if self._stopped:
                raise RuntimeError("coordinator is stopped")
            # Re-check under the lock. The worker writes to the cache *before*
            # it clears the job, so a hit here is authoritative even if the
            # job has just been retired out from under us.
            entry = self.cache.get(doc_id, index, voice=voice)
            if entry is not None:
                return entry.audio, entry.mime, entry.words
            key = (doc_id, index, voice)
            # Prefetched for this walk and finished before it got here. The
            # collector's own admission policy decides where it goes next: an
            # export leaves the LRU alone, while a reader that happens to want
            # the same block keeps it, exactly as if it had made it. Taking it
            # frees a slot in the window, so top the plan back up before
            # letting go of the lock.
            held = self._collect(key)
            if held is not None:
                if cache_result:
                    self.cache.put(doc_id, index, voice, *held)
                self._refill(doc_id)
                return held
            job = self._jobs.get(key)
            if job is None:
                job = _Job(key, text, voice, priority,
                           cache_result=cache_result)
                job.pinned = True
                job.waiters = 1
                self._jobs[key] = job
                gen = self._generations.setdefault(doc_id, 0)
                self._enqueue(WarmRequest(priority, next(self._sequence), doc_id,
                                          index, gen, voice, background=False,
                                          job=job))
            else:
                # Somebody (usually the warmer) is already on it. Wait for
                # that rather than duplicating the work, and if we are more
                # urgent than they were, queue a second, higher-priority
                # pointer at the same job so it gets picked up sooner. The
                # worker ignores whichever copy arrives second.
                job.pinned = True
                job.waiters = getattr(job, "waiters", 0) + 1
                if priority == PRIORITY_INTERACTIVE and cache_result:
                    # Playback owns the strongest admission claim: retaining
                    # the paragraph it is about to play prevents an immediate
                    # repeat synthesis after a browser retry or rewind. The
                    # other direction is deliberately absent - an export
                    # attaching to warm work does not get to un-cache it.
                    job.cache_result = True
                if job.text is None:
                    job.text = text
                if priority < job.priority and not job.claimed:
                    job.priority = priority
                    self._enqueue(WarmRequest(priority, next(self._sequence), doc_id,
                                              index, self._generations.get(doc_id, 0),
                                              voice, background=False, job=job))

        try:
            if not job.event.wait(timeout):
                raise TimeoutError(f"synthesis of block {index} timed out")
            if job.error is not None:
                raise job.error
            if job.result is None:
                raise RuntimeError(f"synthesis of block {index} produced nothing")
            return job.result
        finally:
            with self._lock:
                job.waiters = max(0, getattr(job, "waiters", 1) - 1)
                job.pinned = (job.waiters > 0)

    # --------------------------------------------------------------- worker
    def _run(self):
        while True:
            request = self._queue.get()
            with self._lock:
                self._pending -= 1
                if request is not _SHUTDOWN and not request.background:
                    self._pop_waiting(request)
                # The priority queue can only order what was in it when this
                # dispatcher called get(). With more than one dispatcher there
                # is a window - small, but exactly the width of a page turn -
                # where a request somebody is blocked on arrives just after a
                # speculative one was claimed. Put the speculative one back and
                # take the urgent one instead; _outranked() is what makes that
                # safe, by only ever yielding to a request that really is ahead
                # of us in the queue.
                speculative = (request is not _SHUTDOWN and request.background
                               and not self._stopped)
                requeue = speculative and self._outranked(request)
                if requeue:
                    self.stats["yielded"] += 1
                elif speculative:
                    # Still ours to run, but the document it belongs to may
                    # have stopped being the one on screen while it queued.
                    demoted = self._demoted(request)
                    if demoted is not None:
                        request.priority = demoted
                        self.stats["retiered"] += 1
                        requeue = True
                if requeue:
                    self._pending += 1
                    self._queue.put(request)
            if requeue:
                continue
            if request is _SHUTDOWN:
                return
            try:
                self._process(request)
            except Exception:
                # A dispatcher outliving one bad block matters more than the
                # block does; per-block failures are recorded on the job and
                # raised to whoever is waiting on it.
                pass

    def _process(self, request):
        doc_id = request.doc_id
        job = request.job
        try:
            with self._lock:
                if job.event.is_set() or job.claimed:
                    return                       # a duplicate, higher-priority copy
                current = self._generations.get(doc_id, 0)
                if request.generation < current and not job.pinned:
                    # Stale: the voice changed or the reader seeked away while
                    # this sat in the queue. Drop it without synthesizing.
                    # Pinned jobs are exempt because somebody is blocked on
                    # this exact (block, voice) right now and the answer is
                    # still correct for them - the key names the voice.
                    self.stats["discarded"] += 1
                    self._jobs.pop(job.key, None)
                    job.error = StaleRequest(f"block {request.block_index} superseded")
                    job.event.set()
                    return
                job.claimed = True
                text = job.text

            if text is None:
                text = self._text_for(doc_id, request.block_index)
            if text is None:
                with self._lock:
                    self._jobs.pop(job.key, None)
                job.error = KeyError("no such document or block")
                job.event.set()
                return

            audio, mime, words = self._make(request, job, text)

            with self._lock:
                # Check staleness *again*, now that the slow part is over. A
                # voice switch that lands while a block is mid-synthesis is
                # the one window where old-voice audio could otherwise get
                # written into the cache behind the eviction that switch just
                # did - which is exactly the "cannot publish stale audio"
                # release gate. Nobody is waiting on this one, so dropping it
                # costs only the work already done.
                superseded = (request.generation < self._generations.get(doc_id, 0)
                              and not job.pinned)
                if superseded:
                    self.stats["discarded"] += 1
                    self._jobs.pop(job.key, None)
                    job.error = StaleRequest(f"block {request.block_index} superseded")
                    job.event.set()
                    return

                # Put inside lock to prevent race with voice-switch drop_after()
                if job.cache_result:
                    self.cache.put(doc_id, request.block_index, job.voice,
                                   audio, mime, words)
                elif not job.pinned:
                    # Prefetched for a scan that has not arrived yet. Nobody is
                    # waiting on this result, so dropping it here is dropping
                    # a block of Kokoro - and the consumer that asked for it
                    # would then synthesize the very same block again.
                    self._retain(job.key, audio, mime, words)
                self._jobs.pop(job.key, None)
                self.stats["synthesized"] += 1

            job.result = (audio, mime, words)
            job.event.set()
            if self._on_synth is not None:
                try:
                    self._on_synth(doc_id, request.block_index, job.voice,
                                   audio, mime, words)
                except Exception:
                    pass
        except Exception as e:
            with self._lock:
                self._jobs.pop(job.key, None)
                cancelled = (job.cancel_requested
                             or getattr(e, "code", None) == "cancelled")
                self.stats["cancelled" if cancelled else "failed"] += 1
                if getattr(e, "fatal", False):
                    # The engine itself is broken - a missing kokoro venv, a
                    # model that will not load. Every block would fail the
                    # same way, so stop warming this document rather than
                    # producing one failure per paragraph for the rest of a
                    # book.
                    self._plans.pop(doc_id, None)
                elif not cancelled:
                    # One block failed. Data-dependent failures exist, and a
                    # book is not abandoned for one of them - the index is
                    # remembered so speculative work does not retry it
                    # forever, and a seek, a voice switch or an explicit
                    # request clears the mark.
                    failed = self._failed.setdefault(doc_id, set())
                    if len(failed) < MAX_FAILED_BLOCKS:
                        failed.add(request.block_index)
            job.error = e
            job.event.set()
        finally:
            with self._lock:
                if request.background:
                    left = self._outstanding.get(doc_id, 0) - 1
                    if left > 0:
                        self._outstanding[doc_id] = left
                    else:
                        self._outstanding.pop(doc_id, None)
                # The open document gets first claim on whatever room the
                # finished block just freed - before the document that freed
                # it, which is the case that actually bit. A background book
                # completing a block refilled its own window into the slot it
                # had just vacated, so under a full queue the reader's window
                # was refused every time and the library drained one block at
                # a time in front of it. The reservation in _refill() is the
                # structural half of that fix; taking the slot first is the
                # cheap half.
                active = self._active_doc_id
                if active and active != doc_id and active in self._plans:
                    self._refill(active)
                self._refill(doc_id)
                if self._plans and self._pending < self.max_pending:
                    for other_doc in list(self._plans.keys()):
                        if other_doc != doc_id and other_doc != active:
                            self._refill(other_doc)
                            if self._pending >= self.max_pending:
                                break

    def _make(self, request, job, text):
        """Run one synthesis, outside every lock this object owns.

        With a pipelined engine the handle is stored on the job so that a seek
        arriving mid-synthesis can cancel it. The window between submitting and
        storing it is the reason for cancel_requested: a bump() that lands in
        that window sets the flag and finds no handle, and this closes it by
        cancelling the moment the handle exists.
        """
        if self._submit_fn is None:
            return self._synth_fn(text, job.voice)
        meta = SynthMeta(doc_id=request.doc_id, block_index=request.block_index,
                         generation=request.generation, priority=request.priority,
                         background=request.background)
        handle = self._submit_fn(text, job.voice, meta)
        with self._lock:
            job.handle = handle
            late = job.cancel_requested
        if late:
            self._cancel([handle])
        try:
            return handle.result()
        finally:
            with self._lock:
                job.handle = None

    def _text_for(self, doc_id, index):
        if self._text_provider is None:
            return None
        try:
            return self._text_provider(doc_id, index)
        except Exception:
            return None

    # ------------------------------------------------------------ plumbing
    def forget(self, doc_id):
        """A document was deleted: drop its audio and stop warming it.

        Deliberately does *not* reset the outstanding counter or the
        generation. Requests for this document may still be sitting in the
        queue, and each one decrements the counter as it drains; zeroing it
        here would let those decrements run against a fresh count if the id
        ever came back, so _refill would think fewer blocks were in flight
        than really are and overshoot the window. They drain on their own in
        bounded time - and they cost nothing when they do, because the text
        provider no longer has the document, so each one fails out before it
        reaches the engine.
        """
        with self._lock:
            self._plans.pop(doc_id, None)
            self._failed.pop(doc_id, None)
            if self._active_doc_id == doc_id:
                self._active_doc_id = None
            handles = self._stale_handles(doc_id)
        self._cancel(handles)
        self.drop_prefetch(doc_id)
        self.cache.drop_document(doc_id)

    def voice_counts(self, doc_id):
        """{voice: cached block count} for this document - see AudioCache."""
        return self.cache.voice_counts(doc_id)

    def pending(self):
        with self._lock:
            return self._pending

    def prefetched(self, doc_id=None):
        """Blocks made ahead of a scan and not yet collected - a bounded
        buffer, so the count is what the release gate on it is written
        against."""
        with self._lock:
            if doc_id is None:
                return len(self._handoff)
            return self._handoff_docs.get(doc_id, 0)

    def outstanding(self, doc_id=None):
        with self._lock:
            if doc_id is None:
                return sum(self._outstanding.values())
            return self._outstanding.get(doc_id, 0)

    def shutdown(self, timeout=2.0):
        with self._lock:
            self._stopped = True
            self._plans.clear()
            self._failed.clear()
            self._handoff.clear()
            self._handoff_docs.clear()
            self._handoff_bytes = 0
            self._pending += len(self._workers)
        for _ in self._workers:
            self._queue.put(_SHUTDOWN)
        for worker in self._workers:
            worker.join(timeout)


class PreviewCache:
    """The voice gallery's one-line-per-voice samples.

    Nothing to do with document playback, so it stays out of the byte-bounded
    cache: there are ten voices and one short sentence each, and evicting them
    to make room for a chapter of audio would only mean re-synthesizing them
    the next time the gallery opened. Failures are not cached, so a voice that
    errors once can be tried again.

    Two voices are made concurrently only if two callers ask at once, and two
    callers asking for the *same* voice make it once: the second waits on the
    first's event rather than running the engine again. The global lock is
    only ever held around dict updates - never across the engine call - so a
    slow Kokoro warm of one voice does not stall a hit on another.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._items = {}
        self._making = {}       # voice -> Event, set when that voice finishes

    def ready(self):
        """Which voices can be served with no synthesis at all."""
        with self._lock:
            return sorted(self._items)

    def has(self, voice):
        with self._lock:
            return voice in self._items

    def peek(self, voice):
        with self._lock:
            return self._items.get(voice)

    def get_or_make(self, voice, make):
        return self.get_with_status(voice, make)[0]

    def get_with_status(self, voice, make):
        """(value, cached) - `cached` is False only for the caller that
        actually ran `make`. That flag is what lets the preview route tell the
        browser (and smoke.py) whether it served from memory."""
        while True:
            with self._lock:
                if voice in self._items:
                    return self._items[voice], True
                waiting = self._making.get(voice)
                if waiting is None:
                    event = threading.Event()
                    self._making[voice] = event
                    break
            # Somebody else is already synthesizing this voice. Wait for them
            # and re-check rather than making a second copy of the same clip.
            waiting.wait(60)
            with self._lock:
                if voice in self._items:
                    return self._items[voice], True
                if self._making.get(voice) is waiting:
                    # They failed and left; take the slot ourselves.
                    self._making.pop(voice, None)
        try:
            value = make()
        except BaseException:
            with self._lock:
                self._making.pop(voice, None)
                event.set()
            raise
        with self._lock:
            # Publish before waking waiters.  Waking first creates a small
            # race in which a waiter observes neither the item nor an active
            # producer and starts the same preview a second time.
            self._items[voice] = value
            self._making.pop(voice, None)
            event.set()
        return value, False

    def clear(self):
        with self._lock:
            self._items.clear()
