"""TKF1 - the framed binary protocol between Talkify and its Kokoro worker.

Standard library only, and imported by both sides: the host on the project's
normal Python, and kokoro_worker.py inside kokoro_venv (3.12). Nothing here may
grow a third-party import, because one of those two interpreters would not have
it.

Why a second transport at all
-----------------------------
The JSONL transport carries a WAV as base64 inside a JSON string. That is a
third more bytes on the pipe, and - worse - three copies of the same audio
alive at once on each side while the encode and the decode run. A framed
protocol sends the samples exactly as they came out of the model.

The frame
---------
A fixed 20-byte big-endian header, then compact UTF-8 JSON metadata, then an
opaque body:

    offset  size  meaning
    0       4     ASCII TKF1
    4       1     version, 1
    5       1     frame kind
    6       2     flags, zero in v1
    8       4     unsigned metadata length
    12      8     unsigned body length

Both lengths are validated against a cap *before* anything is allocated, so a
corrupt length field cannot ask this process for four gigabytes.

The rules that keep it honest
-----------------------------
**There is no resynchronisation.** A WAV body can contain the bytes ``TKF1``,
so scanning forward for the magic after a bad frame would eventually lock onto
audio and hand back nonsense that looks structurally valid. Any corruption,
any short frame, any unknown version is fatal to the connection: the caller
retires that worker epoch and completes every pending request exactly once.
The magic is an assertion, not a search target.

**EOF is only clean before a header starts.** A worker that exits between
requests is normal. A worker that dies halfway through a header, metadata or
body has left a partial frame in the pipe and the stream is meaningless from
there on.

**One writer at a time, for the whole frame.** Header, metadata, body and
flush go out under one lock. Two threads interleaving their frames would
produce a stream that parses - into the wrong pairing of metadata and audio.

**Metadata is a JSON object with unique keys.** ``object_pairs_hook`` rejects
duplicates rather than letting the last one win, because "the last one wins"
is precisely how a smuggled field overrides a checked one.
"""
from __future__ import annotations

import json
import struct
import threading

MAGIC = b"TKF1"
VERSION = 1

# >4sBBHIQ - magic, version, kind, flags, metadata length, body length.
HEADER = struct.Struct(">4sBBHIQ")
HEADER_SIZE = HEADER.size          # 20

# Metadata is small by construction: ids, a sample rate, and word timings for
# one paragraph. A megabyte is already far more than any real block needs.
MAX_METADATA = 1 << 20             # 1 MiB
# One block of speech is a few hundred KB of PCM. 32 MiB is roughly ten
# minutes of 24 kHz mono - orders of magnitude past a paragraph, and small
# enough that a corrupt length cannot exhaust memory.
MAX_BODY = 32 << 20                # 32 MiB

# Frame kinds.
HELLO = 0x01
HELLO_ACK = 0x02
SYNTHESIZE = 0x10
CANCEL = 0x11
SHUTDOWN = 0x12
RESULT = 0x20
ERROR = 0x21

KIND_NAMES = {
    HELLO: "hello", HELLO_ACK: "hello_ack", SYNTHESIZE: "synthesize",
    CANCEL: "cancel", SHUTDOWN: "shutdown", RESULT: "result", ERROR: "error",
}

# Only RESULT carries a body. Everything else is metadata alone, and a body on
# one of them means the sender and this reader disagree about the protocol.
BODY_KINDS = frozenset({RESULT})

# Frames that name the request they belong to. HELLO/HELLO_ACK/SHUTDOWN are
# about the connection, not about any one request.
CORRELATED_KINDS = frozenset({SYNTHESIZE, CANCEL, RESULT, ERROR})


class ProtocolError(Exception):
    """The stream cannot be trusted from here on.

    Always fatal to the connection. There is deliberately no "recoverable
    protocol error": every caller's correct response is to retire the worker
    and fail its outstanding requests, and offering a second option would
    invite a resync that cannot be done safely.
    """


class StreamClosed(Exception):
    """Clean EOF: the peer closed the stream between frames.

    Distinct from ProtocolError on purpose. A worker that has finished and
    exited is not a corrupt worker, and the parent must be able to tell the
    difference to decide whether anything was lost.
    """


class Frame:
    """One decoded frame. ``meta`` is always a dict; ``body`` is always bytes."""

    __slots__ = ("kind", "meta", "body")

    def __init__(self, kind, meta, body=b""):
        self.kind = kind
        self.meta = meta
        self.body = body

    @property
    def req_id(self):
        value = self.meta.get("req_id")
        return value if isinstance(value, str) else None

    def __repr__(self):
        return (f"<Frame {KIND_NAMES.get(self.kind, hex(self.kind))} "
                f"meta={len(self.meta)} body={len(self.body)}>")


def _check_fields(kind, meta, body):
    """The fields each kind must actually carry, checked on the way in.

    Structure is not enough. A RESULT with a valid header, a valid JSON object
    and an empty body parses perfectly and means "here is your audio" - and
    what comes back is a zero-length clip that the timing code then fills in
    proportionally, so a broken worker reads as a very fast one. Every field
    below is one the receiving side goes on to use without asking again.
    """
    if kind == RESULT:
        rate = meta.get("sample_rate")
        if not isinstance(rate, int) or isinstance(rate, bool) or rate <= 0:
            raise ProtocolError("result frame has no usable sample rate")
        if not isinstance(meta.get("mime_type"), str):
            raise ProtocolError("result frame has no mime type")
        if not isinstance(meta.get("words"), list):
            raise ProtocolError("result frame has no word list")
        if not body:
            raise ProtocolError("result frame carries no audio")
    elif kind == ERROR:
        if not isinstance(meta.get("error"), dict):
            raise ProtocolError("error frame carries no error object")
    elif kind == SYNTHESIZE:
        if not isinstance(meta.get("text"), str):
            raise ProtocolError("synthesize frame has no text")
    elif kind == HELLO_ACK:
        if meta.get("version") != VERSION:
            raise ProtocolError(
                f"hello acknowledged for version {meta.get('version')!r}")


def _no_duplicates(pairs):
    seen = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate metadata key {key!r}")
        seen[key] = value
    return seen


def encode_metadata(meta):
    """Compact UTF-8 JSON for one object, or ProtocolError.

    ``ensure_ascii=False`` because the words in a block are real text and
    escaping every non-ASCII character would inflate the metadata of exactly
    the documents this project exists to read.
    """
    if not isinstance(meta, dict):
        raise ProtocolError("frame metadata must be an object")
    try:
        raw = json.dumps(meta, ensure_ascii=False,
                         separators=(",", ":")).encode("utf-8")
    except (TypeError, ValueError) as e:
        raise ProtocolError(f"frame metadata is not serialisable: {e}") from e
    if len(raw) > MAX_METADATA:
        raise ProtocolError("frame metadata is over the size cap")
    return raw


def decode_metadata(raw):
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        raise ProtocolError(f"frame metadata is not UTF-8: {e}") from e
    try:
        meta = json.loads(text, object_pairs_hook=_no_duplicates)
    except ValueError as e:
        raise ProtocolError(f"frame metadata is not JSON: {e}") from e
    if not isinstance(meta, dict):
        raise ProtocolError("frame metadata must be an object")
    return meta


def read_exact(stream, count):
    """Exactly `count` bytes, or StreamClosed/ProtocolError.

    A pipe read returns what is available, not what was asked for: a 400 KB
    body arrives in however many chunks the OS felt like, and a header can be
    split across two reads when the writer flushed mid-frame. Looping is not
    an optimisation here, it is the difference between working and silently
    truncating.
    """
    if count == 0:
        return b""
    buf = bytearray(count)
    view = memoryview(buf)
    got = 0
    while got < count:
        try:
            n = stream.readinto(view[got:])
        except AttributeError:
            chunk = stream.read(count - got)
            n = len(chunk) if chunk else 0
            if n:
                view[got:got + n] = chunk
        if not n:
            if got == 0:
                raise StreamClosed("stream closed")
            raise ProtocolError(
                f"stream closed {got} of {count} bytes into a frame")
        got += n
    return bytes(buf)


def read_frame(stream):
    """The next frame, or StreamClosed at a clean boundary.

    EOF is clean only before the first byte of a header. Anywhere after that
    the peer died mid-frame, and what remains in the pipe is not a frame.
    """
    header = read_exact(stream, HEADER_SIZE)
    magic, version, kind, flags, meta_len, body_len = HEADER.unpack(header)
    if magic != MAGIC:
        # Never scan forward for the magic. A WAV body can contain these four
        # bytes, and locking onto one would produce a frame that parses and
        # means nothing.
        raise ProtocolError(f"bad frame magic {magic!r}")
    if version != VERSION:
        raise ProtocolError(f"unsupported protocol version {version}")
    if kind not in KIND_NAMES:
        raise ProtocolError(f"unknown frame kind {kind:#x}")
    if flags:
        raise ProtocolError(f"unsupported frame flags {flags:#x}")
    if meta_len > MAX_METADATA:
        raise ProtocolError("frame metadata is over the size cap")
    if body_len > MAX_BODY:
        raise ProtocolError("frame body is over the size cap")
    if body_len and kind not in BODY_KINDS:
        raise ProtocolError(
            f"{KIND_NAMES[kind]} frames carry no body")
    meta = decode_metadata(read_exact(stream, meta_len))
    body = read_exact(stream, body_len)
    if kind in CORRELATED_KINDS and not isinstance(meta.get("req_id"), str):
        raise ProtocolError(f"{KIND_NAMES[kind]} frame has no request id")
    _check_fields(kind, meta, body)
    return Frame(kind, meta, body)


def write_all(stream, data):
    """Write every byte. `write` on a pipe may take a prefix and return."""
    if not data:
        return
    view = memoryview(data)
    sent = 0
    total = len(view)
    while sent < total:
        n = stream.write(view[sent:])
        if n is None:
            # A buffered writer reports nothing and takes everything.
            break
        if n <= 0:
            raise ProtocolError("stream refused to accept bytes")
        sent += n


class FrameWriter:
    """Serialises whole frames onto one stream.

    The lock is held across header, metadata, body and flush - not per write.
    Two threads each writing "a frame" without it produce a byte stream that
    still parses, into results attached to the wrong requests, which is the
    one failure mode this protocol exists to make impossible.
    """

    def __init__(self, stream):
        self._stream = stream
        self._lock = threading.Lock()

    @property
    def lock(self):
        return self._lock

    def write(self, kind, meta, body=b""):
        if kind not in KIND_NAMES:
            raise ProtocolError(f"unknown frame kind {kind!r}")
        if body and kind not in BODY_KINDS:
            raise ProtocolError(f"{KIND_NAMES[kind]} frames carry no body")
        if kind in CORRELATED_KINDS and not isinstance(meta.get("req_id"), str):
            raise ProtocolError(f"{KIND_NAMES[kind]} frame needs a request id")
        raw_meta = encode_metadata(meta)
        body = b"" if body is None else bytes(body)
        if len(body) > MAX_BODY:
            raise ProtocolError("frame body is over the size cap")
        header = HEADER.pack(MAGIC, VERSION, kind, 0, len(raw_meta), len(body))
        with self._lock:
            write_all(self._stream, header)
            write_all(self._stream, raw_meta)
            write_all(self._stream, body)
            self._stream.flush()


def hello(protocol, backend, native_timing, model_identity):
    """The worker's opening frame: what it is, before it is asked for anything.

    ``model_identity`` is a short hash prefix, never a path - the parent uses
    it to decide whether audio made by this worker can share a cache entry
    with audio made by the last one, and a filename would leak the user's
    directory layout into a cache key for no benefit.
    """
    return {
        "protocol": protocol,
        "backend": backend,
        "native_timing": bool(native_timing),
        "model_identity": model_identity,
        "max_metadata": MAX_METADATA,
        "max_body": MAX_BODY,
        "version": VERSION,
    }


def check_hello(meta, *, expect_protocol=None):
    """Validate a HELLO before any work is admitted. Returns the meta."""
    if not isinstance(meta.get("protocol"), str):
        raise ProtocolError("hello has no protocol name")
    if expect_protocol is not None and meta["protocol"] != expect_protocol:
        raise ProtocolError(
            f"worker speaks {meta['protocol']!r}, not {expect_protocol!r}")
    if meta.get("version") != VERSION:
        raise ProtocolError(f"worker speaks version {meta.get('version')!r}")
    if not isinstance(meta.get("backend"), str):
        raise ProtocolError("hello has no backend name")
    if not isinstance(meta.get("native_timing"), bool):
        # Absent duration output is reported as native_timing false. It is a
        # capability answer, not a protocol failure - the worker falls back to
        # Whisper and the reader never knows.
        raise ProtocolError("hello has no native_timing flag")
    return meta
