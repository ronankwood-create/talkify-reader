#!/usr/bin/env python3
"""Runs inside kokoro_venv/ (Python 3.12) - the only place that imports
kokoro/spacy/faster-whisper. Talkify itself stays on the project's normal
Python; those libraries have no wheels for it yet, so
KokoroEngine in engines.py talks to this as a long-lived subprocess instead.

One JSON request per line in on stdin, one JSON response per line out on
stdout. Models load lazily and stay warm for the life of the process, so
repeat calls (one per paragraph block) don't each pay model-load cost.

    {"op": "synth", "req_id": "a1", "text": "...", "voice": "af_heart",
     "rate": 1.0, "whisper_model": "base.en", "priority": 0}
 -> {"req_id": "a1", "ok": true, "audio_b64": "...",
     "heard": [{"w": "...", "s": 0.0, "e": 0.4}, ...]}
 -> {"req_id": "a1", "ok": false,
     "error": {"code": "...", "stage": "tts|asr|queued",
               "message": "...", "retryable": false, "fatal": false}}

    {"op": "cancel", "req_id": "a1"}      advisory, idempotent, never answered
                                          twice - whichever stage notices the
                                          flag emits the one `cancelled` reply
    {"op": "shutdown"}                    drain and exit

Why request IDs, and why two stages
-----------------------------------
Speaking a normal block is now Kokoro (TTS plus native duration timing).
Whisper (ASR) remains the second-stage fallback for missing or invalid native
marks. Keeping the stages separate still lets block i+1 reach Kokoro while a
fallback alignment for block i is running.

That only works if responses stop being positional. With two requests alive
the second can finish first (a cancel, a failed stage), so every reply names
the request it belongs to and the parent dispatches on that name - never on
arrival order.

Bounds, not a thread pool: at most `Limits.live_requests` requests exist here
at once, one TTS call and one ASR call run at a time, and each queue is small.
That caps decoded audio at roughly the WAV in ASR plus one waiting, which is
what keeps a 900-page book from turning into gigabytes of float32.

Cancellation is cooperative and only at stage boundaries. Killing a thread
inside Torch or ONNX Runtime is not safe, so a cancel that lands mid-inference
lets that call finish and then throws the result away.

The Windows rule that shapes all of this
----------------------------------------
On this machine, a thread blocked in a synchronous read on the stdin pipe
stops *other* threads from loading DLLs. Measured, reproducibly: a thread
importing numpy stalls inside the extension module's LoadLibrary and completes
the instant the reader's read() returns - twenty seconds later, if that is how
long the next line takes. Nothing about it looks like a deadlock from the
outside; it looks like a very slow model.

Torch, ONNX Runtime, libsndfile and ctranslate2 all load their libraries on
first *use*, not on import, so this is not hypothetical: the first version of
this file loaded models inside the stage threads and hung on the first real
request, every time.

Hence real_prepare(), and the rule it exists to keep: **every model load
happens on the reader thread, before the request is admitted.** The stages
only ever run code whose libraries are already in the process. If you add a
backend here, load it in prepare or it will hang.
"""
from __future__ import annotations

import base64
import io
import json
import math
import os
import queue
import re
import sys
import threading
import traceback
from dataclasses import dataclass, field

import g2p_cache
import kokoro_assets
import worker_protocol

G2P_CACHE = g2p_cache.LexicalPhonemeCache()

PROTOCOL_VERSION = 2


class FatalWorkerError(RuntimeError):
    """A model or process-level failure, not a failure of one block.

    The difference matters to the parent: a bad block is one error to one
    waiter, while a half-initialised worker would answer every future request
    the same way. A fatal error is reported once and then the process stops
    accepting work.
    """


@dataclass(frozen=True)
class Limits:
    """Everything that bounds memory here, in one place.

    `live_requests` is the real ceiling, and it is the *only* one that bounds
    decoded audio: a request holds a WAV from the moment TTS returns until its
    reply has been written, so at most this many WAVs can exist at once no
    matter how the queues are sized. The rest is admission control - anything
    beyond the ceiling is answered `busy` rather than buffered.

    `responses` is generous on purpose. A reply queue small enough to fill is
    a reply queue the stdin reader can block on, and the one message that must
    never wait behind a full queue is a cancel.
    """
    live_requests: int = 2
    tts_queue: int = 2
    asr_queue: int = 1
    responses: int = 256


@dataclass(order=True)
class _Queued:
    """A TTS queue entry. Priority first, then FIFO sequence - and the request
    itself never takes part in the comparison, or two entries tying on
    priority would try to order two objects that have no ordering."""
    priority: int
    sequence: int
    req: object = field(default=None, compare=False)


class _Request:
    __slots__ = ("req_id", "text", "voice", "rate", "whisper_model",
                 "priority", "cancelled", "audio", "native_heard",
                 "native_reason", "native_coverage", "sample_rate",
                 "alignment_method")

    def __init__(self, req_id, text, voice, rate, whisper_model, priority):
        self.req_id = req_id
        self.text = text
        self.voice = voice
        self.rate = rate
        self.whisper_model = whisper_model
        self.priority = priority
        self.cancelled = False
        self.audio = None
        self.native_heard = None
        self.native_reason = None
        self.native_coverage = 0.0
        self.sample_rate = 24000
        self.alignment_method = None


@dataclass(frozen=True)
class _TTSOutput:
    """Audio plus optional timings already computed by Kokoro itself."""
    audio: bytes
    native_heard: list | None = None
    native_reason: str | None = None
    native_coverage: float = 0.0
    sample_rate: int = 24000
    alignment_method: str | None = None


class _Response:
    """One terminal reply. The WAV stays raw until the writer serialises it -
    base64 is a third of the size again, and holding both copies for every
    queued response is exactly the memory this design is trying not to spend.

    `req` is the admitted request this answers, or None for a request that was
    refused before it ever entered the live table. The writer retires only the
    former: a refusal that carried a live id would otherwise evict the request
    that legitimately holds it, and that request would then be answered twice.

    `then` runs after the line is written and flushed. It exists for exactly
    one case - stopping the process after a fatal reply has actually reached
    the parent, rather than sleeping and hoping.
    """
    __slots__ = ("req_id", "audio", "heard", "error", "req", "then",
                 "alignment_method", "fallback_reason", "native_coverage",
                 "sample_rate")

    def __init__(self, req_id, audio=None, heard=None, error=None, req=None,
                 then=None, alignment_method=None, fallback_reason=None,
                 native_coverage=0.0, sample_rate=24000):
        self.req_id = req_id
        self.audio = audio
        self.heard = heard
        self.error = error
        self.req = req
        self.then = then
        self.alignment_method = alignment_method
        self.fallback_reason = fallback_reason
        self.native_coverage = native_coverage
        self.sample_rate = sample_rate

    def payload(self):
        if self.error is not None:
            return {"req_id": self.req_id, "ok": False, "error": self.error}
        payload = {"req_id": self.req_id, "ok": True,
                   "audio_b64": base64.b64encode(self.audio).decode("ascii"),
                   "heard": self.heard or [], "sample_rate": self.sample_rate}
        if self.alignment_method:
            payload["alignment_method"] = self.alignment_method
            payload["alignment_fallback_reason"] = self.fallback_reason
            payload["native_direct_coverage"] = self.native_coverage
        return payload

    def frame(self):
        """(kind, metadata, body) for the binary transport.

        The samples are the body, exactly as they came out of soundfile -
        which is the entire reason this protocol exists. Nothing here may put
        audio into the metadata: an audio_b64 field on a binary RESULT would
        send the clip twice and quietly restore the cost being removed.
        """
        req_id = self.req_id if isinstance(self.req_id, str) else ""
        if self.error is not None:
            return worker_protocol.ERROR, {"req_id": req_id,
                                           "error": self.error}, b""
        meta = {"req_id": req_id,
                "sample_rate": self.sample_rate,
                "mime_type": "audio/wav",
                "words": self.heard or [],
                "alignment_method": self.alignment_method or "whisper-v1",
                "diagnostics": {
                    "alignment_fallback_reason": self.fallback_reason,
                    "native_direct_coverage": self.native_coverage,
                }}
        return worker_protocol.RESULT, meta, self.audio or b""


def _error(code, stage, message, retryable=False, fatal=False):
    return {"code": code, "stage": stage, "message": message,
            "retryable": retryable, "fatal": fatal}


def _brief(exc):
    """What crosses the wire. The full traceback goes to the log; the message
    the parent may surface stays short and carries no document text."""
    text = f"{type(exc).__name__}: {exc}"
    return text[:200]


_SHUTDOWN = object()


_WORD = re.compile(r"\S+")


def _word_key(text):
    return re.sub(r"[^\w']+", "", text).lower()


def _kokoro_word_groups(tokens, offset=0.0):
    """Group Misaki tokens back into the whitespace words Kokoro received.

    Punctuation and hyphen pieces are separate MTokens, while Talkify's
    source-of-truth is whitespace-delimited. ``MToken.whitespace`` is the
    lossless boundary that joins those pieces without guessing.
    """
    groups = []
    text_parts = []
    starts = []
    ends = []
    partial = False

    def flush():
        nonlocal text_parts, starts, ends, partial
        text = "".join(text_parts).strip()
        if text:
            groups.append({"w": text,
                           "s": min(starts) + offset if starts else None,
                           "e": max(ends) + offset if ends else None,
                           "partial": partial})
        text_parts, starts, ends, partial = [], [], [], False

    for token in tokens or []:
        text_parts.append(str(getattr(token, "text", "") or ""))
        start = getattr(token, "start_ts", None)
        end = getattr(token, "end_ts", None)
        if (start is None) != (end is None):
            partial = True
        elif start is not None:
            try:
                starts.append(float(start))
                ends.append(float(end))
            except (TypeError, ValueError, OverflowError):
                partial = True
        if getattr(token, "whitespace", ""):
            flush()
    flush()
    return groups


def _validated_native_words(text, groups, audio_seconds,
                            minimum_direct_coverage=0.95):
    """Return protocol marks or a reason to run the Whisper fallback."""
    expected = _WORD.findall(text)
    if not expected:
        return None, "native timing has no source words", 0.0
    if len(groups) != len(expected):
        return None, "native timing word-count mismatch", 0.0

    lexical = 0
    direct = 0
    heard = []
    last_end = 0.0
    for source_index, (source, group) in enumerate(zip(expected, groups)):
        source_key = _word_key(source)
        if source_key and source_key != _word_key(group.get("w", "")):
            return None, "native timing text mismatch", 0.0
        if source_key:
            lexical += 1
        start, end = group.get("s"), group.get("e")
        if group.get("partial"):
            return None, "native timing has a partial mark", 0.0
        if start is None and end is None:
            continue
        if not isinstance(start, (int, float)) or not isinstance(end, (int, float)):
            return None, "native timing has a non-numeric mark", 0.0
        start, end = float(start), float(end)
        if (not math.isfinite(start) or not math.isfinite(end)
                or start < 0.0 or end <= start):
            return None, "native timing has an invalid interval", 0.0
        if start + 1e-6 < last_end:
            return None, "native timing is not monotonic", 0.0
        if end > audio_seconds + 0.075:
            return None, "native timing exceeds its audio", 0.0
        last_end = end
        if source_key:
            direct += 1
        # Use the exact text sent to Kokoro. The reconstructed token text is
        # evidence for validation, not a new source of truth.
        heard.append({"w": source, "s": start, "e": end,
                      "i": source_index})

    coverage = direct / lexical if lexical else 1.0
    if coverage < minimum_direct_coverage:
        return None, "native timing coverage is too low", coverage
    if not heard:
        return None, "native timing has no voiced marks", coverage
    return heard, None, coverage


# --------------------------------------------------------------- real models

# --------------------------------------------------------------- backends

class BackendSynthesis:
    """What any speech backend hands back: audio, and whatever it knows about
    where the words fell inside it.

    ``native_marks`` is the protocol mark list when the backend's own duration
    model produced usable spans, and None when it did not - in which case
    ``native_reason`` says why and the ASR stage falls back to Whisper. A
    backend that has no duration model at all is not a broken backend; it just
    always answers None.
    """

    __slots__ = ("wav", "sample_rate", "native_marks", "native_reason",
                 "native_coverage", "alignment_method")

    def __init__(self, wav, sample_rate=24000, native_marks=None,
                 native_reason=None, native_coverage=0.0,
                 alignment_method=None):
        self.wav = wav
        self.sample_rate = sample_rate
        self.native_marks = native_marks
        self.native_reason = native_reason
        self.native_coverage = native_coverage
        self.alignment_method = alignment_method


def _native_timing_enabled():
    return os.environ.get("TALKIFY_KOKORO_NATIVE_TIMING", "1").lower() not in {
        "0", "false", "off", "no"}


class PyTorchKokoroBackend:
    """Kokoro through the PyTorch pipeline. The path that has always shipped.

    Its duration predictor already knows where each word sits, so a normal
    block never reaches Whisper at all - see _validated_native_words for what
    "usable" means. Skipping Whisper saves about 1.2s a block.
    """

    name = "pytorch"

    def __init__(self):
        self._pipes = {}
        self._threads_set = False

    def identity(self):
        """What distinguishes audio made here from audio made elsewhere.

        Goes into the parent's cache key, so a cached block made by one
        backend is never played back as if the other had made it."""
        return "pytorch"

    def native_timing(self):
        return _native_timing_enabled()

    def _pipeline(self, lang_code="a"):
        if lang_code not in self._pipes:
            try:
                from kokoro import KPipeline
                self._pipes[lang_code] = KPipeline(lang_code=lang_code)
            except Exception as e:
                raise FatalWorkerError(f"kokoro failed to load: {_brief(e)}") from e
        return self._pipes[lang_code]

    def prepare(self, req):
        """Reader-thread only - see the module docstring. Everything a
        synthesis will touch has to have its DLLs in the process before the
        request is admitted."""
        import numpy                                # noqa: F401 - for the DLL
        import soundfile                            # noqa: F401 - libsndfile
        import torch
        if not self._threads_set:
            # Kokoro is compute-bound and scales with threads, and Torch
            # defaults to half the machine. Measured here on 8 logical cores,
            # median seconds for one 280-character block: 1 thread 13.5,
            # 2 threads 11.3, 4 (the default) 8.4, 8 threads 7.4. One core is
            # left for everything else - the browser, the OCR of a book still
            # being scanned, the rest of the desktop - because the last core
            # buys very little on a curve that flat.
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
            self._threads_set = True
        self._pipeline().load_voice(req.voice)

    def synthesize(self, *, text, voice, speed, cancel=None):
        import numpy as np
        import soundfile as sf

        chunks = []
        groups = []
        sample_count = 0
        for result in self._pipeline()(text, voice=voice, speed=speed):
            audio = result.audio if hasattr(result, "audio") else result[2]
            if audio is None:
                continue
            groups.extend(_kokoro_word_groups(
                getattr(result, "tokens", None), sample_count / 24000.0))
            chunks.append(audio)
            sample_count += len(audio)
        audio = np.concatenate(chunks) if chunks else np.zeros(1, dtype="float32")
        buf = io.BytesIO()
        sf.write(buf, audio, 24000, format="WAV", subtype="PCM_16")
        native, reason, coverage = _validated_native_words(
            text, groups, len(audio) / 24000.0)
        if not _native_timing_enabled():
            native, reason, coverage = None, "native timing disabled", 0.0
        return BackendSynthesis(
            buf.getvalue(), 24000, native, reason, coverage,
            alignment_method="kokoro-pytorch-duration-v1" if native else None)


class OnnxKokoroBackend:
    """Kokoro through ONNX Runtime, off unless TALKIFY_KOKORO_BACKEND=onnx.

    Everything about it is allowed to fail into the PyTorch path: a missing or
    unverified asset, a provider that will not start, a session whose outputs
    do not include duration, a voice the model does not carry. None of those
    may stop the reader hearing the paragraph, so each one is caught here and
    reported as a reason rather than raised at the request.

    The session is built by hand rather than left to kokoro-onnx's helper,
    because the thread settings are the whole point: ONNX Runtime's default
    of zero threads spawns one per physical core and spins them, which on this
    machine starves the OCR pass and the reply reader that share it.
    """

    name = "onnx"

    def __init__(self):
        self._kokoro = None
        self._session = None
        self._identity = "onnx"
        self._duration_output = False
        self._duration_output_name = None
        self._audio_output_name = "audio"
        self._voices = {}
        self._voices_path = None
        self._reason = None
        self._g2p_lock = threading.Lock()

    def identity(self):
        return self._identity

    def native_timing(self):
        return bool(self._duration_output) and _native_timing_enabled()

    def _load_voices(self, path):
        try:
            import zipfile
            import numpy as np
            with zipfile.ZipFile(path, "r") as zf:
                for filename in zf.namelist():
                    if filename.endswith(".npy"):
                        voice_name = filename[:-4]
                        with zf.open(filename) as f:
                            self._voices[voice_name] = np.load(f).astype("float32")
            if self._voices:
                return
        except Exception:
            pass
        try:
            import numpy as np
            import torch
            loaded = torch.load(path, map_location="cpu", weights_only=True)
            for voice_name, style_tensor in loaded.items():
                if isinstance(style_tensor, torch.Tensor):
                    self._voices[voice_name] = style_tensor.numpy().astype("float32")
                else:
                    self._voices[voice_name] = np.asarray(style_tensor, dtype="float32")
        except Exception:
            pass

    def _warmup_session(self):
        if self._session is None:
            return
        try:
            import numpy as np
            dummy_tokens = np.zeros((1, 16), dtype=np.int64)
            dummy_style = np.zeros((1, 256), dtype=np.float32)
            dummy_speed = np.array([1.0], dtype=np.float32)
            feed = {"tokens": dummy_tokens, "style": dummy_style, "speed": dummy_speed}
            valid_feed = {k: v for k, v in feed.items() if any(i.name == k for i in self._session.get_inputs())}
            self._session.run(None, valid_feed)
        except Exception:
            pass

    def prepare(self, req):
        """Load and introspect on the reader thread, once.

        A failure here is *not* fatal to the worker: it sets a reason and
        leaves self._kokoro None, and serve() falls back to PyTorch. Raising
        would take the whole worker down over an optional accelerator."""
        if self._kokoro is not None or self._session is not None or self._reason is not None:
            return
        try:
            import onnxruntime
            model_path, voices_path = kokoro_assets.verified_paths()
            options = onnxruntime.SessionOptions()
            options.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
            options.graph_optimization_level = (
                onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL)
            options.inter_op_num_threads = 1
            options.intra_op_num_threads = _onnx_intra_op_threads()
            options.enable_cpu_mem_arena = True
            options.enable_mem_pattern = True
            session = onnxruntime.InferenceSession(
                model_path, sess_options=options,
                providers=["CPUExecutionProvider"])
            # Capability probe, not a guess: only a real output actually named
            # for duration may be read as one.
            outputs = [o.name for o in session.get_outputs()]
            self._duration_output_name = next(
                (name for name in outputs if any(k in name.lower() for k in ("dur", "pred_dur", "duration"))),
                None
            )
            self._duration_output = self._duration_output_name is not None
            self._audio_output_name = next(
                (name for name in outputs if any(k in name.lower() for k in ("audio", "waveform"))),
                "audio"
            )
            try:
                from kokoro_onnx import Kokoro
                self._kokoro = Kokoro.from_session(session, voices_path)
            except Exception:
                self._kokoro = None
            self._session = session
            self._voices_path = voices_path
            self._identity = "onnx:" + kokoro_assets.identity()
            self._load_voices(voices_path)
            self._warmup_session()
        except Exception as e:
            self._reason = f"onnx backend unavailable: {_brief(e)}"
            self._kokoro = None
            self._session = None

    @property
    def available(self):
        return self._kokoro is not None or self._session is not None

    @property
    def reason(self):
        return self._reason

    def synthesize(self, *, text, voice, speed, cancel=None):
        import numpy as np
        import soundfile as sf

        if self._kokoro is None and self._session is None:
            raise FatalWorkerError(self._reason or "onnx backend unavailable")

        audio = None
        sample_rate = 24000
        pred_dur = None
        token_word_indices = None
        source_words = _WORD.findall(text)

        # Fast G2P tokenization and native duration extraction if duration output available
        if self._duration_output and self._session is not None:
            with self._g2p_lock:
                try:
                    tokens_list, token_word_indices = G2P_CACHE.process_clause(
                        text, tuple(range(len(source_words)))
                    )
                    if tokens_list:
                        style = self._voices.get(voice)
                        if style is None and self._voices:
                            style = next(iter(self._voices.values()))
                        elif style is None and self._kokoro is not None and hasattr(self._kokoro, "get_voice_style"):
                            try:
                                style = self._kokoro.get_voice_style(voice)
                            except Exception:
                                style = None
                        if style is None:
                            style = np.zeros((1, 256), dtype=np.float32)
                        elif style.ndim == 1:
                            style = style[np.newaxis, :]

                        tokens_arr = np.array([tokens_list], dtype=np.int64)
                        feed = {
                            "tokens": tokens_arr,
                            "style": np.ascontiguousarray(style, dtype=np.float32),
                            "speed": np.array([speed], dtype=np.float32)
                        }
                        input_names = {inp.name for inp in self._session.get_inputs()}
                        feed = {k: v for k, v in feed.items() if k in input_names}

                        outputs_to_fetch = [self._audio_output_name]
                        if self._duration_output_name:
                            outputs_to_fetch.append(self._duration_output_name)

                        results = self._session.run(outputs_to_fetch, feed)
                        audio = np.asarray(results[0], dtype=np.float32).squeeze()
                        if len(results) > 1 and self._duration_output_name:
                            pred_dur = np.asarray(results[1]).squeeze()
                except Exception:
                    audio = None
                    pred_dur = None

        # Fallback to kokoro-onnx wrapper if direct run was not performed or failed
        if audio is None:
            if self._kokoro is None:
                raise FatalWorkerError(self._reason or "onnx backend audio synthesis failed")
            with self._g2p_lock:
                audio, sample_rate = self._kokoro.create(
                    text, voice=voice, speed=speed, lang="en-us")
            audio = np.asarray(audio, dtype="float32")

        audio = np.clip(audio, -1.0, 1.0)
        buf = io.BytesIO()
        sf.write(buf, audio, int(sample_rate), format="WAV", subtype="PCM_16")
        audio_sec = len(audio) / float(sample_rate)

        # If pred_dur was extracted, map to words and validate
        if pred_dur is not None and token_word_indices is not None and source_words:
            groups = self._map_pred_dur_to_groups(
                pred_dur, token_word_indices, source_words, audio_sec, speed
            )
            if groups is not None:
                native, reason, coverage = _validated_native_words(
                    text, groups, audio_sec
                )
                if not _native_timing_enabled():
                    native, reason, coverage = None, "native timing disabled", 0.0
                if native is not None:
                    return BackendSynthesis(
                        buf.getvalue(), int(sample_rate), native, reason, coverage,
                        alignment_method="kokoro-onnx-duration-v1"
                    )

        # No valid duration output means no native marks -> fallback to Whisper
        return BackendSynthesis(
            buf.getvalue(), int(sample_rate), None,
            "onnx build exposes no duration output"
            if not self._duration_output else "onnx native timing not wired",
            0.0
        )

    def _map_pred_dur_to_groups(self, pred_dur, token_word_indices, source_words, audio_duration_sec, speed=1.0):
        import numpy as np
        pred_dur = np.asarray(pred_dur)
        if pred_dur.ndim != 1 or pred_dur.size == 0:
            return None
        if not np.isfinite(pred_dur).all() or (pred_dur < 0).any():
            return None
        num_tokens = len(pred_dur)
        if num_tokens != len(token_word_indices):
            return None

        frame_sec = (512.0 / 24000.0) / max(0.2, speed)
        token_durs = np.maximum(0, pred_dur.astype(np.float64)) * frame_sec
        token_ends = np.cumsum(token_durs)
        token_starts = np.empty_like(token_ends)
        token_starts[0] = 0.0
        token_starts[1:] = token_ends[:-1]

        num_words = len(source_words)
        word_starts = [None] * num_words
        word_ends = [None] * num_words

        for t_idx, (t_start, t_end) in enumerate(zip(token_starts, token_ends)):
            w_idx = token_word_indices[t_idx]
            if 0 <= w_idx < num_words:
                if word_starts[w_idx] is None or t_start < word_starts[w_idx]:
                    word_starts[w_idx] = float(t_start)
                if word_ends[w_idx] is None or t_end > word_ends[w_idx]:
                    word_ends[w_idx] = float(t_end)

        groups = []
        last_end = 0.0
        for i, word in enumerate(source_words):
            s = word_starts[i]
            e = word_ends[i]
            if s is None or e is None:
                # A zero-duration or unmapped word is not evidence. Do not
                # manufacture a tiny interval here: that would make the
                # caller believe native timing is complete and incorrectly
                # skip Whisper. The normal fallback is the safe answer.
                return None
            s = max(float(s), last_end)
            e = float(e)
            if not math.isfinite(s) or not math.isfinite(e) or e <= s:
                return None
            if e > audio_duration_sec + 0.075:
                return None
            groups.append({"w": word, "s": round(s, 4), "e": round(e, 4), "partial": False})
            last_end = e
        return groups


def _onnx_intra_op_threads():
    """Threads for the ONNX session, benchmarked rather than assumed.

    TALKIFY_KOKORO_ONNX_THREADS overrides it. The default mirrors what the
    Torch measurement found - leave one core for the browser, the
    OCR pass and the reply reader - because the failure mode of taking them
    all is not a slower model, it is a stuttering UI while the model is fast.
    """
    count = os.cpu_count() or 2
    raw = os.environ.get("TALKIFY_KOKORO_ONNX_THREADS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return min(value, max(1, count))
        except ValueError:
            pass
    # os.cpu_count() reports logical processors on Windows. Prefer the
    # physical-core count when psutil is present so one ONNX worker does not
    # occupy both SMT siblings of every core; retain a stdlib-only fallback
    # because psutil is optional in the Kokoro venv.
    physical = None
    try:
        import psutil
        physical = psutil.cpu_count(logical=False)
    except Exception:
        pass
    effective_physical = physical or max(1, count // 2 if count >= 4 else count)
    return max(1, effective_physical - 1)


def build_backend(name=None):
    """The backend this worker will use, chosen by environment only.

    Selection is independent of the transport: a protocol problem is fixed by
    going back to JSONL, a backend problem by going back to PyTorch, and
    neither rollback drags the other with it."""
    name = (name or os.environ.get("TALKIFY_KOKORO_BACKEND") or "pytorch").lower()
    if name == "onnx":
        return OnnxKokoroBackend()
    return PyTorchKokoroBackend()


_asrs = {}
BACKEND = build_backend()


def _recogniser(model):
    if model not in _asrs:
        try:
            from faster_whisper import WhisperModel
            _asrs[model] = WhisperModel(model, device="cpu", compute_type="int8")
        except Exception as e:
            raise FatalWorkerError(f"whisper failed to load: {_brief(e)}") from e
    return _asrs[model]


def real_tts(req):
    """Whichever backend is configured speaks, through one interface.

    The ONNX backend is allowed to be absent, unverified or half-working: it
    reports that on prepare and this falls through to PyTorch. A reader whose
    accelerator did not load hears the paragraph anyway, a little slower.
    """
    backend = BACKEND
    if getattr(backend, "available", True) is False:
        backend = _pytorch_fallback()
    product = backend.synthesize(text=req.text, voice=req.voice, speed=req.rate)
    return _TTSOutput(product.wav, product.native_marks, product.native_reason,
                      product.native_coverage, product.sample_rate,
                      getattr(product, "alignment_method", None))


_fallback_backend = None


def _pytorch_fallback():
    global _fallback_backend
    if _fallback_backend is None:
        _fallback_backend = PyTorchKokoroBackend()
    return _fallback_backend


def real_asr(req, wav):
    """faster-whisper listens back. Returns [{"w","s","e"}]."""
    segments, _ = _recogniser(req.whisper_model).transcribe(
        io.BytesIO(wav), word_timestamps=True, language="en", beam_size=1,
        best_of=1, temperature=0.0, condition_on_previous_text=False)
    return [{"w": w.word.strip(), "s": w.start, "e": w.end}
            for seg in segments for w in (seg.words or [])]


_warmed = False


def real_prepare(req):
    """Load everything this request will need, on the thread that reads stdin.

    Not an optimisation. On Windows, a thread sitting in a blocking read on
    the stdin pipe stops *other* threads from loading DLLs - measured on this
    machine, reproducibly: a thread importing numpy stalls at the extension
    module's LoadLibrary and only completes the instant the reader's read()
    returns. Torch, ONNX Runtime, libsndfile and ctranslate2 all load their
    libraries on first use rather than on import, so a stage thread reaching
    for a model while the reader waits would hang until the next request
    arrived - and for a parent blocked on this one, that is never. It looks
    exactly like a slow model.

    So every model load happens here, on the reader thread, at a moment when
    nothing is blocked on the pipe. The stages then only ever run code whose
    libraries are already in the process.

    Everything here is cached and idempotent, so this costs a dict lookup on
    every request after the first. A voice or a Whisper model the process has
    not seen before pays its load once, in the same safe place.
    """
    global _warmed
    try:
        BACKEND.prepare(req)
        if getattr(BACKEND, "available", True) is False:
            # An optional accelerator that would not start is not an error the
            # reader should ever meet. Load the backend that always works and
            # carry on; the reason is already noted in the log.
            _pytorch_fallback().prepare(req)
        _recogniser(req.whisper_model)
        if not _warmed:
            # One real pass, end to end. Importing the packages is not enough:
            # the first actual inference is what pulls in the compute kernels,
            # and those are DLL loads too.
            warm = _Request("warm", "Ready.", req.voice, 1.0, req.whisper_model, 0)
            warm_output = real_tts(warm)
            real_asr(warm, warm_output.audio)
            _warmed = True
    except FatalWorkerError:
        raise
    except Exception as e:
        raise FatalWorkerError(f"models would not load: {_brief(e)}") from e


# ------------------------------------------------------------- transports

# What a transport owes serve(): a stream of request dicts, a way to send one
# terminal reply, and an announce() for whatever handshake it has. Nothing in
# the state machine below knows which one it is talking through, which is what
# keeps "roll back to JSONL" a one-word change rather than a revert.

BAD_REQUEST = "__malformed__"


class JsonlTransport:
    """One JSON object per line each way. The transport that has always
    shipped, kept as the rollback path and as the shape every existing test
    drives serve() through.

    A line that will not parse is *not* fatal here: a third-party library
    printing a warning onto stdout used to be exactly that, and killing every
    live request over it would turn a stray print into a failed paragraph.
    """

    protocol = "jsonl-v1"

    def __init__(self, inp, out):
        self.inp = inp
        self.out = out

    def announce(self, meta):
        return None                     # no handshake in this protocol

    def requests(self):
        for line in self.inp:
            line = line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("request is not an object")
            except Exception as e:
                yield {"op": BAD_REQUEST, "message": _brief(e)}
                continue
            yield req

    def send(self, resp):
        self.out.write(json.dumps(resp.payload()) + "\n")
        self.out.flush()


class BinaryTransport:
    """TKF1 frames: raw WAV bodies, no base64, no third of the bytes wasted.

    Unlike JSONL there is no benign garbage. Every byte on this stream was
    written by a FrameWriter under a lock, so anything that does not parse
    means the stream is no longer aligned - and a WAV body can contain the
    magic, so there is nothing safe to resynchronise onto. A ProtocolError
    from here ends the loop and the process; the parent sees EOF, fails what
    was outstanding once each, and starts a fresh worker.
    """

    protocol = "binary-v1"

    def __init__(self, inp, out):
        self.inp = inp
        self.writer = worker_protocol.FrameWriter(out)
        self.acknowledged = False

    def announce(self, meta):
        self.writer.write(worker_protocol.HELLO, meta)

    def requests(self):
        while True:
            try:
                frame = worker_protocol.read_frame(self.inp)
            except worker_protocol.StreamClosed:
                return
            kind = frame.kind
            if kind == worker_protocol.HELLO_ACK:
                self.acknowledged = True
                continue
            if kind == worker_protocol.SHUTDOWN:
                return
            if kind == worker_protocol.CANCEL:
                yield {"op": "cancel", "req_id": frame.req_id}
                continue
            if kind == worker_protocol.SYNTHESIZE:
                if not self.acknowledged:
                    # The parent must validate the hello before it asks for
                    # anything. Serving work to a parent that has not agreed
                    # what this worker is would be answering a question nobody
                    # confirmed they asked.
                    raise worker_protocol.ProtocolError(
                        "synthesize arrived before the hello was acknowledged")
                yield {"op": "synth", **frame.meta}
                continue
            raise worker_protocol.ProtocolError(
                f"unexpected {worker_protocol.KIND_NAMES.get(kind, kind)} frame")

    def send(self, resp):
        kind, meta, body = resp.frame()
        self.writer.write(kind, meta, body)


# ------------------------------------------------------------- the pipeline

def serve(inp=None, out=None, *, transport=None, tts=real_tts, asr=real_asr,
          prepare=None, limits=Limits(), log=None, stats=None, on_fatal=None,
          hello=None):
    """Run the request/response loop until stdin closes or a shutdown arrives.

    `tts`/`asr` are injectable so the state machine - admission, ordering,
    cancellation at each boundary, per-stage failure - can be tested without
    Kokoro, Whisper or a subprocess. Everything the models touch lives behind
    those two callables and nothing else.

    `prepare` runs on this thread, before a request is admitted, and is where
    every model load has to happen - see real_prepare for why that is a
    correctness requirement on Windows and not a preference. Left out, nothing
    is loaded and the stages are assumed to need nothing, which is what a test
    with injected stages wants.

    `on_fatal` is called once, from the writer, after a `fatal: true` reply has
    actually been written and flushed. main() passes the thing that stops the
    process; tests pass something that records it. Default None means "stay up
    but refuse everything", which is the only sensible behaviour when nobody
    has said how to die.

    The one structural rule in here: **a stage thread never returns early.**
    It exits only on the shutdown sentinel from its own input queue. A stage
    that returned on a fatal error would leave the stage upstream of it
    blocked forever on a queue nobody drains, and the requests behind it with
    no reply at all - which is worse than the fatal error it was reacting to.
    """
    # (inp, out) is the JSONL shape this has always had, and every existing
    # caller and test still passes it. A transport= argument replaces both and
    # is how the binary protocol arrives - the state machine below never finds
    # out which one it is running on.
    if transport is None:
        transport = JsonlTransport(inp, out)
    stats = stats if stats is not None else {}
    # Each of these counters has exactly one writing thread - tts_calls the TTS
    # stage, asr_calls the ASR stage, live_max/busy/dropped the reader - so
    # they need no lock. Keep it that way if you add one.
    stats.setdefault("live_max", 0)
    stats.setdefault("busy", 0)
    stats.setdefault("dropped", 0)
    stats.setdefault("tts_calls", 0)
    stats.setdefault("asr_calls", 0)

    tts_q = queue.PriorityQueue(maxsize=limits.tts_queue)
    asr_q = queue.Queue(maxsize=limits.asr_queue)
    resp_q = queue.Queue(maxsize=limits.responses)

    live = {}                       # req_id -> _Request, terminal reply pending
    live_lock = threading.Lock()
    sequence = [0]
    stopping = threading.Event()    # a fatal error: stop admitting new work

    def note(message):
        if log is not None:
            try:
                log.write(message + "\n")
                log.flush()
            except Exception:
                pass

    def emit(resp):
        """Hand a terminal reply to the writer. Blocking, but only a stage
        thread ever calls it, and a stage cannot produce a reply for a request
        that is not live - so at most `live_requests` of these can be
        outstanding, far below the queue's size."""
        resp_q.put(resp)

    def refuse(resp):
        """A reply from the reader, for a request that was never admitted.
        Never blocks: the reader must stay free to process a cancel, and a
        parent that has stopped reading stdout is not going to be helped by
        one more queued refusal."""
        try:
            resp_q.put_nowait(resp)
        except queue.Full:
            stats["dropped"] += 1
            note(f"dropped a refusal for {resp.req_id!r}: reply queue is full")

    def cancelled_reply(req, stage, message="request superseded"):
        return _Response(req.req_id, req=req,
                         error=_error("cancelled", stage, message))

    def fatal_reply(req, code, stage, exc):
        note(traceback.format_exc())
        stopping.set()
        return _Response(req.req_id, req=req,
                         error=_error(code, stage, _brief(exc), fatal=True),
                         then=on_fatal)

    # ------------------------------------------------------------ TTS stage
    def tts_stage():
        while True:
            item = tts_q.get()
            if item.req is _SHUTDOWN:
                # The sentinel sorts ahead of real work on purpose - shutdown
                # should not wait behind two blocks of inference - so anything
                # still queued has to be answered here or it would get no
                # reply at all. The reader has already stopped admitting by
                # the time this sentinel exists, so this drain is complete.
                while True:
                    try:
                        left = tts_q.get_nowait()
                    except queue.Empty:
                        break
                    if left.req is not _SHUTDOWN:
                        emit(cancelled_reply(left.req, "queued",
                                             "worker is shutting down"))
                asr_q.put(_SHUTDOWN)
                return
            req = item.req
            if stopping.is_set():
                emit(_Response(req.req_id, req=req, error=_error(
                    "worker_stopping", "queued", "worker is shutting down")))
                continue
            if req.cancelled:
                emit(cancelled_reply(req, "queued"))
                continue
            try:
                stats["tts_calls"] += 1
                product = tts(req)
                if isinstance(product, _TTSOutput):
                    wav = product.audio
                    req.native_heard = product.native_heard
                    req.native_reason = product.native_reason
                    req.native_coverage = product.native_coverage
                    req.alignment_method = product.alignment_method
                else:
                    # Injectable/legacy stages still return raw bytes and use
                    # the same Whisper path they always did.
                    wav = product
            except FatalWorkerError as e:
                emit(fatal_reply(req, "tts_failed", "tts", e))
                continue
            except Exception as e:
                note(traceback.format_exc())
                emit(_Response(req.req_id, req=req, error=_error(
                    "tts_failed", "tts", _brief(e))))
                continue
            if req.cancelled:
                # The samples exist but nobody wants them, and Whisper is the
                # expensive half. Drop them here rather than aligning audio
                # that will never be played.
                emit(cancelled_reply(req, "tts"))
                continue
            req.audio = wav
            while True:
                try:
                    asr_q.put(req, timeout=0.1)
                    break
                except queue.Full:
                    if req.cancelled:
                        req.audio = None
                        emit(cancelled_reply(req, "tts"))
                        break

    # ------------------------------------------------------------ ASR stage
    def asr_stage():
        while True:
            req = asr_q.get()
            if req is _SHUTDOWN:
                emit(_SHUTDOWN)
                return
            wav, req.audio = req.audio, None
            if req.cancelled:
                emit(cancelled_reply(req, "asr"))
                continue
            if req.native_heard is not None:
                heard = req.native_heard
                method = req.alignment_method or "kokoro-pytorch-duration-v1"
            else:
                try:
                    stats["asr_calls"] += 1
                    heard = asr(req, wav)
                    method = "whisper-v1"
                except FatalWorkerError as e:
                    emit(fatal_reply(req, "alignment_failed", "asr", e))
                    continue
                except Exception as e:
                    # Audio without timings is never published, so there is
                    # nothing partial to hand back: the pair fails together.
                    note(traceback.format_exc())
                    emit(_Response(req.req_id, req=req, error=_error(
                        "alignment_failed", "asr", _brief(e))))
                    continue
            if req.cancelled:
                emit(cancelled_reply(req, "asr"))
                continue
            emit(_Response(req.req_id, audio=wav, heard=heard, req=req,
                           alignment_method=method,
                           fallback_reason=req.native_reason,
                           native_coverage=req.native_coverage))

    # --------------------------------------------------------------- writer
    def writer():
        broken = False
        while True:
            resp = resp_q.get()
            if resp is _SHUTDOWN:
                return
            if resp.req is not None and resp.req.cancelled and resp.error is None:
                # A cancel that landed after ASR returned but before the reply
                # was written. Publishing here would hand back audio the
                # parent has already moved past. The last word on this is
                # still the parent's - a cancel sent after the line is on the
                # wire cannot be caught by anyone - which is why the
                # coordinator re-checks generation before it caches anything.
                resp = cancelled_reply(resp.req, "asr")
            if not broken:
                try:
                    transport.send(resp)
                except Exception:
                    # The parent is gone. Keep draining so no stage thread
                    # blocks on a full queue on the way out.
                    broken = True
            if resp.req is not None:
                # Retire only what was actually admitted. A refusal carrying a
                # live id would otherwise evict the request that owns it, and
                # that request would then be answered a second time.
                with live_lock:
                    live.pop(resp.req.req_id, None)
            if resp.then is not None:
                try:
                    resp.then()
                except Exception:
                    note(traceback.format_exc())

    threads = [threading.Thread(target=tts_stage, name="worker-tts", daemon=True),
               threading.Thread(target=asr_stage, name="worker-asr", daemon=True),
               threading.Thread(target=writer, name="worker-writer", daemon=True)]
    for t in threads:
        t.start()

    # --------------------------------------------------------------- reader
    def reject(req_id, code, message, stage="protocol"):
        refuse(_Response(req_id, error=_error(code, stage, message)))

    if hello is not None:
        # Said before a single request is admitted, and before any model has
        # been asked for: the parent has to know what it is talking to in
        # order to decide whether this worker's audio may share a cache with
        # the last one's.
        try:
            transport.announce(hello)
        except Exception as e:
            note(f"could not announce: {_brief(e)}")

    try:
        for req in transport.requests():
            op = req.get("op", "synth")
            if op == BAD_REQUEST:
                # No usable id: answer with a null one so a parent watching
                # the stream sees the protocol error, and keep going. Only the
                # JSONL transport produces this - on a framed stream a bad
                # frame is fatal and never reaches here.
                reject(None, "bad_request", req.get("message") or "bad request")
                continue
            if op == "shutdown":
                break
            req_id = req.get("req_id")

            if op == "cancel":
                # Advisory: set the flag and let whichever stage next looks at
                # this request emit the single terminal reply. Answering here
                # as well would double-answer the same id.
                if isinstance(req_id, str):
                    with live_lock:
                        held = live.get(req_id)
                    if held is not None:
                        held.cancelled = True
                continue

            if op != "synth":
                reject(req_id if isinstance(req_id, str) else None,
                       "bad_request", f"unknown op {op!r}")
                continue
            if not isinstance(req_id, str) or not req_id:
                reject(None, "bad_request", "synth needs a string req_id")
                continue
            # This thread is the only one that inserts into `live`, so looking
            # before admitting is safe: an entry cannot appear between the two
            # checks, and one disappearing only makes the duplicate check more
            # permissive than it needed to be.
            with live_lock:
                already_live = req_id in live
            if already_live:
                # Never attach a second waiter to a live id, and never answer
                # under it either - the reply would be ambiguous about which
                # request it settles. The parent generates unique ids, so this
                # is a protocol defence, not a normal path.
                reject(None, "duplicate_req_id", f"{req_id} is already live")
                continue
            if not isinstance(req.get("text"), str):
                reject(req_id, "bad_request", "synth needs text")
                continue
            if stopping.is_set():
                reject(req_id, "worker_stopping", "worker is shutting down",
                       stage="queued")
                continue

            priority = req.get("priority", 0)
            if not isinstance(priority, int) or isinstance(priority, bool):
                priority = 0
            record = _Request(req_id, req["text"], req.get("voice", "af_heart"),
                              req.get("rate", 1.0),
                              req.get("whisper_model", "base.en"), priority)
            with live_lock:
                full = len(live) >= limits.live_requests
                if not full:
                    live[req_id] = record
                    stats["live_max"] = max(stats["live_max"], len(live))
            if full:
                stats["busy"] += 1
                reject(req_id, "busy", "worker is at capacity", stage="queued")
                continue

            if prepare is not None:
                try:
                    prepare(record)
                except Exception as e:
                    # A model that will not load is not this block's problem -
                    # every block would fail the same way - so it is reported
                    # fatal and the process stops.
                    note(traceback.format_exc())
                    stopping.set()
                    refuse(_Response(req_id, req=record, error=_error(
                        "model_failed", "prepare", _brief(e), fatal=True),
                        then=on_fatal))
                    continue

            try:
                sequence[0] += 1
                tts_q.put_nowait(_Queued(record.priority, sequence[0], record))
            except queue.Full:
                # Admission must never block the reader: a cancel arriving
                # behind a blocked put is the one message that could have
                # freed the queue. This one *was* admitted, so its reply
                # carries the record and the writer retires it.
                stats["busy"] += 1
                refuse(_Response(req_id, req=record, error=_error(
                    "busy", "queued", "worker queue is full")))
    except worker_protocol.ProtocolError as e:
        # The stream is no longer aligned, and there is nothing safe to
        # resynchronise onto - a WAV body can contain the frame magic. Stop
        # reading, let the stages drain, and exit: the parent notices EOF,
        # completes every outstanding request exactly once, and starts a
        # fresh worker for the next one.
        note(f"protocol failure: {_brief(e)}")
        stopping.set()
    finally:
        tts_q.put(_Queued(-1, -1, _SHUTDOWN))
        for t in threads:
            t.join(30)


def worker_hello():
    """What this process is, for the parent's handshake and cache identity.

    Sent before any model loads, so it names the *configured* backend rather
    than a proven one - proving it means building an ONNX session, and a
    parent blocked for thirty seconds on a handshake would be a worse bug
    than the one this prevents. What actually happened to a given block
    travels with that block, in the result's alignment_method.
    """
    backend = BACKEND
    return worker_protocol.hello(
        protocol="binary-v1", backend=getattr(backend, "name", "pytorch"),
        native_timing=bool(getattr(backend, "native_timing", lambda: False)()),
        model_identity=str(getattr(backend, "identity", lambda: "unknown")()))


def _protocol_choice(argv):
    """binary-v1 only when it is asked for, by name, on the command line.

    Never inferred from a first byte. A guess here would mean a worker and a
    parent that disagree about the framing and produce a stream that looks
    corrupt rather than misconfigured, which is a far harder thing to read in
    a log.
    """
    for i, arg in enumerate(argv):
        if arg == "--protocol" and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--protocol="):
            return arg.split("=", 1)[1]
    return "jsonl-v1"


def main(argv=None):
    # The protocol owns stdout - but torch/onnxruntime/huggingface_hub can
    # print their own stray progress and warnings straight to the stdout file
    # descriptor (not just sys.stdout, so reassigning that alone isn't
    # enough). Duplicate the real stdout fd for replies, then point fd 1
    # itself at stderr so anything else that writes there lands out of the
    # protocol's way. This happens before any third-party import, because an
    # import is exactly when those libraries print.
    #
    # This is deliberately inside main() and not at import time: smoke.py
    # imports this module to drive serve() with fake stages, and rewiring the
    # test runner's stdout as a side effect of an import would be its own bug.
    argv = sys.argv if argv is None else argv
    protocol = _protocol_choice(argv)
    if protocol == "binary-v1":
        # Binary both ways, and never through a text wrapper: a TextIOWrapper
        # on Windows would translate a lone \n inside a WAV body into \r\n and
        # corrupt the audio in a way that still plays.
        reply = os.fdopen(os.dup(1), "wb")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        transport = BinaryTransport(sys.stdin.buffer, reply)
        hello = worker_hello()
    else:
        reply = os.fdopen(os.dup(1), "w", encoding="utf-8", newline="\n")
        os.dup2(2, 1)
        sys.stdout = sys.stderr
        transport = JsonlTransport(sys.stdin, reply)
        hello = None
    # A fatal model failure closes the process rather than serving requests
    # from a half-initialised worker. The parent notices EOF, fails every
    # outstanding request with one error, and starts a fresh worker for the
    # next explicit request - which is a far better story than every block
    # from here on failing the same way.
    serve(transport=transport, prepare=real_prepare, log=sys.stderr,
          on_fatal=lambda: os._exit(3), hello=hello)


if __name__ == "__main__":
    main()
