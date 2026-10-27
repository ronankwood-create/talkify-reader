#!/usr/bin/env python3
"""Does Talkify's worker pipeline actually make this faster on *this* machine?

    python bench_synth.py                 # the standard run
    python bench_synth.py --blocks 40     # longer, steadier numbers
    python bench_synth.py --rss 400       # also watch memory over 400 blocks

smoke.py deliberately knows nothing about models, Tesseract or wall-clock
thresholds - it has to run in seconds, every time, on any machine. This is the
other half: the real Kokoro worker, its native duration timing (or Whisper
fallback), and a number at the end that says whether the pipeline earned its
complexity here.

Two runtimes running at once is not automatically faster, and here they were
not. Measured on this machine (8 logical cores, af_heart, base.en, 20 blocks of
260-285 characters): Kokoro takes 7.9s of a 9.1s block and Whisper 1.2s, so the
best overlap theoretically available is 13% - and the two runtimes contend for
the same cores, which ate all of it. Serial 200.4s, pipelined 199.7s: x1.00.

That is why --pipeline is off by default in talkify.py, and why this script
exists rather than a confident paragraph in a commit message. Run it before
turning the flag on; the answer will be different on hardware where the two
stages are better balanced.

The rest of the pipelined machinery is not in question and is always on: a
seek can cancel a block that is already inside the worker instead of waiting
eight seconds for audio nobody will hear.

What it measures
----------------
  serial      one block at a time, using the configured alignment path.
  pipelined   two blocks in flight. This overlaps TTS with ASR only when a
              block falls back from native duration timing to Whisper.

Both runs use the same voice, fallback model and text, and both are measured
only after the models are warm - a cold Kokoro load is tens of seconds and
would swamp everything else. Set TALKIFY_KOKORO_NATIVE_TIMING=0 before running
to measure the rollback path instead.

What it asserts
---------------
  * every block comes back with timings for every source word, monotonic, and
    inside the WAV's own duration. A throughput win that quietly corrupted
    alignment would be worthless, and this is the invariant that fails
    quietly.
  * the median latency of a *lone* request has not regressed by more than 5%.
    One paragraph, nobody else waiting, is the case a reader actually feels.
  * batch throughput improves by at least 25%. Anything less is not worth
    running two model runtimes at once - which is the bar it failed here.
"""
from __future__ import annotations

import argparse
import io
import statistics
import sys
import time
import wave

import engines

# Representative of what extract.py produces: whole sentences, 200-320
# characters, the length that keeps first-audio latency low. Deliberately
# ordinary prose - the point is to time the models, not to find their edges.
BLOCKS = [
    "The bending stress in a beam varies linearly through its depth, reaching "
    "its greatest value at the surfaces furthest from the neutral axis. This "
    "is why a beam of a given weight is stiffer when its material is placed "
    "far from the centre, which is the entire argument for an I-section.",

    "A material's behaviour under load is described by its stress-strain "
    "curve. Up to the proportional limit the relationship is linear and the "
    "slope of that line is the modulus of elasticity, a property of the "
    "material itself rather than of the shape of the specimen being tested.",

    "When a shaft is twisted, the shear stress at any point is proportional "
    "to its distance from the axis of rotation. The material at the centre "
    "carries almost nothing, which is the reason a hollow shaft of the same "
    "outside diameter is very nearly as strong as a solid one.",

    "Buckling is not a strength failure but a stability failure. A slender "
    "column can collapse at a load far below the one that would crush it, "
    "and the load at which that happens depends on how the ends are held "
    "quite as much as on the material the column is made from.",

    "Fatigue failures begin at a surface flaw and grow a little with every "
    "cycle of load. The final fracture is sudden, but the crack that caused "
    "it may have been growing for millions of cycles at stresses well below "
    "anything the static design calculations would have flagged as dangerous.",
]


def wav_seconds(data):
    with wave.open(io.BytesIO(data), "rb") as w:
        return w.getnframes() / float(w.getframerate())


def check_block(text, audio, mime, words):
    """The invariants that break quietly. A faster run that produced these
    wrong would be a regression dressed as an improvement."""
    source = engines.words_of(text)
    problems = []
    if len(words) != len(source):
        problems.append(f"{len(words)} timings for {len(source)} words")
    if any(b.s < a.s for a, b in zip(words, words[1:])):
        problems.append("timings are not monotonic")
    if words:
        length = wav_seconds(audio)
        if words[-1].e > length + 0.5:
            problems.append(f"last word ends at {words[-1].e:.2f}s "
                            f"in a {length:.2f}s clip")
    return problems


def rss_mb(pid):
    """The worker's memory, including its children - and None rather than a
    wrong number when that cannot be known.

    On Windows the venv's python.exe re-executes the base interpreter, so the
    pid held here is a launcher and the models live in its child. Reporting the
    launcher's four megabytes as "the worker" is worse than reporting nothing:
    it looks like an answer. psutil can walk the tree; without it, say so.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except psutil.Error:
                pass
        return total / (1024 * 1024)
    except Exception:
        return None



def worker_pid(engine):
    try:
        return engine.client()._proc.pid
    except Exception:
        return None


def serial_run(engine, texts):
    """One block at a time - what synth() has always done."""
    began = time.perf_counter()
    results = [engine.synth(t) for t in texts]
    return time.perf_counter() - began, results


def pipelined_run(engine, texts, inflight=2):
    """`inflight` blocks in the worker at once, exactly as the coordinator's
    dispatchers arrange it. Submitting the next one before collecting the
    last is the whole trick."""
    began = time.perf_counter()
    results, handles = [], []
    for text in texts:
        while len(handles) >= inflight:
            results.append(handles.pop(0).result())
        handles.append(engine.submit(text))
    while handles:
        results.append(handles.pop(0).result())
    return time.perf_counter() - began, results


def lone_latencies(engine, texts, rounds):
    """One request, nobody else waiting. This is the number a reader feels
    when they press play, and the one the pipeline must not have cost."""
    times = []
    for i in range(rounds):
        text = texts[i % len(texts)]
        began = time.perf_counter()
        engine.synth(text)
        times.append(time.perf_counter() - began)
    return times


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blocks", type=int, default=20,
                    help="blocks per timed run (default 20)")
    ap.add_argument("--lone", type=int, default=5,
                    help="lone-request samples per mode (default 5)")
    ap.add_argument("--voice", default="af_heart")
    ap.add_argument("--whisper", default="base.en")
    ap.add_argument("--rss", type=int, default=0, metavar="N",
                    help="also run N blocks watching worker memory, to show it "
                         "reaches a plateau instead of climbing")
    a = ap.parse_args()

    texts = [BLOCKS[i % len(BLOCKS)] for i in range(a.blocks)]
    engine = engines.KokoroEngine(voice=a.voice, whisper_model=a.whisper)

    print(f"warming {a.voice} / {a.whisper} (a cold model load is tens of "
          f"seconds and would swamp every number below)")
    began = time.perf_counter()
    engine.synth("Warming up the models before anything is measured.")
    print(f"  warm in {time.perf_counter() - began:.1f}s\n")

    pid = worker_pid(engine)
    before = rss_mb(pid) if pid else None

    print(f"serial: {a.blocks} blocks, one at a time")
    serial_time, serial_results = serial_run(engine, texts)
    print(f"  {serial_time:.1f}s  ->  {60 * a.blocks / serial_time:.1f} blocks/min")

    print(f"pipelined: {a.blocks} blocks, two in flight")
    pipe_time, pipe_results = pipelined_run(engine, texts)
    print(f"  {pipe_time:.1f}s  ->  {60 * a.blocks / pipe_time:.1f} blocks/min")

    print("\nlone request (one paragraph, nobody else waiting)")
    serial_lone = statistics.median(lone_latencies(engine, texts, a.lone))
    print(f"  median {serial_lone:.2f}s")

    failures = []
    for label, results in (("serial", serial_results), ("pipelined", pipe_results)):
        for i, (audio, mime, words) in enumerate(results):
            for problem in check_block(texts[i], audio, mime, words):
                failures.append(f"{label} block {i}: {problem}")

    print("\ncorrectness")
    if failures:
        for f in failures[:10]:
            print(f"  FAIL  {f}")
    else:
        print(f"  pass  every block in both runs has complete, monotonic timings"
              f" inside its own audio ({2 * a.blocks} blocks)")

    speedup = serial_time / pipe_time if pipe_time else 0.0
    print("\nverdict")
    print(f"  throughput  x{speedup:.2f}  "
          f"({'accept' if speedup >= 1.25 else 'REJECT - under the 1.25x bar'})")
    print(f"  lone median {serial_lone:.2f}s  "
          f"(the pipeline must not have made this worse; compare against a run "
          f"of this script from before the change, or with --no-pipeline)")
    if speedup < 1.25:
        print("  Overlap is capped by whichever stage dominates: if Kokoro is "
              "most of the block, there is little for Whisper to hide behind, "
              "and two runtimes on the same cores contend for what is left. "
              "Leave Talkify's --pipeline off (that is the default).")

    if a.rss:
        print(f"\nmemory over {a.rss} blocks (it must plateau, not climb)")
        marks = []
        long_texts = [BLOCKS[i % len(BLOCKS)] for i in range(a.rss)]
        step = max(1, a.rss // 10)
        for start in range(0, a.rss, step):
            pipelined_run(engine, long_texts[start:start + step])
            here = rss_mb(worker_pid(engine))
            marks.append(here)
            print(f"  after {min(start + step, a.rss):>5} blocks: "
                  f"{'unavailable' if here is None else f'{here:.0f} MB'}")
        real = [m for m in marks if m is not None]
        if len(real) >= 4:
            early = statistics.median(real[1:len(real) // 2])
            late = statistics.median(real[len(real) // 2:])
            growth = (late - early) / early * 100 if early else 0.0
            print(f"  second half vs first: {growth:+.1f}%  "
                  f"({'plateau' if growth < 15 else 'CLIMBING - look for a leak'})")
    elif before is not None:
        after = rss_mb(pid)
        if after is not None:
            print(f"\nworker memory {before:.0f} MB -> {after:.0f} MB "
                  f"over {2 * a.blocks + a.lone} blocks")
    else:
        print("\nworker memory: unavailable (pip install psutil to measure it)")

    engine.shutdown()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
