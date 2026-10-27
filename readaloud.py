#!/usr/bin/env python3
"""
readaloud.py — turn a textbook PDF chapter into text that actually sounds right
read aloud.

The voice is the easy part now. This handles the part that makes textbook
audio unlistenable: reading order in two-column pages, running heads and page
numbers cutting in mid-sentence, words hyphenated across line breaks, figure
captions interrupting a paragraph, and equations and units read out as
punctuation instead of English.

  python3 readaloud.py book.pdf -o ch4.txt --pages 184-212
  python3 readaloud.py book.pdf --math skip --chunks chunks/
"""
import argparse, base64, hashlib, json, os, re, statistics, sys, threading, time
from concurrent.futures import ProcessPoolExecutor
from collections import Counter
from dataclasses import dataclass

# Before any third-party import, and deliberately so. OpenMP and MKL read these
# once, when their runtime is loaded by the first library that pulls it in - so
# a cap set after `import pymupdf` is a cap set too late, and Tesseract and any
# BLAS underneath it go back to claiming every core on the machine. Ingestion
# runs beside playback here; a page of OCR that saturates the CPU is a page
# that makes the paragraph being spoken arrive late.
os.environ['OMP_THREAD_LIMIT'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import pymupdf


# ============================================================== extraction

CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def page_lines(page):
    """Every text line with its geometry. Line-level, not block-level: PyMuPDF
    merges side-by-side columns into single blocks, which is exactly how
    two-column reading order gets scrambled."""
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for ln in block["lines"]:
            text = "".join(s["text"] for s in ln["spans"]).strip()
            # Some PDFs' page-number/running-head glyphs decode to a stray
            # control byte glued onto real text - a real example: a page
            # number came back from PyMuPDF as "56\x08" (a literal backspace
            # after the digits). It's invisible on screen, so it looks like
            # a clean match at a glance, but it breaks every exact-string
            # comparison downstream: PAGENUM's $ anchor no longer matches,
            # and two occurrences of the "same" running head only compare
            # equal if the same stray byte survived OCR/extraction
            # identically both times. Strip it here, once, rather than
            # teach every regex and every norm() call to tolerate it.
            text = CONTROL_CHARS.sub("", text).strip()
            if not text:
                continue
            x0, y0, x1, y1 = ln["bbox"]
            size = max((s["size"] for s in ln["spans"]), default=10)
            font = ln["spans"][0]["font"] if ln["spans"] else ""
            lines.append({"x0": x0, "y0": y0, "x1": x1, "y1": y1,
                          "text": text, "size": size, "font": font})
    return lines


OCR_DPI = 220

# The Windows Tesseract installer routinely does not add itself to PATH, so
# look where it actually lands before giving up.
TESSERACT_GUESSES = [
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Programs\Tesseract-OCR\tesseract.exe"),
    os.path.join(os.environ.get("LOCALAPPDATA", ""), r"Tesseract-OCR\tesseract.exe"),
    "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract",
]

INSTALL_HINT = ("Tesseract OCR is needed to read scanned pages.\n"
                "  Windows:  winget install UB-Mannheim.TesseractOCR\n"
                "  macOS:    brew install tesseract\n"
                "  Linux:    sudo apt install tesseract-ocr")


_OCR_STACK = None
_OCR_FAILURE = None                  # (exception, monotonic time it was raised)
# How long a "no Tesseract here" answer is reused before looking again. Long
# enough that a 980-page scan does not re-walk PATH once per page, short enough
# that installing Tesseract while Talkify is open starts working by itself -
# memoizing the failure for the life of the process meant it never would.
_OCR_RETRY_AFTER = 30.0


def setup_ocr():
    """Import the OCR stack and point it at the Tesseract binary.

    Answered once per process on success. Every OCR'd page and every
    running-head probe calls this, and the lookup underneath is a walk of the
    whole PATH with every PATHEXT suffix tried at each entry - dozens of stat
    calls to rediscover a binary that has not moved since the last page.

    A failure is remembered too, but only briefly: see _OCR_RETRY_AFTER.
    """
    global _OCR_STACK, _OCR_FAILURE
    os.environ['OMP_THREAD_LIMIT'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    if _OCR_STACK is not None:
        return _OCR_STACK
    if _OCR_FAILURE is not None:
        error, when = _OCR_FAILURE
        if time.monotonic() - when < _OCR_RETRY_AFTER:
            raise error
        _OCR_FAILURE = None

    import shutil
    try:
        try:
            import pytesseract
            from PIL import Image
        except ImportError:
            raise RuntimeError("Scanned PDF needs OCR support:\n"
                               "  python -m pip install pytesseract pillow\n\n" + INSTALL_HINT)

        exe = shutil.which("tesseract")
        if not exe:
            exe = next((p for p in TESSERACT_GUESSES if p and os.path.exists(p)), None)
        if not exe:
            raise RuntimeError(INSTALL_HINT + "\n\n(Looked on PATH and in the usual "
                               "install folders and found nothing.)")
    except Exception as e:
        _OCR_FAILURE = (e, time.monotonic())
        raise
    pytesseract.pytesseract.tesseract_cmd = exe
    _OCR_STACK = (pytesseract, Image)
    return _OCR_STACK


def pixmap_image(Image, pix):
    """A PIL image over a PyMuPDF pixmap's own bytes.

    The obvious route - `Image.open(io.BytesIO(pix.tobytes("png")))` - deflates
    a multi-megapixel page into PNG and immediately inflates it again, purely
    to hand the same pixels to Pillow. Measured on a letter page at 150 DPI:
    55ms to encode plus up to 20ms to decode, against 6ms to wrap the samples
    directly, and every OCR'd page and every running-head probe paid it.

    The pixels are identical either way (PNG is lossless), so nothing
    downstream - including the OCR cache on disk - can tell the difference.
    Anything with an unexpected sample layout falls back to the long way
    rather than guessing at it.
    """
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(pix.n)
    if mode is None or pix.stride != pix.width * pix.n:
        import io
        return Image.open(io.BytesIO(pix.tobytes("png")))
    return Image.frombytes(mode, (pix.width, pix.height), pix.samples)


def _native_dpi(page):
    """The scan's actual pixel resolution, so OCR never renders above it -
    that's pure upsampling for zero accuracy gain (a 200 DPI book gained nothing from
    rendering at 300). Measured on the Callister scan: OCR_DPI's default of
    220 was already oversampling a 200 DPI source by ~20% for no benefit,
    which is ~20% of OCR time spent on nothing. None if the page doesn't
    have one single clear embedded image to measure against."""
    imgs = page.get_images()
    if len(imgs) != 1:
        return None
    # get_images() already carries the image's pixel dimensions (entries 2 and
    # 3 of each tuple), straight out of the PDF's image dictionary.
    # extract_image() returns the same two numbers but decodes the entire
    # image to do it - a full-page 200 DPI scan, decompressed once per page,
    # to read a width. Only fall back to it if the dictionary is missing them.
    try:
        w, h = imgs[0][2], imgs[0][3]
        if not w or not h:
            info = page.parent.extract_image(imgs[0][0])
            w, h = info.get("width"), info.get("height")
    except Exception:
        return None
    if not w or not h or page.rect.width <= 0 or page.rect.height <= 0:
        return None
    return min(w / (page.rect.width / 72), h / (page.rect.height / 72))


NUMBERISH = re.compile(r"^\d+(\.\d+)*[a-z]?$", re.I)


def ocr_lines(page, dpi=OCR_DPI):
    """Same line-with-geometry structure as page_lines, but recovered from a
    scanned page by Tesseract. Keeping the geometry is the point: it lets the
    column, running-head and paragraph logic work identically on scans."""
    pytesseract, Image = setup_ocr()

    native = _native_dpi(page)
    if native:
        dpi = max(72, min(dpi, round(native)))

    print(f"  OCR page {page.number + 1}...", end="\r", flush=True, file=sys.stderr)
    pix = page.get_pixmap(dpi=dpi)
    img = pixmap_image(Image, pix)
    scale = 72.0 / dpi          # image pixels back to PDF points

    data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

    flat = []
    for i, word in enumerate(data["text"]):
        word = word.strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        x, y = data["left"][i], data["top"][i]
        w, h = data["width"][i], data["height"][i]
        flat.append({"x0": x, "y0": y, "x1": x + w, "y1": y + h, "text": word,
                     "conf": int(data["conf"][i]), "key": key})

    # Confidence-flanking runs on the flat emission order, not per Tesseract
    # line-group: a word wrapping to the start of a new line has no
    # predecessor within its own group to flank it, even though the last
    # word of the previous line is right there in the natural reading
    # sequence - a real example below ("ships") starts a new line-group
    # and would never get flanked-checked at all if this only looked
    # within one group.
    kept = []
    for idx, w in enumerate(flat):
        if w["conf"] >= 30:
            kept.append(w)
            continue
        # A figure/table/equation number sitting right after its caption
        # keyword is exactly the shape Tesseract tends to underrate - a
        # real example: "Figure" at 95% confidence, then "1.3" at 18%,
        # then "The Liberty ship..." back at 94%+. The flat 30% floor
        # drops "1.3" outright and the caption comes out numberless.
        # Trust a short, purely numeric token (never body prose) sitting
        # directly between two words Tesseract WAS confident about -
        # isolated low-confidence noise elsewhere still gets dropped
        # exactly as before.
        flanked = (0 < idx < len(flat) - 1
                   and flat[idx - 1]["conf"] >= 30
                   and flat[idx + 1]["conf"] >= 30)
        if not flanked:
            continue
        if NUMBERISH.match(w["text"]):
            kept.append(w)
            continue
        # A footnote-reference mark superscripted right onto its word
        # ("ships" + a raised "4") drags the whole token's confidence
        # down with it, and the flat 30% floor drops the real word along
        # with the garbage: a real example, "ships<junk>" at 19%
        # confidence between two 90%+ neighbors, vanished entirely from
        # "The failure of many of the World War II Liberty ships is a
        # well-known..." Recover the real word and drop only the
        # corrupted tail - a superscript mark was never going to be
        # legible after downsampling anyway.
        m = re.match(r"^([A-Za-z]{2,})[^\w\s]{0,2}$", w["text"])
        if m:
            w = dict(w)
            w["text"] = m.group(1)
            kept.append(w)

    grouped = {}
    for w in kept:
        grouped.setdefault(w["key"], []).append(w)
    for words in grouped.values():
        words.sort(key=lambda w: w["x0"])

    lines = []
    for words in grouped.values():
        for sub in _split_on_column_gap(words):
            x0 = min(w["x0"] for w in sub)
            y0 = min(w["y0"] for w in sub)
            x1 = max(w["x1"] for w in sub)
            y1 = max(w["y1"] for w in sub)
            text = " ".join(w["text"] for w in sub)
            if not text.strip():
                continue
            lines.append({"x0": x0 * scale, "y0": y0 * scale,
                          "x1": x1 * scale, "y1": y1 * scale,
                          "text": text, "size": (y1 - y0) * scale,
                          "font": "ocr"})
    return lines


def _split_on_column_gap(words):
    """Tesseract groups words into a "line" purely by vertical position -
    it has no idea a page might be two columns, so a row where a left-
    column item and a right-column item happen to sit at the same height
    gets read straight across and called one line: "1. List six different
    property classifications of 4. (a) List the three primary
    classifications..." (a real example from a two-column numbered list).
    No amount of reordering lines afterward can undo that - the columns
    are already merged into one text string before ordered_lines() or
    group_paragraphs() ever sees it, which is why fixing reading order
    alone (see order_page_lines) never touched this case.

    Measured on that real line: normal word-to-word gaps ran ~3.2pt;
    the actual column gutter between "of" and "4." was 48pt - about 15x
    the local median, not a subtle difference. Splitting wherever a gap
    is that far outside the line's own normal spacing separates the
    columns back into their own line objects, the same shape column
    detection and reading order already expect."""
    if len(words) < 2:
        return [words]
    ordered = sorted(words, key=lambda w: w["x0"])
    gaps = [b["x0"] - a["x1"] for a, b in zip(ordered, ordered[1:])]
    cut = max(20, statistics.median(gaps) * 6)
    out, cur = [], [ordered[0]]
    for w, gap in zip(ordered[1:], gaps):
        if gap > cut:
            out.append(cur)
            cur = []
        cur.append(w)
    out.append(cur)
    return out


def needs_ocr(page, mode):
    if mode == "never":
        return False
    if mode == "always":
        return True
    # Same test, asked in the cheap order. get_images() reads the page's
    # resource dictionary; get_text() runs the whole text extractor. A native
    # PDF has no page image, so on the common case this now answers without
    # extracting anything - and every page was being asked twice (once to
    # decide what to OCR, once inside ordered_lines) on top of the extraction
    # ordered_lines does for real.
    return bool(page.get_images()) and len(page.get_text().strip()) < 100


_WORKER_DOCS = {}


def _ocr_one(args):
    """Runs in a worker process: OCR a single page, return plain dicts.

    The document stays open for the life of the worker. Each worker is handed
    page after page of the same book, and reopening a 291MB scan for every one
    of them was a real share of the wall clock before any pixels were read.
    """
    path, idx, dpi = args
    doc = _WORKER_DOCS.get(path)
    if doc is None:
        doc = _WORKER_DOCS[path] = pymupdf.open(path)
    return idx, ocr_lines(doc[idx], dpi)


WORKDIR = os.path.dirname(os.path.abspath(__file__))


# Bump this whenever a code change alters what OCR-derived line geometry
# looks like (page_lines/ocr_lines/_split_on_column_gap/order_page_lines/
# _split_trailing/extract_sidebar and similar). The cache below is keyed off
# this on top of the source file, specifically so a fix to how OCR output is
# shaped can't keep serving pre-fix geometry forever - a real example: this
# book's OCR cache was hours old, predating _split_on_column_gap ever
# splitting its two-column Learning Objectives list, and every downstream
# reading-order fix landed today kept silently doing nothing in the actual
# app because the cache short-circuited before any of that code ever ran.
OCR_CACHE_VERSION = 2


def cache_dir_for(path):
    """OCR cache lives beside the scripts, so nothing is scattered next to the
    source PDFs (which may sit on a synced drive)."""
    stamp = hashlib.sha1(
        f"{os.path.abspath(path)}:{os.path.getsize(path)}:{OCR_CACHE_VERSION}"
        .encode()).hexdigest()[:12]
    d = os.path.join(WORKDIR, ".ocr-cache", stamp)
    os.makedirs(d, exist_ok=True)
    return d


def _ocr_worker_init():
    """Stop this OCR process if the application that started it is gone.

    Pool workers are children, and on Windows a child does not die with a
    parent that is killed rather than closed. A clean stop shuts the pool down;
    a force-quit, a crash or a hard kill does not, and a background scan of a
    900-page book runs long enough for that to be a real window - it left two
    idle interpreters holding Tesseract behind, at a couple of hundred
    megabytes each.

    parent_process().join() waits on the parent's sentinel, which is portable
    and needs no pid guessing: os.kill(pid, 0) is a liveness probe on POSIX and
    an unconditional TerminateProcess on Windows, which would kill the very
    thing it is checking for.
    """
    os.environ['OMP_THREAD_LIMIT'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    if sys.platform == 'win32':
        try:
            import psutil
            psutil.Process().nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        except Exception:
            pass

    import multiprocessing
    parent = multiprocessing.parent_process()
    if parent is None:
        return

    def watch():
        parent.join()
        os._exit(0)

    threading.Thread(target=watch, daemon=True).start()


def ocr_many(path, indices, dpi=OCR_DPI, workers=None, progress=None):
    """OCR pages in parallel, caching each one to disk.

    Tesseract is single-threaded, so one page at a time leaves most of the CPU
    idle. Spreading pages across processes is close to a linear speed-up, and
    the cache means an interrupted run never repeats work.

    `progress`, when given, is called as `progress(done, total)` - counts,
    not a preformatted message, so a caller can drive either a CLI's `\r`
    line or a web progress bar off the same callback. `total` is the whole
    page count asked for (already-cached pages included), so a caller
    that opens the same book twice sees progress start from wherever the
    cache already got to, not from zero.
    """
    cache = cache_dir_for(path)
    out, todo = {}, []
    for i in indices:
        f = os.path.join(cache, f"{i:05d}.json")
        if os.path.exists(f):
            try:
                with open(f, encoding="utf-8") as fh:
                    out[i] = json.load(fh)
                continue
            except Exception:
                pass
        todo.append(i)

    total = len(indices)
    if progress:
        progress(len(out), total)
    if not todo:
        return out

    workers = workers or max(1, min(2, (os.cpu_count() or 2) - 2))
    done = len(out)
    try:
        with ProcessPoolExecutor(max_workers=workers,
                                 initializer=_ocr_worker_init) as ex:
            for idx, lines in ex.map(_ocr_one, [(path, i, dpi) for i in todo]):
                out[idx] = lines
                with open(os.path.join(cache, f"{idx:05d}.json"), "w",
                          encoding="utf-8") as fh:
                    json.dump(lines, fh)
                done += 1
                if progress:
                    progress(done, total)
    except Exception as e:
        print(f"  parallel OCR unavailable ({e}); falling back to one at a time",
              file=sys.stderr)
        doc = pymupdf.open(path)
        for i in todo:
            out[i] = ocr_lines(doc[i], dpi)
            done += 1
            if progress:
                progress(done, total)
        doc.close()
    return out


JUNK_TOKEN = re.compile(r"^[^A-Za-z0-9]+$")


def is_ocr_junk(line):
    """Figures, plots and axis labels OCR into confetti. Drop a line when it
    reads as debris rather than prose."""
    s = line.strip()
    # A chapter-opening drop cap is a single large letter on its own line -
    # real content, not debris, even though it fails every length check
    # below. Losing it silently dropped the first letter of a chapter's
    # opening word ("A" from "A photograph of...", "M" from "Materials...").
    if len(s) == 1 and s.isalpha() and s.isupper():
        return False
    if len(s) < 3:
        return True
    letters = sum(c.isalpha() for c in s)
    if letters / len(s) < 0.45:
        return True
    tokens = s.split()
    # Strip leading/trailing punctuation before judging a token as a "real
    # word" - `t.isalpha()` on the raw token is False for "materials.",
    # "categories:" or "(bottom)." (trailing period, colon, parens), which
    # silently failed the long-word check below and dropped whole sentence
    # fragments and bullet tails that are perfectly good prose.
    clean = [re.sub(r"^[^\w]+|[^\w]+$", "", t) for t in tokens]
    clean = [c for c in clean if c]
    if not clean:
        return True
    long_words = [c for c in clean if len(c) >= 4 and c.isalpha()]
    # A line with a genuine long word is a real sentence fragment even if
    # most of its *other* tokens are short ("as it applies to materials."
    # is 3 stubs out of 5 - 60% - but "applies" and "materials" make it
    # unmistakably prose, not axis-label confetti). Only apply the stub-
    # ratio filter when there is no such anchor word to trust.
    if not long_words:
        if len(clean) >= 3:
            stubs = sum(1 for c in clean if len(c) <= 2 or JUNK_TOKEN.match(c))
            if stubs / len(clean) > 0.5:
                return True
        if not any(len(c) >= 3 and c.isalpha() for c in clean):
            return True
    return False


def is_rotated_text(line):
    """A sideways photo credit running down the margin next to a figure -
    "iStockphoto/Jill Chen" printed vertically was the very first thing
    read aloud in one book, wedged into the middle of the opening sentence.
    Tesseract still reports its words as a "line", but that line's box is
    far taller than it is wide (measured: 6.9pt wide, 73.6pt tall for an
    actual ~8pt caption) because the text runs top-to-bottom, not
    left-to-right. Two problems follow: its "size" is meaningless noise
    computed from that box height, and worse, sorting purely by y0 drops
    it wherever its vertical span happens to overlap the real paragraph
    beside it - which is exactly the middle of a sentence, as often as
    not. Normal prose lines are always wider than they are tall, so this
    catches the rotated case without touching real body text."""
    w = line["x1"] - line["x0"]
    h = line["y1"] - line["y0"]
    return h > 3 * max(w, 1e-6) and h > 20


def _merge_edge_fragments(lines, page_height, tol=3.0):
    """A running head can come back from OCR as two separate line objects
    for one physical row - a real example: "6 + Chapter1 /" and
    "Introduction" landed as independent lines at the same y0, because
    Tesseract's line-grouping (or this file's own _split_on_column_gap,
    tuned to split genuine two-column gutters) treated the whitespace
    between the chapter number and the section title as a break. Neither
    fragment alone looks like a running head - VERSO_HEAD needs the whole
    row, and "Introduction" by itself is just an ordinary word - so both
    survive to be misread as junk or, worse, as body text. Recombining
    same-row fragments that sit in the page's edge band (where running
    heads and feet live - real body text glued into two pieces the same
    way would not both land in that narrow band) before any furniture
    pattern ever sees them fixes both at once."""
    if page_height <= 0 or len(lines) < 2:
        return lines

    def is_edge(l):
        return l["y0"] < EDGE * page_height or l["y1"] > (1 - EDGE) * page_height

    out, used = [], [False] * len(lines)
    for i, l in enumerate(lines):
        if used[i]:
            continue
        if not is_edge(l):
            out.append(l)
            continue
        group = [(i, l)]
        for j in range(i + 1, len(lines)):
            if not used[j] and is_edge(lines[j]) and abs(lines[j]["y0"] - l["y0"]) < tol:
                group.append((j, lines[j]))
        for j, _ in group:
            used[j] = True
        if len(group) == 1:
            out.append(l)
            continue
        pieces = sorted((g for _, g in group), key=lambda g: g["x0"])
        merged = dict(pieces[0])
        merged["text"] = " ".join(p["text"] for p in pieces)
        merged["x1"] = max(p["x1"] for p in pieces)
        merged["y1"] = max(p["y1"] for p in pieces)
        out.append(merged)
    return out


def drop_floating_labels(lines, page_height=None):
    """Axis labels and callouts scattered inside a figure ("Wave 1'",
    "Position -->") survive the junk filter because they are real words. What
    gives them away is isolation: body lines are packed tightly, and headings
    have text immediately below, but a label floats with whitespace on both
    sides.

    That isolation test has a blind spot at the top and bottom of the page:
    the first line's "gap above" and the last line's "gap below" are both
    defined as infinite (there's nothing there to measure against), so a
    running head sitting alone above the body text - exactly what a running
    head is - reads as "isolated" by the same test that catches a stray axis
    label, and never gets the chance to be recognized as furniture by
    strip_furniture()/find_furniture() downstream. Skipping the isolation
    check within the page's edge band (the same EDGE margin furniture
    detection already uses) leaves that to the furniture logic it actually
    belongs to, while still catching real floating labels - which show up
    in a figure body, not the margin."""
    if len(lines) < 4:
        return lines
    gaps = [b["y0"] - a["y1"] for a, b in zip(lines, lines[1:])
            if 0 <= b["y0"] - a["y1"] < 60]
    if not gaps:
        return lines
    loose = statistics.median(gaps) * 2.5 + 6

    keep = []
    for i, l in enumerate(lines):
        if page_height and (l["y0"] < EDGE * page_height
                             or l["y1"] > (1 - EDGE) * page_height):
            keep.append(l)
            continue
        above = l["y0"] - lines[i - 1]["y1"] if i > 0 else 1e6
        below = lines[i + 1]["y0"] - l["y1"] if i + 1 < len(lines) else 1e6
        isolated = above > loose and below > loose
        sentence_like = len(l["text"]) >= 30 or l["text"].rstrip().endswith((".", ":", "?"))
        if isolated and not sentence_like:
            continue
        keep.append(l)
    return keep


def _figure_image_rects(page, max_area_frac=0.6):
    """Bounding boxes of embedded raster images small enough to be a figure
    or photograph, not the single full-page background a scan renders as.
    Printed text photographed as part of a figure's scene (a label baked
    into the picture, a word printed on the object being photographed) can
    come back as extractable text - either from an OCR pass over the whole
    page, or from a hidden text layer already in the PDF - and reads as a
    perfectly valid English word or two, so is_ocr_junk has nothing to catch
    it on. The only thing that marks it as figure content rather than body
    text is that it sits inside the figure's own image, so that's what this
    checks instead of the text itself. A full-page scan's one image covers
    nearly the whole page, so excluding anything above max_area_frac keeps
    this from swallowing every line on a scanned page."""
    page_area = page.rect.width * page.rect.height
    if page_area <= 0:
        return []
    rects = []
    for img in page.get_images(full=True):
        try:
            for r in page.get_image_rects(img[0]):
                if 0 < r.width * r.height < page_area * max_area_frac:
                    rects.append(r)
        except Exception:
            continue
    return rects


def _inside_figure(line, rects, overlap=0.6):
    lx0, ly0, lx1, ly1 = line["x0"], line["y0"], line["x1"], line["y1"]
    larea = max(1e-6, (lx1 - lx0) * (ly1 - ly0))
    for r in rects:
        ix0, iy0 = max(lx0, r.x0), max(ly0, r.y0)
        ix1, iy1 = min(lx1, r.x1), min(ly1, r.y1)
        if ix1 > ix0 and iy1 > iy0 and (ix1 - ix0) * (iy1 - iy0) / larea >= overlap:
            return True
    return False


def extract_sidebar(lines, page_height, min_lines=3, width_ratio=0.4, min_gap=6):
    """Find a persistent narrow side column - a photo caption running down
    the margin next to a much wider body column - before order_page_lines
    ever sees the page.

    order_page_lines's two-column handling assumes a genuine two-column
    layout splits somewhere near the page's own midline. A real textbook
    page instead often has ONE wide body column that itself runs past the
    midline (justified to ~70% of the page width) plus a narrow sidebar
    column squeezed into the remaining margin - nowhere near a 50/50
    split. On a page like that, every ordinary body line "crosses the
    midline" by order_page_lines's own definition (starts left of it,
    ends right of it) and gets treated as a page-spanning divider, which
    flushes the band before the sidebar's lines ever get a matching
    "other side" to pair with - they fall back to raw top-to-bottom
    order and land interleaved mid-sentence into the body column: a real
    example turned "Whether they came as servants... or powerful
    planters..." into "Whether they came as servants... Unidentified or
    powerful planters... artist, The Old...", the sidebar's own caption
    text injected between the body's own words.

    Detecting this doesn't depend on the midline at all: group the
    page's non-edge lines into column blocks by their own horizontal
    extent (lines whose x-ranges overlap, or sit within min_gap of each
    other, are the same block - see the merge below), independent of
    where that block happens to sit on the page. On the pages this was
    built against, the body block's own span ran ~300-400pt wide; the
    sidebar's ran ~50-70pt - under a quarter of the body's - with a real
    gutter (12-18pt on those pages) between them, comfortably above
    min_gap. A block only counts as a sidebar if it clears both min_lines
    (guards against a single short line, like a paragraph's last line,
    being mistaken for a column of its own) and width_ratio (guards
    against a genuine, roughly-equal two-column layout, which this
    leaves untouched for order_page_lines to handle as it already does).

    Unlike a labeled figure caption, this text never gets a chance to
    match CAPTION_START ("Figure N", "Table N", ...) - these captions are
    plain credit lines ("Unidentified artist, The Old Plantation, c.
    1790-1800... Wikimedia") with no such marker, so keyword matching can
    never catch them. The geometry here - a run of narrow lines forming
    their own column, parallel to a much wider one - is a stronger signal
    that this isn't body prose than a keyword ever could be for this
    case, and the same silent-drop treatment already given to furniture
    and footnotes (also identified purely by position, not content)
    applies here: the caller drops these lines rather than reading them.

    Returns (lines with sidebar lines removed, sidebar lines sorted by
    y0)."""
    if page_height <= 0:
        return lines, []
    body = [l for l in lines if EDGE <= l["y0"] / page_height <= 1 - EDGE]
    edge = [l for l in lines if not (EDGE <= l["y0"] / page_height <= 1 - EDGE)]
    if len(body) < min_lines + 2:
        return lines, []

    blocks = []
    for l in sorted(body, key=lambda l: l["x0"]):
        if blocks and l["x0"] <= blocks[-1]["x1"] + min_gap:
            blocks[-1]["lines"].append(l)
            blocks[-1]["x1"] = max(blocks[-1]["x1"], l["x1"])
        else:
            blocks.append({"lines": [l], "x0": l["x0"], "x1": l["x1"]})
    if len(blocks) < 2:
        return lines, []

    primary = max(blocks, key=lambda b: b["x1"] - b["x0"])
    primary_span = primary["x1"] - primary["x0"]
    sidebar, kept = [], list(primary["lines"])
    for b in blocks:
        if b is primary:
            continue
        span = b["x1"] - b["x0"]
        if len(b["lines"]) >= min_lines and span <= primary_span * width_ratio:
            sidebar.extend(b["lines"])
        else:
            kept.extend(b["lines"])
    if not sidebar:
        return lines, []
    kept.sort(key=lambda l: l["y0"])
    sidebar.sort(key=lambda l: l["y0"])
    return edge + kept, sidebar


# See the trailing-element comment in order_page_lines. Sized the same way
# FOOTNOTE_GAP_MULT was: measured on the real example (12.0pt list pitch,
# 38.6pt gap before the trailing heading, ~3.2x) with margin on both sides.
TRAILING_GAP_MULT = 1.8
TRAILING_Y_SLACK = 0.5


def _split_trailing(left, right):
    """Peel a run of trailing single-column lines off the END of `left` -
    lines that sit below the right column's own range AND are separated
    from the left column's real text by a gap wider than its own normal
    line pitch. `left` and `right` must already be sorted by y0. Returns
    (left with the trailing run removed, the trailing run itself)."""
    if len(left) < 2 or not right:
        return left, []
    pitches = sorted(round(b["y0"] - a["y0"], 1) for a, b in zip(left, left[1:]))
    pitch = pitches[len(pitches) // 2] or 12.0
    max_right_y0 = right[-1]["y0"]

    split = len(left)
    for i in range(len(left) - 1, 0, -1):
        gap = left[i]["y0"] - left[i - 1]["y0"]
        if (left[i]["y0"] > max_right_y0 + TRAILING_Y_SLACK * pitch
                and gap > TRAILING_GAP_MULT * pitch):
            split = i
        else:
            break
    return left[:split], left[split:]


def order_page_lines(lines, width):
    """Reading order for one page: left-column-then-right-column within a
    two-column section, top-to-bottom everywhere else. A line that
    genuinely crosses the page's midline - starts left of it and ends
    right of it - is a horizontal divider between sections (a heading, an
    intro sentence, a spanning figure or table), not something that gets
    hoisted to the top of the page just because it doesn't belong to
    either column - see the band-flushing loop below.

    (An earlier version measured "divider" as ">62% of the widest line
    span on the page" instead of "crosses the midline". That silently
    failed on a real page: the sentence right before a two-column list -
    "After studying this chapter, you should be able to do the
    following:" - was 61.4% of the page's span, just under the cutoff, so
    it never triggered a divider. It and the entire two-column list below
    it then landed in one band together, and because that one intro line
    doesn't cleanly belong to either column, the band's whole coverage
    check failed and the real two-column content fell back to top-to-
    bottom order - exactly the interleaving this function exists to
    prevent. Crossing the midline is what a divider actually *is*,
    independent of how long other lines on the page happen to be.)

    Column detection runs per band, not once for the whole page. A single
    page-wide check ("are there at least 5 lines on each side?") is what a
    genuine page-long two-column layout looks like, but it gives up
    entirely - falling back to pure top-to-bottom order for the WHOLE page -
    on a short two-column section like a numbered list split 3 items left,
    2 right. When a left-column line and a right-column line land at the
    same height, top-to-bottom order reads them interleaved: "1. List six...
    4. (a) List the three..." (a real example). Checking each band's own,
    smaller population instead catches a short list on an otherwise
    single-column page, the same way a page-long layout already worked.

    A line only counts toward left or right if it sits cleanly on that
    side (with a little slack for justified text overrunning the gutter) -
    a line straddling the midline without fully crossing it means the band
    isn't really two clean columns, so the whole band falls back to
    top-to-bottom rather than guessing.

    One more thing can end up in `left` besides real left-column text: a
    single-column element (a section heading, a short intro line) that
    comes AFTER the two-column content but is too narrow to trigger the
    divider check above - a real example: "1.1 HISTORICAL PERSPECTIVE"
    (x0=57, x1=247.7) sits well clear of the midline (mid=288, tol=15)
    but never crosses it, so it never flushes the band on its own and
    lands in `left` along with the genuine list items above it. Emitting
    `left + right` then reads it right before the right column starts,
    even though it's vertically BELOW both columns - a 123pt jump back up
    the page after reading it. Second-opinion review (see git log) flagged
    two tempting fixes that both regress unbalanced columns: peeling off
    anything in `left` whose y0 exceeds the right column's max y0 also
    peels off a longer left column's own last few genuine lines whenever
    it simply runs past where the right column ends - normal when columns
    aren't the same length. What actually distinguishes a trailing
    element from a longer column is a genuine gap: real column lines keep
    the same pitch line after line, but a trailing heading is preceded by
    a break the column's own pitch never produces - measured on the real
    example, 12.0pt pitch throughout the list, then a 38.6pt gap before
    the heading (~3.2x pitch, comfortably clear of TRAILING_GAP_MULT).
    Requiring BOTH conditions - below the right column's own range AND
    that real gap - is what keeps a same-pitch unbalanced column intact
    while still catching this case.
    """
    if len(lines) < 2:
        return lines
    mid = width / 2
    tol = 15   # points of midline slack for justified text overrunning the gutter

    def order_band(band):
        left = [l for l in band if l["x1"] <= mid + tol]
        right = [l for l in band if l["x0"] >= mid - tol]
        if len(left) >= 2 and len(right) >= 2 and len(left) + len(right) == len(band):
            left.sort(key=lambda l: (round(l["y0"], 1), l["x0"]))
            right.sort(key=lambda l: (round(l["y0"], 1), l["x0"]))
            left, trailing = _split_trailing(left, right)
            return left + right + trailing
        return sorted(band, key=lambda l: (round(l["y0"], 1), l["x0"]))

    out, band = [], []
    for l in sorted(lines, key=lambda l: round(l["y0"], 1)):
        if l["x0"] < mid - tol and l["x1"] > mid + tol:
            out.extend(order_band(band))
            out.append(l)
            band = []
        else:
            band.append(l)
    out.extend(order_band(band))
    return out


def ordered_lines(page, ocr="auto", pre=None, tag=None):
    """`tag`, if given, is called as `tag(line, reason)` for every line
    dropped here ("junk" or "rotated") - used only by the triage view
    (diagnose_pages) to show why a line never reached the reader. Costs
    nothing when None, and changes no filtering decision either way."""
    lines = page_lines(page)
    scanned = False
    if pre is not None and page.number in pre:
        lines = [dict(l) for l in pre[page.number]]
        scanned = True
    elif needs_ocr(page, ocr):
        lines = ocr_lines(page)
        scanned = True
    if not lines:
        return []
    fig_rects = _figure_image_rects(page)
    if fig_rects:
        surviving = []
        for l in lines:
            if _inside_figure(l, fig_rects):
                if tag: tag(l, "in_figure")
            else:
                surviving.append(l)
        lines = surviving
    if scanned:
        lines = _merge_edge_fragments(lines, page.rect.height)
        kept = []
        for l in lines:
            if is_ocr_junk(l["text"]):
                if tag: tag(l, "junk")
            elif is_rotated_text(l):
                if tag: tag(l, "rotated")
            else:
                kept.append(l)
        kept.sort(key=lambda l: (round(l["y0"], 1), l["x0"]))
        lines = drop_floating_labels(kept, page.rect.height)
        if tag:
            survived = {id(l) for l in lines}
            for l in kept:
                if id(l) not in survived:
                    tag(l, "floating_label")
    lines, sidebar = extract_sidebar(lines, page.rect.height)
    if tag:
        for l in sidebar:
            tag(l, "caption")
    lines = order_page_lines(lines, page.rect.width)
    h = page.rect.height
    for l in lines:
        l["ypos"] = l["y0"] / h
        l["ocr"] = scanned
    return lines


# ------------------------------------------------------ page furniture

def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", s.lower())).strip()


EDGE = 0.09


def find_furniture(pages, threshold=0.3, min_pages=None):
    """Lines recurring near the top or bottom edge across pages are running
    heads and feet, not content.

    On a scan, OCR often glues the running head onto the first line of real
    text, so the whole line never repeats. Counting the leading fragment as
    well catches those.

    `pages` here is often padded with book-wide samples so a short excerpt
    still has enough pages to spot a recurring head - but a header text is
    only ever recurring *within the section it belongs to*, which is a
    handful of pages either way, changing header text every few pages.
    Sizing the recurrence floor off the padded sample count (instead of the
    excerpt actually being read) made that floor unreachable for a 6-page
    request: the header only appeared on 2 of those 6 pages, nowhere near a
    threshold sized for ~30. `min_pages` lets the floor track what was
    actually asked for while `pages` still supplies the wider search.
    """
    counts = Counter()
    for lines in pages:
        seen = set()
        for l in lines:
            if l["ypos"] < EDGE or l["ypos"] > 1 - EDGE:
                whole = norm(l["text"])
                # only treat a prefix as a candidate when it is a real
                # truncation of a long line, or short body lines collide
                keys = {whole} | ({whole[:45]} if len(whole) > 60 else set())
                for k in keys:
                    # Lower bound guards against coincidental short edge-band
                    # matches, but a real recto running head can be this
                    # short - "chapter 3" norm()s to "chapter #", 9 chars -
                    # and was landing under the old 12-char floor on every
                    # page of a real chapter, never once counted no matter
                    # how many pages it recurred on.
                    if 4 < len(k) < 90 and k not in seen:
                        seen.add(k)
                        counts[k] += 1
    floor = max(2, (min_pages if min_pages is not None else len(pages)) * threshold)
    return {k for k, c in counts.items() if c >= floor}


def strip_furniture(text, furniture):
    """Remove a matched running head from the front of a line, keeping any
    real text OCR glued to it."""
    n = norm(text)
    if n in furniture:
        return ""
    if len(n) > 60 and n[:45] in furniture:
        # walk back to a word boundary in the original string
        words = text.split()
        for cut in range(len(words), 0, -1):
            if norm(" ".join(words[:cut]))[:45] == n[:45]:
                return " ".join(words[cut:]).strip()
        return ""
    return text


PAGENUM = re.compile(r"^\s*[\[\(]?\s*(?:page\s*)?[\divxlcIVXLC]{1,7}\s*[\]\)]?\s*$", re.I)
# A page-corner running head naming the *current* section changes on every
# section boundary, so it may recur on only one page within a short excerpt -
# too rarely for find_furniture's recurrence count to ever catch it, no
# matter how the floor is tuned. Its shape is distinctive though: a section
# number, a title, and a trailing "+ page" or "- page" tag. Checked against
# 31 pages sampled across the whole 980-page book (all chapters, not just
# this one) and it matched only genuine running heads, never body text -
# a real heading like "1.2 MATERIALS SCIENCE AND ENGINEERING" has no
# trailing page tag, so it's untouched by this.
RUNNING_HEAD = re.compile(r"^\d+(\.\d+)?\s+\S.{0,70}[+\-–]\s*\d{1,4}$")
# The verso twin of RUNNING_HEAD: a page number, a bullet-like separator
# (Tesseract reads the actual bullet dot as almost anything - "+", "*",
# a degree sign, a circled-C - the same confusability BULLET_LEAD exists
# to handle for list markers), "Chapter N", then "/ Title". Recurrence-based
# furniture detection (find_furniture) should also catch this on a full
# chapter, but only every other page carries it (verso only), and a short
# excerpt or a narrow diagnostic page range can fall short of the
# recurrence floor before it ever gets the chance - this regex classifies
# it correctly on sight, the same reason RUNNING_HEAD exists for the recto
# form.
VERSO_HEAD = re.compile(
    r"^\d{1,4}\s*[•+°©*\-]\s*Chapter\s*\d{1,2}\s*/\s*\S.{0,60}$", re.I)
# The `re.I` this used to carry globally made [\dA-Z]+ - meant to require
# a real number or capitalized label right after the keyword ("Figure 3",
# "Plate IV") - match a run of ORDINARY LOWERCASE WORDS too, since case
# insensitivity applies to a character class the same as literal text.
# Real example: a body sentence spanning a page break on a real PDF started
# "plate to a 'great and firme post'..." (continuing "...affixed a metal
# plate..." from the previous page) - "plate" hit the keyword, and re.I let
# "to" satisfy [\dA-Z]+, so the whole paragraph was dropped as a caption.
# Scoping (?i:...) to just the keyword keeps "Plate", "PLATE", and "plate"
# all recognized while leaving [\dA-Z]+ genuinely case-sensitive, so only
# an actual digit or capitalized label counts as the caption's number.
CAPTION_START = re.compile(
    r"^\s*(?i:fig(?:ure)?\.?|table|tbl\.?|plate|chart|exhibit|scheme|photo|"
    r"diagram|listing|algorithm)\s*[\dA-Z]+[\.\-–:]?[\d]*\s*[\.\-–:—]?\s*\S")

# Some trade-nonfiction books run an endnote-style essay after each section
# instead of page-bottom footnotes: a bare heading, then a paragraph or two
# of citations in normal body type, sized and margined exactly like the
# narrative around it - no font-size or gap signal tells them apart, only
# the heading text does. Confirmed on a real book (a JSTOR chapter): a
# standalone line reading exactly "sources" (nothing else on the line)
# opens 2-3 paragraphs of citations, immediately after a real paragraph-end
# gap (20.98pt on the measured page, comfortably clear of any within-body
# wrap gap). Matches only this exact word for now because that's the only
# heading actually seen in real data - guessing at "notes"/"bibliography"/
# "further reading" without a real example to check against is exactly the
# kind of unverified heuristic this file's other comments warn against.
NOTES_HEADING = re.compile(r"^\s*sources\s*$", re.I)

# A solid list-item bullet OCRs as whatever glyph its font renders closest
# to - a ring reads as "°", a circled dot as "©", a diamond or star as "*".
# A footnote marker glyph gets misread the same way, and - unlike a list
# bullet - sits glued directly to the word with no space at all ("°This
# paradigm has recently been updated...", a real example: the footnote
# marker in the source PDF OCR'd as a degree sign with zero gap before the
# text). \s* (not \s+) has to accept that zero-space case, or the mark
# survives to verbalize(), which turns any bare "°" into " degrees " -
# "degrees This paradigm...". Each of these symbols means something real
# elsewhere (a unit's degree sign, an actual copyright line), so they can't
# be stripped everywhere verbalize() sees them - only right here, where a
# line is standing alone as the very first thing in a new paragraph, which
# is exactly where a list bullet or footnote marker (and nothing else
# legitimate) sits.
#
# This only covers the glyphs actually seen so far. A single bullet dot
# OCRs as whatever character its font happens to render closest to, and
# that turned out to include a plain Latin letter ("e Atomic structure...",
# a real example from the test book) - not a symbol at all, so no fixed
# character class can ever fully cover it. group_paragraphs() has a second,
# geometry-based check for exactly that case: see HANGING_INDENT_PT below.
BULLET_LEAD = re.compile(r"^[•◦▪▫‣⁃∙·°©®¢\*]\s*(?=\S)")

# The same footnote-marker misread, at the OTHER end of a sentence - a real
# example: "...contributed to each failure:°" (the marker glued onto the
# colon at the very end of a paragraph). BULLET_LEAD only ever looks at the
# front of a paragraph, so this survives untouched into verbalize(), which
# turns any bare "°" into " degrees " - "...to each failure: degrees". The
# character before the marker is what tells a genuine trailing degree sign
# ("at 100°") apart from a misread footnote mark: a real measurement is
# preceded by a digit, a footnote mark by a letter or sentence punctuation.
TRAILING_MARKER = re.compile(r"(?<=[A-Za-z:;,.])[•◦▪▫‣⁃∙·°©®¢\*]$")

# A hanging list marker (a bullet, or its OCR misread) sits to the LEFT of
# the paragraph's own wrapped text - "° Mechanical properties—relate..."
# starts at x0=167.8 (PDF points), but its wrapped continuation "include
# elastic modulus..." starts at x0=177.5, a real measurement from the test
# book (~9.7pt gap - normal OCR position jitter between lines of the same
# text is under 2pt, measured on this same page, so this is a real
# structural gap, not noise). Ordinary prose never does this: a paragraph's
# first line is either flush with its wrap (x0 equal) or indented further
# right for a traditional paragraph-opening indent (x0 greater) - never
# markedly to the LEFT of its own continuation. That makes the direction
# of the gap, not its specific
# character content, the reliable signal: whatever short token sits before
# the gap is a marker to drop, regardless of what glyph OCR assigned it -
# covers "e", "¢", or anything else BULLET_LEAD's fixed character class
# does not anticipate. A numbered marker ("1.", "(a)") gets the same
# hanging layout but is deliberately spared by requiring the token to be
# non-numeric - a reader should still hear "1." and "(a)" in a numbered
# list.
HANGING_INDENT_PT = 6

# See the footnote-detection comment in group_paragraphs for how these two
# are used together. Measured on the test book: real footnote separators
# sit at ~2.1-2.5x the page's own gap_break and ~0.79-0.87 down the page;
# these are set with margin on both sides of that.
FOOTNOTE_GAP_MULT = 1.8
FOOTNOTE_ZONE = 0.6

# On a native (non-scanned) PDF, "size" is a real font size, not OCR's
# noisy box-height proxy - and typesetting convention puts footnotes in
# distinctly smaller type than body. Measured on a real JSTOR chapter
# (a text-layer PDF, no OCR involved) where the gap+position check alone
# produced six false "footnote" starts across the chapter: a section
# heading measured 12pt against an 11pt body (LARGER, not smaller), and
# five ordinary paragraphs resuming after a quote or a heading measured
# 11pt (the SAME as body - block quotes get extra leading before and
# after, wide enough to clear FOOTNOTE_GAP_MULT on their own). The one
# genuine footnote-shaped block on every page of that same chapter (a
# JSTOR "This content downloaded..." notice) measured a real 8pt under
# 11-12pt body - a true drop every time. Requiring that drop rejects all
# six false positives and keeps the real one.
FOOTNOTE_SIZE_DROP = 0.5


def _size_changed(a, b):
    """A real font-size change, not the box-height jitter that OCR's
    per-line bounding boxes introduce between two lines of the same body
    text (ascenders/descenders, detection noise). Measured on a scanned
    page: two consecutive body lines came back as 9.49 and 10.15 - a 0.66pt
    gap that used to read as a style change and split the paragraph mid
    word. Native PDF font sizes don't have this noise, so only loosen the
    check where a line actually came from OCR."""
    diff = abs(a["size"] - b["size"])
    if a.get("font") == "ocr" or b.get("font") == "ocr":
        return diff > 0.20 * max(a["size"], b["size"], 1.0)
    return diff > 0.6


def _page_body_size(lines):
    """The page's own dominant type size - the mode of every native
    (non-OCR) line's size, rounded to the nearest half point so 10.95 and
    11.0 count as one bucket. None on a scanned page, where "size" is
    OCR's box-height proxy (see _size_changed) rather than a real font
    size and can't be pooled this way.

    Compared against `prev` instead: on a real JSTOR chapter's title page,
    the genuine footnote-shaped block ("Pluto Press is collaborating with
    JSTOR...", 9-10pt) sat right after that same page's own small-print
    JSTOR notice (also 9pt) - smaller than the page's actual dominant
    size (12pt, the title-block lines) but NOT smaller than `prev`, so a
    prev-only comparison rejected it. The page's dominant size is stable
    against a `prev` that happens to be atypical."""
    sizes = [round(l["size"] * 2) / 2 for l in lines if l.get("font") != "ocr"]
    return statistics.mode(sizes) if sizes else None


def _looks_like_footnote_size(l, body_size):
    """True if `l` is real footnote type - measurably smaller than the
    page's own dominant type size - or if `body_size` is None (a scanned
    page, where that size isn't trustworthy - see _page_body_size), in
    which case the gap+position check below has to stand on its own, same
    as before this existed."""
    if body_size is None:
        return True
    return l["size"] < body_size - FOOTNOTE_SIZE_DROP


# ------------------------------------------------- paragraph reassembly

def gap_break_for(lines):
    """The vertical gap that separates two visual blocks on this page, as
    opposed to two wrapped lines of the same paragraph. Pulled out of
    group_paragraphs so figures.py can ask the same question of the same
    page and get the same answer - see caption_starts() below."""
    gaps = [b["y0"] - a["y1"] for a, b in zip(lines, lines[1:])
            if b["y0"] > a["y1"] and b["y0"] - a["y1"] < 40]
    return (statistics.median(gaps) + 4) if gaps else 6


def is_isolated_start(l, prev, gap_break):
    """True when `l` opens a new visual block rather than continuing the one
    `prev` belongs to. This is the rule that stops "...as shown in Figure
    1.2. The model..." being mistaken for a caption; it is the *only* such
    rule, deliberately, so the figure index and the narration agree exactly
    on what a caption is."""
    return prev is None or (
        (l["y0"] - prev["y1"]) > gap_break
        or _size_changed(l, prev)
        or l.get("font") != prev.get("font")
        or (l["x0"] - prev["x0"]) > 8
    )


def caption_starts(lines):
    """Every line on a page that opens a real figure/table caption, as
    (index, line) pairs - the same lines group_paragraphs() drops from the
    narration. figures.py uses this to decide what to illustrate, so the two
    share one definition of "caption" rather than each carrying its own.

    Not quite identical, and deliberately so: group_paragraphs() runs its
    furniture filter first, so its `prev` skips running heads and page
    numbers, while this looks at every line. On a page whose caption sits
    immediately under a running head the two can therefore disagree - which
    costs at most one figure that never gets offered, or one candidate that
    fails placement matching and is dropped anyway. Neither outcome touches
    a word of what is read aloud."""
    gap_break = gap_break_for(lines)
    out, prev = [], None
    for i, l in enumerate(lines):
        s = l["text"].strip()
        if s and is_isolated_start(l, prev, gap_break) and CAPTION_START.match(s):
            out.append((i, l))
        prev = l
    return out


def group_paragraphs(lines, furniture, drop_captions, tag=None):
    """Walk the ordered lines and rebuild paragraphs, discarding furniture and
    figure captions (including their continuation lines).

    `tag`, if given, is called as `tag(line, category)` for every line as
    its fate is decided - "furniture", "caption", or "body" (kept). Used
    only by the triage view (diagnose_pages); costs nothing when None and
    changes no decision this function makes either way."""
    paras, cur = [], []
    # x0 of recent continuation lines (never a paragraph's own first line) -
    # a rolling estimate of "the real text column", used to recognize a
    # hanging list marker even on a single-line item that never gets a
    # second line of its own to compare against. See HANGING_INDENT_PT.
    recent_x0 = []
    gap_break = gap_break_for(lines)
    body_size = _page_body_size(lines)
    in_caption = False
    caption_lines = 0
    in_footnote = False
    prev = None

    for l in lines:
        s = l["text"].strip()
        edge = l["ypos"] < EDGE or l["ypos"] > 1 - EDGE

        if edge:
            s = strip_furniture(s, furniture)
            if not s:
                if tag: tag(l, "furniture")
                continue
        if edge and (PAGENUM.match(s) or RUNNING_HEAD.match(s) or VERSO_HEAD.match(s)):
            if tag: tag(l, "furniture")
            continue

        # "Figure 1.2." also opens plenty of ordinary sentences mid-paragraph
        # ("...as shown in Figure 1.2. The model represented by this
        # diagram..."). Only trust the caption match when this line is
        # actually a new visual block - the same signals newpara uses below -
        # otherwise it isn't a caption, it's prose that happens to reference
        # one, and matching on text alone was deleting the next 1-3 real
        # sentences after it.
        isolated_start = is_isolated_start(l, prev, gap_break)

        if drop_captions and isolated_start and CAPTION_START.match(s):
            if tag: tag(l, "caption")
            if cur:
                paras.append(cur)
                cur = []
            in_caption = True
            caption_lines = 0
            prev = l
            continue

        if in_caption:
            # Caption continuation lines wrap from the caption above, so they
            # start mid-sentence. Anything starting with a capital is body text
            # resuming - err on the side of keeping it, since dropping real
            # prose is far worse than reading one stray caption line.
            #
            # A second opinion flagged that a lowercase start alone could
            # also match unrelated body text resuming from the opposite
            # column, and suggested also requiring the continuation to match
            # the caption's own font/size. Tried it - it regressed a real,
            # already-verified case: a genuine 4-line figure caption on the
            # actual test book, where OCR's per-line size noise (the same
            # noise _size_changed exists to tolerate elsewhere) made line 2
            # of the caption look like a "size change" from line 1, so only
            # the first line got dropped and the other three leaked into
            # the narration. Reverted - no real example of the cross-column
            # failure this was meant to prevent, but a confirmed regression
            # from trying to prevent it.
            #
            # Verified again directly against that same test book while
            # chasing a reported "caption leaks into body" bug: the line the
            # report pointed at ("The disk on the left is transparent...")
            # is not a caption continuation at all - it's a genuine new body
            # paragraph explaining the figure, correctly excluded by this
            # exact lowercase-start check, right after a correctly-captured
            # 4-line caption. So this check was already doing its job; no
            # change needed here.
            #
            # Except lowercase-start alone is still wrong the other
            # direction, confirmed by a second real example on the same
            # page: a figure credit line - "(Reprinted with permission of
            # Earl R. Parker, Brittle Behavior of Engineering" - wraps mid
            # book-title onto "Structures, National Academy of Sciences...",
            # a capitalized continuation, not a new sentence. The reliable
            # signal isn't the next line's case, it's whether the line
            # *before* it ended a sentence: no terminal punctuation means
            # the sentence is grammatically unfinished and must continue
            # regardless of case; terminal punctuation means the previous
            # heuristic (lowercase-start, under a line cap) is the right
            # fallback for the genuinely ambiguous case.
            close = prev is not None and (l["y0"] - prev["y1"]) < gap_break * 1.5
            prev_unfinished = prev is not None and not re.search(r"[.!?]\s*$", prev["text"])
            continues = close and caption_lines < 12 and (
                prev_unfinished or re.match(r"^[a-z(]", s))
            if continues:
                if tag: tag(l, "caption")
                caption_lines += 1
                prev = l
                continue
            in_caption = False

        # See NOTES_HEADING: a bare "sources" heading is a much stronger,
        # more specific signal than the gap+position check below, so it
        # doesn't need either of them - text alone is trustworthy here
        # exactly because a real paragraph could never wrap down to just
        # this one word on its own line (isolated_start rules that out).
        # Reuses in_footnote/the same "drop to end of page" fate as a real
        # footnote - correct for the page the heading itself is on, but
        # this doesn't persist across a page boundary (group_paragraphs
        # runs once per page - see its docstring), so a citation essay
        # that spills onto a second or third page currently still reads
        # as body there. Chasing that would mean deciding, with no
        # structural signal at all, where such a block quietly ends and
        # real narrative resumes on a later page - exactly the kind of
        # guess that could eat real body text instead of just missing a
        # citation, so it's left alone rather than risked.
        if (not in_footnote and not in_caption and not edge and prev is not None
                and isolated_start and NOTES_HEADING.match(s)):
            if tag: tag(l, "footnote")
            if cur:
                paras.append(cur)
                cur = []
            in_footnote = True
            prev = l
            continue

        # A footnote block sits below a real separator (a rule or extra
        # whitespace in print) that produces a much bigger gap than any
        # ordinary paragraph break - measured on the test book, ~26-27pt
        # against a typical body gap_break of ~11-13pt, comfortably above
        # FOOTNOTE_GAP_MULT. Position alone can't tell a footnote from body
        # text that simply runs to the bottom of a page (both happen), and
        # gap alone can't tell it from a big paragraph break higher up the
        # page - it's the two together, low on the page AND a gap wider
        # than a paragraph break ever produces, that's distinctive.
        # Everything from there to the end of the page is treated as the
        # footnote - real footnotes never hand back to body text midway
        # down the same column. On a scanned page, font/size still can't
        # gate entry (OCR's own per-line box-height noise swings ~7-13pt
        # within one footnote block on that book), so _looks_like_footnote_size
        # is a no-op there and this is gap+position alone, same as before
        # it existed. On a native PDF it also requires real type size to
        # have dropped - see FOOTNOTE_SIZE_DROP for why that's needed.
        #
        # Must run after the CAPTION_START check, not before: a figure
        # caption sitting below its photo has exactly the same big-gap,
        # low-on-the-page shape (the photo itself is the gap), and this
        # check ran first the first time, misclassifying a real caption
        # ("Figure 1.3 The Liberty ship...") as a footnote.
        if (not in_footnote and not in_caption and not edge and prev is not None
                and l["ypos"] > FOOTNOTE_ZONE
                and (l["y0"] - prev["y1"]) > gap_break * FOOTNOTE_GAP_MULT
                and _looks_like_footnote_size(l, body_size)):
            if tag: tag(l, "footnote")
            if cur:
                paras.append(cur)
                cur = []
            in_footnote = True
            prev = l
            continue

        if in_footnote:
            if tag: tag(l, "footnote")
            prev = l
            continue

        if prev is not None:
            newpara = (
                (l["y0"] - prev["y1"]) > gap_break
                or _size_changed(l, prev)
                or l.get("font") != prev.get("font")
                or (l["x0"] - prev["x0"]) > 8            # indented first line
                or re.search(r"[.!?:]\s*$", prev["text"]) and l["x0"] < prev["x0"] - 8
            )
            if newpara and cur:
                paras.append(cur)
                cur = []
        if not cur:
            s = BULLET_LEAD.sub("", s, count=1)
            # The single most recent continuation line, not a wider
            # average - a bulleted list sits right after ordinary
            # paragraphs with a completely different flush indent, and
            # averaging across that boundary dilutes the list's own
            # indent with leftover samples from the paragraph before it,
            # which is exactly what let this check miss "e Atomic
            # structure..." (second item in a list) the first time
            # around. The most recent sample is always from whatever
            # block we're actually inside right now.
            if recent_x0 and recent_x0[-1] - l["x0"] > HANGING_INDENT_PT:
                parts = s.split(None, 1)
                if len(parts) == 2 and len(parts[0]) <= 2 and not any(
                        c.isdigit() for c in parts[0]):
                    s = parts[1]
        elif len(s) >= 20:
            # A real continuation line is prose wrapping across most of a
            # column, so it's almost always long. A short one is exactly
            # the shape of the noise that occasionally survives is_ocr_junk
            # from a busy figure - a real example, a 2-word photo-bleed
            # fragment ("anid ax") that happened to contain one clean
            # 4-letter word, became this baseline once, and the next
            # paragraph starting with the real word "In" got mistaken for
            # a hanging bullet marker and silently truncated to "addition
            # to structure...". Missing a legitimate short continuation
            # just leaves the previous (still probably-correct) baseline
            # in place; trusting a short line that turns out to be noise
            # corrupts real text - not a symmetric trade-off.
            recent_x0.append(l["x0"])
        cur.append(s)
        if tag: tag(l, "body")
        prev = l

    if cur:
        paras.append(cur)
    return paras


def join_lines(lines):
    """Un-hyphenate across line breaks, then reflow to running prose."""
    out = ""
    for line in lines:
        if not out:
            out = line
            continue
        if re.search(r"[a-zé]-$", out) and re.match(r"^[a-z]", line):
            out = out[:-1] + line          # de-  signer -> designer
        else:
            out = out + " " + line
    return out


# ========================================================= speaking the math

GREEK = {
    "α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon",
    "ϵ": "epsilon", "ζ": "zeta", "η": "eta", "θ": "theta", "ϑ": "theta",
    "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu", "µ": "mu", "ν": "nu",
    "ξ": "xi", "ρ": "rho", "ς": "sigma", "σ": "sigma", "τ": "tau",
    "υ": "upsilon", "φ": "phi", "ϕ": "phi", "χ": "chi", "ψ": "psi",
    "ω": "omega", "π": "pi", "Γ": "gamma", "Δ": "delta", "Θ": "theta",
    "Λ": "lambda", "Ξ": "xi", "Π": "pi", "Σ": "sigma", "Φ": "phi",
    "Ψ": "psi", "Ω": "omega",
}

OPERATORS = {
    "∫": " the integral of ", "∮": " the closed line integral of ",
    "∬": " the double integral of ", "∭": " the triple integral of ",
    "∂": " partial ", "∇": " del ", "√": " the square root of ",
    "∛": " the cube root of ", "∞": " infinity ", "±": " plus or minus ",
    "∓": " minus or plus ", "×": " times ", "⋅": " times ", "·": " times ",
    "÷": " divided by ", "−": " minus ",
    "≈": " approximately equals ", "≅": " approximately equals ",
    "≃": " approximately equals ", "≡": " is identical to ",
    "≠": " does not equal ", "≤": " is less than or equal to ",
    "≥": " is greater than or equal to ", "⩽": " is less than or equal to ",
    "⩾": " is greater than or equal to ", "≪": " is much less than ",
    "≫": " is much greater than ", "→": " goes to ", "⇒": " implies ",
    "↔": " corresponds to ", "∝": " is proportional to ", "∴": " therefore ",
    "∵": " because ", "∈": " in ", "∀": " for all ", "∃": " there exists ",
    "∅": " the empty set ", "∠": " angle ", "⊥": " perpendicular to ",
    "∥": " parallel to ", "°": " degrees ", "′": " prime ", "″": " double prime ",
    "ℏ": " h bar ", "∑": " the sum of ", "∏": " the product of ",
    "…": " and so on ", "≜": " is defined as ", "≐": " equals ",
}

SUPERS = {"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5",
          "⁶": "6", "⁷": "7", "⁸": "8", "⁹": "9", "⁻": "-", "⁺": "+", "ⁿ": "n"}
SUBS = {"₀": "0", "₁": "1", "₂": "2", "₃": "3", "₄": "4", "₅": "5", "₆": "6",
        "₇": "7", "₈": "8", "₉": "9", "ₓ": "x", "ₙ": "n", "ₐ": "a", "ₑ": "e"}
VULGAR = {"½": " one half ", "⅓": " one third ", "⅔": " two thirds ",
          "¼": " one quarter ", "¾": " three quarters ", "⅛": " one eighth "}

# Units expand only where a number precedes them. Bare "m" and "N" in a
# textbook are far more often variables than metres and newtons, and expanding
# those turns every sentence to mush. Longest match first: kN·m before kN.
S = r"[·⋅\*\-\s]"
UNITS = [
    (rf"kN{S}*m\b", "kilonewton meters"), (rf"N{S}*m\b", "newton meters"),
    (rf"lb{S}*ft\b", "pound feet"), (rf"ft{S}*lb\b", "foot pounds"),
    (rf"W/m{S}*K\b", "watts per meter kelvin"),
    (rf"J/kg{S}*K\b", "joules per kilogram kelvin"),
    (r"m\s*/\s*s\s*[²2]\b", "meters per second squared"),
    (r"m\s*/\s*s\b", "meters per second"),
    (r"ft\s*/\s*s\s*[²2]\b", "feet per second squared"),
    (r"ft\s*/\s*s\b", "feet per second"),
    (r"km\s*/\s*h\b", "kilometers per hour"),
    (r"kg\s*/\s*m\s*[³3]\b", "kilograms per cubic meter"),
    (r"kg\s*/\s*m\s*[²2]\b", "kilograms per square meter"),
    (r"kN\s*/\s*m\s*[²2]\b", "kilonewtons per square meter"),
    (r"kN\s*/\s*m\b", "kilonewtons per meter"),
    (r"N\s*/\s*m\s*[²2]\b", "newtons per square meter"),
    (r"N\s*/\s*m\s*m\s*[²2]\b", "newtons per square millimeter"),
    (r"N\s*/\s*m\b", "newtons per meter"),
    (r"rad\s*/\s*s\b", "radians per second"),
    (r"GPa\b", "gigapascals"), (r"MPa\b", "megapascals"),
    (r"kPa\b", "kilopascals"), (r"Pa\b", "pascals"),
    (r"°\s*C\b", "degrees Celsius"), (r"°\s*F\b", "degrees Fahrenheit"),
    (r"ksi\b", "k s i"), (r"psi\b", "p s i"),
    (r"rpm\b", "revolutions per minute"),
    (r"kHz\b", "kilohertz"), (r"MHz\b", "megahertz"), (r"Hz\b", "hertz"),
    (r"mm\s*[³3]\b", "cubic millimeters"), (r"mm\s*[²2]\b", "square millimeters"),
    (r"cm\s*[³3]\b", "cubic centimeters"), (r"cm\s*[²2]\b", "square centimeters"),
    (r"m\s*[³3]\b", "cubic meters"), (r"m\s*[²2]\b", "square meters"),
    (r"[μµ]m\b", "micrometers"), (r"nm\b", "nanometers"),
    (r"mm\b", "millimeters"), (r"cm\b", "centimeters"), (r"km\b", "kilometers"),
    (r"kg\b", "kilograms"), (r"kN\b", "kilonewtons"), (r"MN\b", "meganewtons"),
    (r"kJ\b", "kilojoules"), (r"MJ\b", "megajoules"),
    (r"kW\b", "kilowatts"), (r"MW\b", "megawatts"),
    (r"mA\b", "milliamps"), (r"kΩ\b", "kilohms"), (r"Ω\b", "ohms"),
    (r"lbf\b", "pounds force"), (r"lbm\b", "pounds mass"), (r"hp\b", "horsepower"),
    (r"in\.(?!\w)", "inches"), (r"ft\b", "feet"),
    (r"(?<=\s)m\b", "meters"), (r"(?<=\s)s\b", "seconds"), (r"(?<=\s)N\b", "newtons"),
    (r"(?<=\s)J\b", "joules"), (r"(?<=\s)W\b", "watts"), (r"(?<=\s)V\b", "volts"),
]

ABBREV = [
    (r"\bEqs?\.\s*\(\s*([\d][\d\.\-]*[a-z]?)\s*\)", r"Equation \1"),
    (r"\bEqs?\.\s*", "Equation "), (r"\bFigs?\.\s*", "Figure "),
    (r"\bSec\.\s*", "Section "), (r"\bCh\.\s*", "Chapter "),
    (r"\bApp\.\s*", "Appendix "), (r"\bRef\.\s*", "Reference "),
    (r"\bet al\.", "and others"),
    # "i.e." OCR'd off a scan comes back mangled two different ways in
    # practice: the middle period dropped ("ie.") or the e misread as a c
    # ("i.c.", "i.c") - both verified against actual Tesseract output on a
    # scanned page. Catch those alongside the clean form.
    (r"\bi\.?[ec]\.,?", "that is,"),
    (r"\be\.g\.,?", "for example,"), (r"\bvs\.", "versus"),
    (r"\bcf\.", "compare"), (r"\bapprox\.", "approximately"),
    (r"\bw\.r\.t\.", "with respect to"), (r"\bAssy\.", "assembly"),
]


def _supers(m):
    d = "".join(SUPERS.get(c, c) for c in m.group(1))
    if d == "2":
        return " squared"
    if d == "3":
        return " cubed"
    if d.startswith("-"):
        return f" to the power negative {d[1:]}"
    return f" to the power {d}"


# --------------------------------------------------------------------------
# verbalize() runs on every paragraph of every document, and the tables above
# add up to about ninety substitutions per call. Two things cost most of that
# and neither did any work: the patterns were re-parsed from strings on every
# call (the UNITS loop even concatenated its lookbehind on each pass), and
# every table was walked in full over text that could not possibly match it -
# a paragraph of ordinary prose has no Greek letter, no integral sign, and no
# digit for a unit to attach itself to.
#
# So the patterns are compiled once, here, and each table gets a gate: one
# character-class scan that is a strict necessary condition for anything in
# that table to match. A gate can only skip work that would have changed
# nothing, so the output is identical - checked directly against the previous
# implementation over the text of real books, as well as by the smoke suite.

def _class_gate(chars):
    """A regex matching any one of `chars`, for asking in a single pass
    whether a table of substitutions can possibly apply at all."""
    return re.compile("[" + "".join(re.escape(c) for c in sorted(set(chars))) + "]")


_ABBREV_RE = [(re.compile(pat), rep) for pat, rep in ABBREV]

# Every UNITS pattern is anchored behind a digit, a closing paren or a
# superscript digit. With none of those anywhere in the text, not one of the
# sixty-odd can fire - so the gate is that same character class, built from the
# same string as the lookbehind rather than spelled out a second time. Written
# out separately it was not the same class: \d matches every Unicode decimal
# digit, so "\u0663 mm" (an Arabic-Indic three) satisfied the lookbehind but
# not a gate listing 0-9, and the unit stopped being expanded.
_UNIT_CLASS = r"[\d\)\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079]"
_UNIT_PREFIX = r"(?<=" + _UNIT_CLASS + r")\s*"
_UNITS_RE = [(re.compile(_UNIT_PREFIX + pat), " " + rep) for pat, rep in UNITS]
_UNIT_GATE = re.compile(_UNIT_CLASS)

_VULGAR_GATE = _class_gate(VULGAR)
_OPERATOR_GATE = _class_gate(k[0] for k in OPERATORS)
_GREEK_GATE = _class_gate(k[0] for k in GREEK)

_SUPER_RE = re.compile(r"([\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207b\u207a\u207f]+)")
_SUB_RE = re.compile(r"([\u2080\u2081\u2082\u2083\u2084\u2085\u2086\u2087\u2088\u2089\u2093\u2099\u2090\u2091]+)")
_ASCII_POW_RE = re.compile(r"\^\s*\{?\s*(-?\w{1,3})\s*\}?")
_ASCII_SUB_RE = re.compile(r"(?<=[A-Za-z\u03b1-\u03c9\u0391-\u03a9])_\s*\{?\s*([A-Za-z0-9]{1,4})\s*\}?")
_NUM_RANGE_RE = re.compile(r"(\d)\s*[\u2013\u2014]\s*(\d)")
_DASH_RE = re.compile(r"\s*[\u2013\u2014]\s*")
_EQUALS_RE = re.compile(r"\s*=\s*")
_SLASH_EQ_RE = re.compile(r"\s*/\s*")
_SLASH_PROSE_RE = re.compile(r"\b([A-Za-z0-9]{1,3})\s*/\s*([A-Za-z0-9]{1,3})\b")
_CITATION_RE = re.compile(r"\s*\[\d+(?:\s*[,\u2013-]\s*\d+)*\]")
_RUNS_RE = re.compile(r"[ \t]{2,}")
_SPACE_PUNCT_RE = re.compile(r"\s+([,.;:!?])")


def _sub_words(m):
    return " sub " + " ".join(SUBS.get(c, c) for c in m.group(1)) + " "


def _ascii_pow(m):
    e = m.group(1)
    if e == "2":
        return " squared "
    if e == "3":
        return " cubed "
    return " to the power " + e.replace("-", "negative ") + " "


def _slash_prose(m):
    """A slash between quantities is a fraction; between words it is not, so
    "and/or" survives."""
    a, b = m.group(1), m.group(2)
    symbolic = (lambda t: t.isdigit() or len(t) == 1)
    return f"{a} over {b}" if symbolic(a) or symbolic(b) else m.group(0)


def verbalize(text, mode="speak", equation=False):
    if mode == "raw":
        return text

    if not equation:
        text = TRAILING_MARKER.sub("", text)

    for pat, rep in _ABBREV_RE:
        text = pat.sub(rep, text)

    if _UNIT_GATE.search(text):
        for pat, rep in _UNITS_RE:
            text = pat.sub(rep, text)

    if _VULGAR_GATE.search(text):
        for sym, word in VULGAR.items():
            text = text.replace(sym, word)

    text = _SUPER_RE.sub(_supers, text)
    text = _SUB_RE.sub(_sub_words, text)

    # ASCII exponents and subscripts: x^2, sigma_max, C_{p}
    if "^" in text:
        text = _ASCII_POW_RE.sub(_ascii_pow, text)
    if "_" in text:
        text = _ASCII_SUB_RE.sub(r" sub \1 ", text)

    # An en dash between numbers is a range - "1540-1769" is read "to", not
    # "minus". Only a true minus sign means subtraction outside equations.
    if "\u2013" in text or "\u2014" in text:
        text = _NUM_RANGE_RE.sub(r"\1 to \2", text)
        if not equation:
            text = _DASH_RE.sub(", ", text)

    if _OPERATOR_GATE.search(text):
        for sym, word in OPERATORS.items():
            text = text.replace(sym, word)
    if _GREEK_GATE.search(text):
        for sym, word in GREEK.items():
            text = text.replace(sym, " " + word + " ")

    if "=" in text:
        text = _EQUALS_RE.sub(" equals ", text)
    if "/" in text:
        text = (_SLASH_EQ_RE.sub(" over ", text) if equation
                else _SLASH_PROSE_RE.sub(_slash_prose, text))

    if "[" in text:
        text = _CITATION_RE.sub("", text)               # citation marks
    text = _RUNS_RE.sub(" ", text)
    text = _SPACE_PUNCT_RE.sub(r"\1", text)
    return text.strip()


EQ_TAG = re.compile(r"\(\s*(\d+[\.\-]\d+[a-z]?|\d{1,3})\s*\)\s*$")

_EQ_SYMBOLS = "=+-\u2212*/^_(){}[]<>\u2264\u2265\u2248\u222b\u2211\u2202\u2207\u221a\u00b1\u00d7\u00b7\u22c5"


def is_display_equation(line):
    if len(line) > 130:
        return False
    # map() over the method descriptor keeps the per-character test in C, and
    # the early return skips the second pass over any line that could not
    # qualify however its non-letters are made up: every character counted as
    # a symbol below is itself a non-letter, so (length - letters) is a
    # ceiling on the symbol count.
    letters = sum(map(str.isalpha, line))
    ceiling = len(line) - letters
    if ceiling < 3 or ceiling < letters * 0.55:
        return False
    symbols = sum(c in _EQ_SYMBOLS or c.isdigit() for c in line)
    return symbols >= 3 and symbols >= letters * 0.55


# =============================================================== assembly

def chunk(text, limit=4200):
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", text)
    out, cur = [], ""
    for s in sentences:
        if len(cur) + len(s) + 1 > limit and cur:
            out.append(cur.strip())
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def parse_pages(spec, total):
    if not spec:
        return list(range(total))
    out = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a) - 1, min(int(b), total)))
        else:
            out.append(int(part) - 1)
    return [p for p in out if 0 <= p < total]


def _furniture_for(doc, idxs, ocr, pre):
    """Running heads repeat across the whole book, so sample widely even
    when only a short range was asked for - otherwise a 3-page range never
    sees them repeat and every page opens with the book's title read
    aloud. On a scan that sampling would mean OCRing pages nobody asked
    for, which costs seconds each, so there we make do with the pages in
    hand. Returns (per_page lines, furniture set) - shared by
    _convert_paragraphs and the triage view's diagnose_pages() so both
    agree on what counts as furniture for the same page range."""
    per_page = [ordered_lines(doc[i], ocr, pre) for i in idxs]
    scanned = any(l.get("ocr") for lines in per_page for l in lines)
    if scanned or len(idxs) > 12:
        sample = per_page
    else:
        # The wider sample deliberately includes the pages actually asked for,
        # and those have just been read - reuse them rather than running the
        # whole line-extraction and ordering pass over them a second time.
        have = dict(zip(idxs, per_page))
        sample_idx = sorted(set(idxs) | set(range(0, doc.page_count,
                                                  max(1, doc.page_count // 24))))
        sample = [have[i] if i in have else ordered_lines(doc[i], ocr, pre)
                  for i in sample_idx]
    furniture = find_furniture(sample, 0.25 if scanned else 0.3, min_pages=len(idxs))
    return per_page, furniture


# ------------------------------------------------------ progressive reading
#
# A 900-page scan takes minutes to OCR. Nothing about that has to happen
# before the first page is readable, so these are the pieces that let a
# document be opened from its first two pages and grown from behind: one page
# converted at a time, in reading order, with failures isolated to the page
# that caused them.
#
# There is exactly one conversion implementation and it lives here.
# _convert_paragraphs() (the whole-document path) and progressive ingestion
# both go through paragraphs_for_page() and merge_paragraphs(), so a scanned
# book reads identically whichever way it was opened.


@dataclass(frozen=True)
class PageResult:
    """One page's paragraphs, or why there are none.

    `error` is set for a page that could not be read at all. It is a record,
    not an exception, because page k failing must never stop page k+1 from
    being readable - a book with one unreadable page is a book with one
    unreadable page, not a book that will not open.
    """
    page: int
    paragraphs: tuple = ()
    error: str = None


@dataclass(frozen=True)
class PdfIngestionPlan:
    """What opening this PDF is going to involve, worked out cheaply."""
    path: str
    selected_pages: tuple
    progressive: bool
    source_fingerprint: str
    math: str = "skip"
    drop_captions: bool = True
    page_count: int = 0


def source_fingerprint(path):
    """Names this exact file as it is now. Size and mtime rather than a hash
    of 300MB: the question being asked is only ever "is this still the file we
    started reading", which a replaced or edited file fails on either."""
    try:
        st = os.stat(path)
    except OSError:
        return ""
    return f"{os.path.abspath(path)}:{st.st_size}:{st.st_mtime_ns}"


def plan_ingestion(path, pages=None, math="skip", ocr="auto", drop_captions=True,
                   probe=2):
    """Decide whether this document can be opened a page at a time.

    Deliberately cheap: it opens the PDF, resolves the page selection, and
    asks whether the *first couple* of selected pages need OCR. Walking 900
    pages to find out how to open the first one would be its own version of
    the mistake this is trying to avoid.

    Progressive when the opening pages need OCR and there are more than two of
    them - that is the fully scanned book, the case where waiting for the
    whole conversion means waiting minutes. A PDF with a text layer converts
    fast enough that the ordinary path is simpler and no slower; a mixed PDF
    whose scanned pages start later also takes the ordinary path, and reads
    correctly, just without the early first page.
    """
    doc = pymupdf.open(path)
    try:
        idxs = tuple(parse_pages(pages, doc.page_count))
        scanned_head = any(needs_ocr(doc[i], ocr) for i in idxs[:probe])
        return PdfIngestionPlan(
            path=path, selected_pages=idxs,
            progressive=bool(scanned_head and len(idxs) > probe),
            source_fingerprint=source_fingerprint(path), math=math,
            drop_captions=drop_captions, page_count=doc.page_count)
    finally:
        doc.close()


def paragraphs_for_page(lines, page, furniture, math="speak", drop_captions=True):
    """One page's lines -> its paragraphs, in spoken form.

    An equation doesn't force a paragraph break on its own (that decision is
    purely geometric, in group_paragraphs), so prose before and after one
    routinely land in the same para - "The relation is: [equation] where E
    is..." Collecting all the body lines into one bucket and all the equations
    into another used to flush the whole bucket first, then every equation
    after it, so that "where E is..." got read before "sigma equals E epsilon"
    no matter which came first on the page. Flushing body text in place, right
    before each equation it actually precedes, keeps them in their real order.
    """
    out = []
    for para in group_paragraphs(lines, furniture, drop_captions):
        body = []

        def flush_body():
            joined = join_lines(body)
            if joined:
                out.append(verbalize(joined, math))
            body.clear()

        for ln in para:
            if is_display_equation(ln):
                flush_body()
                tag = EQ_TAG.search(ln)
                if math == "skip":
                    e = f"Equation {tag.group(1)}." if tag else None
                else:
                    spoken = verbalize(EQ_TAG.sub("", ln), math, equation=True)
                    e = spoken if spoken.endswith((".", "?", "!")) else spoken + "."
                if e:
                    out.append(e)
            else:
                body.append(ln)
        flush_body()
    return PageResult(page, tuple(out))


_TERMINATED = re.compile(r"[.!?:;]\"?\s*$")


def _continues(previous, nxt):
    return not _TERMINATED.search(previous) and (nxt[:1].islower() or nxt[:1].isdigit())


def _stitch(previous, nxt):
    # A word hyphenated right at the break gets the same treatment join_lines
    # gives one inside a single paragraph - otherwise a false paragraph split
    # (a noisy OCR line-height jump, say) leaves the hyphen and a stray space
    # behind: "dif- ferent".
    if re.search(r"[a-zé]-$", previous) and nxt[:1].islower():
        return previous[:-1] + nxt
    # A chapter-opening drop cap ("M", large font) is its own paragraph next to
    # the rest of that first word ("aterials...", body-sized font) - a plain
    # letter with nothing else. Joining it with a space would speak the letter
    # on its own ("M... aterials").
    if len(previous) == 1 and previous.isupper():
        return previous + nxt
    return previous + " " + nxt


def merge_paragraphs(pairs, carry=None, flush=True):
    """Stitch paragraphs broken across a column or page boundary.

    A merged paragraph keeps the page of its *first* piece, not wherever it
    ends.

    `carry` and `flush` are what make this usable a page at a time. A
    progressive publisher cannot retroactively change a paragraph it has
    already shown, so with flush=False the last paragraph is held back if it
    looks unfinished, and handed to the next call as `carry` to be joined with
    whatever comes next. Returns (pairs, carry).

    At the very first publication, holding the carry back can leave nothing to
    read at all. That is the caller's call to make - pass flush=True - because
    a first page the reader can actually hear outranks a perfect join across
    its bottom edge.
    """
    merged = []
    if carry is not None:
        merged.append([carry[0], carry[1]])
    for text, page in pairs:
        if merged and _continues(merged[-1][0], text):
            merged[-1][0] = _stitch(merged[-1][0], text)
        else:
            merged.append([text, page])
    out = [(t, p) for t, p in merged if len(t.strip()) > 1]
    if flush or not out or _TERMINATED.search(out[-1][0]):
        return out, None
    return out[:-1], out[-1]


def _cached_page(cache, index):
    """A previously OCR'd page, or None. A corrupt cache file is not an error
    to report - it is a page to read again."""
    path = os.path.join(cache, f"{index:05d}.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            lines = json.load(fh)
        return lines if isinstance(lines, list) else None
    except Exception:
        return None


def _cache_page(cache, index, lines):
    """Write a page's OCR atomically. A crash halfway through a plain write
    leaves JSON that looks complete and is not, and the next run would trust
    it - so the file only ever appears whole."""
    final = os.path.join(cache, f"{index:05d}.json")
    tmp = f"{final}.{os.getpid()}.part"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(lines, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, final)
    except Exception:
        try:
            os.remove(tmp)
        except OSError:
            pass


class _Ready:
    """A page that needed no work - already cached, or cancelled before it
    started. Shaped like a future so the loop below has one kind of thing."""

    __slots__ = ("_value", "_error")

    def __init__(self, value=None, error=None):
        self._value = value
        self._error = error

    def result(self, timeout=None):
        if self._error is not None:
            raise RuntimeError(self._error)
        return self._value

    def cancel(self):
        return True


def iter_ocr_pages(path, indices, *, dpi=OCR_DPI, workers=None, max_inflight=None,
                   cancelled=None, progress=None, ocr_one=None):
    """Yield (page_index, lines, error) for each page, in selected-page order.

    The difference from ocr_many() is failure granularity and memory. ocr_many
    reads every page into one dict and, if the process pool fails, redoes the
    lot serially; that is right for a conversion somebody is waiting on and
    wrong for a 900-page background scan where one bad page must not cost the
    other 899.

    Ordered output with a bounded window is what keeps this flat: a few pages
    in flight, a few finished results held while the frontier catches up, and
    nothing else. Two Tesseract processes by default rather than all of them -
    this runs behind a reader who is listening to page one, and a background
    scan must not take the CPU that narration
    needs.
    """
    indices = list(indices)
    if not indices:
        return
    stop = cancelled or (lambda: False)
    submit_one = ocr_one or _ocr_one
    cache = cache_dir_for(path)
    workers = workers or max(1, min(2, (os.cpu_count() or 2) - 2))
    window = max(1, max_inflight or 2 * workers)

    pool = None
    try:
        pool = ProcessPoolExecutor(max_workers=workers,
                                   initializer=_ocr_worker_init)
    except Exception as e:
        # No pool: one page at a time, in this process. Slower, still ordered,
        # still isolated per page.
        print(f"  parallel OCR unavailable ({e}); falling back to one at a time",
              file=sys.stderr)

    inflight = {}
    cursor = 0
    done = 0
    try:
        for page in indices:
            while cursor < len(indices) and len(inflight) < window:
                nxt = indices[cursor]
                cursor += 1
                held = _cached_page(cache, nxt)
                if held is not None:
                    inflight[nxt] = (_Ready(held), True)
                elif stop():
                    inflight[nxt] = (_Ready(error="cancelled"), True)
                elif pool is None:
                    inflight[nxt] = (None, False)      # read it when we reach it
                else:
                    inflight[nxt] = (pool.submit(submit_one, (path, nxt, dpi)), False)
            future, from_cache = inflight.pop(page)
            if stop():
                yield page, None, "cancelled"
                continue
            try:
                if future is None:
                    doc = pymupdf.open(path)
                    try:
                        lines = ocr_lines(doc[page], dpi)
                    finally:
                        doc.close()
                else:
                    result = future.result()
                    lines = result[1] if isinstance(result, tuple) else result
            except Exception as e:
                yield page, None, f"{type(e).__name__}: {e}"
                done += 1
                if progress:
                    progress(done, len(indices))
                continue
            if not from_cache and lines is not None and not stop():
                _cache_page(cache, page, lines)
            done += 1
            if progress:
                progress(done, len(indices))
            yield page, lines, None
    finally:
        for future, _cached in inflight.values():
            if future is not None:
                try:
                    future.cancel()
                except Exception:
                    pass
        if pool is not None:
            pool.shutdown(wait=False, cancel_futures=True)


def iter_page_lines(path, indices, *, ocr="auto", dpi=OCR_DPI, workers=None,
                    max_inflight=None, cancelled=None, progress=None):
    """Yield (page_index, lines, error) for each page, in order, OCRing the
    ones that need it and reading the text layer of the ones that do not.

    This is the single page source for progressive ingestion. Call it with the
    slice you actually want: it asks needs_ocr() of every index it is given
    before it starts, and asking that of 900 pages before the first one can be
    read is exactly the delay this whole feature exists to remove.
    """
    indices = list(indices)
    if not indices:
        return
    stop = cancelled or (lambda: False)
    doc = pymupdf.open(path)
    try:
        scanned = [i for i in indices if needs_ocr(doc[i], ocr)]
        held = set(scanned)
        stream = (iter_ocr_pages(path, scanned, dpi=dpi, workers=workers,
                                 max_inflight=max_inflight, cancelled=cancelled)
                  if scanned else iter(()))
        done = 0
        for index in indices:
            if stop():
                return
            pre = None
            if index in held:
                try:
                    page, lines, error = next(stream)
                except StopIteration:
                    page, lines, error = index, None, "OCR stopped early"
                if page != index:
                    # Ordering is the contract; a mismatch means a bug, and
                    # guessing which page this is would put the wrong text
                    # under the wrong page number.
                    error = error or f"OCR returned page {page}, expected {index}"
                if error is not None:
                    done += 1
                    if progress:
                        progress(done, len(indices))
                    yield index, None, error
                    continue
                pre = {index: lines}
            try:
                lines = ordered_lines(doc[index], ocr, pre)
                error = None
            except Exception as e:
                lines, error = None, f"{type(e).__name__}: {e}"
            done += 1
            if progress:
                progress(done, len(indices))
            yield index, lines, error
    finally:
        doc.close()


def _convert_paragraphs(path, pages=None, math="speak", drop_captions=True, ocr="auto",
                         progress=None):
    """Does the actual work for convert() below, but stops one step short of
    joining the paragraphs into one string - it also returns, for each
    paragraph, the 0-based PDF page index it started on. convert() just
    joins and discards that; chapters.py's block-mapping needs to keep it,
    to translate a chapter's detected starting page into a block index
    without re-deriving one from the other."""
    doc = pymupdf.open(path)
    idxs = parse_pages(pages, doc.page_count)

    scan_idx = [i for i in idxs if needs_ocr(doc[i], ocr)]
    pre = ocr_many(path, scan_idx, progress=progress) if scan_idx else {}

    per_page, furniture = _furniture_for(doc, idxs, ocr, pre)
    results = [paragraphs_for_page(lines, idxs[k], furniture, math, drop_captions)
               for k, lines in enumerate(per_page)]
    doc.close()

    pairs, _carry = merge_paragraphs(
        [(text, r.page) for r in results for text in r.paragraphs])
    return [text for text, _p in pairs], [page for _t, page in pairs]


def convert(path, pages=None, math="speak", drop_captions=True, ocr="auto",
            progress=None):
    merged, _ = _convert_paragraphs(path, pages, math, drop_captions, ocr, progress)
    return "\n\n".join(merged)


def diagnose_pages(path, pages=None, ocr="auto", dpi=150):
    """For the triage view: classify every line on the given pages as kept
    (body text that reaches the reader) or dropped, and why - furniture,
    a caption, OCR junk, rotated text, or an isolated floating label -
    without changing anything about how the real pipeline decides.

    Reuses ordered_lines()/group_paragraphs() with a tag() callback rather
    than re-implementing their filtering logic, so this view can never
    silently drift from what actually happens when a document is opened
    for real - the whole point is to trust what it shows.

    Returns one dict per page: its 0-based index, the rendered image size,
    a base64 PNG for display, and every line's box/text/category, scaled
    to that image so the caller can draw boxes directly on it."""
    doc = pymupdf.open(path)
    idxs = parse_pages(pages, doc.page_count)
    scan_idx = [i for i in idxs if needs_ocr(doc[i], ocr)]
    pre = ocr_many(path, scan_idx) if scan_idx else {}

    _, furniture = _furniture_for(doc, idxs, ocr, pre)

    pages_out = []
    for i in idxs:
        dropped = []   # [(line, category)] - decided inside ordered_lines
        survivors = ordered_lines(doc[i], ocr, pre,
                                   tag=lambda l, c: dropped.append((l, c)))
        kept = {}      # id(line) -> category - decided inside group_paragraphs
        group_paragraphs(survivors, furniture, True,
                          tag=lambda l, c: kept.__setitem__(id(l), c))

        page = doc[i]
        pix = page.get_pixmap(dpi=dpi)
        scale = dpi / 72.0

        def render(l, category):
            return {"x0": l["x0"] * scale, "y0": l["y0"] * scale,
                    "x1": l["x1"] * scale, "y1": l["y1"] * scale,
                    "text": l["text"], "category": category}

        out_lines = [render(l, c) for l, c in dropped]
        out_lines += [render(l, kept.get(id(l), "body")) for l in survivors]

        pages_out.append({
            "page": i, "width": pix.width, "height": pix.height,
            "image_b64": base64.b64encode(pix.tobytes("png")).decode("ascii"),
            "lines": out_lines,
        })
    doc.close()
    return pages_out


def main():
    ap = argparse.ArgumentParser(
        description="Turn a textbook PDF into text that reads well aloud.")
    ap.add_argument("pdf")
    ap.add_argument("-o", "--out")
    ap.add_argument("--pages", help="1-indexed range, e.g. 184-212 or 3,5,7-9")
    ap.add_argument("--math", choices=["speak", "skip", "raw"], default="speak",
                    help="speak: equations in words (default); "
                         "skip: just 'Equation 4.8'; raw: leave symbols alone")
    ap.add_argument("--keep-captions", action="store_true")
    ap.add_argument("--ocr", choices=["auto", "never", "always"], default="auto",
                    help="auto: OCR pages that carry no text layer (default); "
                         "always: OCR every page; never: fail loudly on a scan")
    ap.add_argument("--chunks", help="also write numbered chunks here")
    ap.add_argument("--limit", type=int, default=4200)
    a = ap.parse_args()

    text = convert(a.pdf, a.pages, a.math, not a.keep_captions, a.ocr,
                   progress=lambda done, total: print(
                       f"  read page {done} of {total}", end="\r", file=sys.stderr))
    out = a.out or os.path.join(
        WORKDIR, os.path.splitext(os.path.basename(a.pdf))[0] + ".txt")
    with open(out, "w", encoding="utf-8") as f:
        f.write(text)

    if not text.strip():
        print("No text extracted. If this PDF is a scan, that is expected with "
              "--ocr never;\nrerun with --ocr always to read the pages as images.")
        return
    w = len(text.split())
    print(f"{out}: {len(text):,} chars, ~{w:,} words, "
          f"~{w/150/60:.1f} h at 150 wpm, ~{w/150/60/2:.1f} h at 2x")

    if a.chunks:
        os.makedirs(a.chunks, exist_ok=True)
        parts = chunk(text, a.limit)
        for i, c in enumerate(parts, 1):
            with open(os.path.join(a.chunks, f"{i:04d}.txt"), "w",
                      encoding="utf-8") as f:
                f.write(c)
        print(f"{len(parts)} chunks -> {a.chunks}/")


if __name__ == "__main__":
    main()
