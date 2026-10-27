"""
Chapter detection for PDFs.

Two techniques were tried, and one clearly won:

  * **Running heads** ("the method that actually worked"). Most textbooks
    print "82 * Chapter 3 / Structures of Metals" across the top of every
    body page. Crop the top ~9%, read it, and the chapter number *and* a
    clean title fall out of one short line in a fixed position. It is
    self-verifying, because the same head repeats on every page of the
    chapter, and it needs no language model at all.
  * **The contents page** parsed by a language model. Tried first, and it is
    the harder problem - two-column layouts, wrapped titles, "1.3" losing its
    dot and becoming a phantom chapter 13.

So running heads go first and the contents page is the fallback:

    detect(path) -> [{"title", "page", "block"?, "number", "head_page",
                      "printed_page", "method", "confidence"}, ...]

Every condition this feature has to meet is enforced here:
  - Off the critical path: always runs in a background thread (see
    talkify.py), and returns [] rather than raising on absolutely anything
    that goes wrong - no Tesseract, no running heads, no Ollama, no contents
    page, a model that answered nonsense, page numbers that don't check out.
  - The model proposes, code verifies: chapter pages must be monotonically
    increasing, inside the actual page count, and not all bunched onto one
    page. The whole result is discarded if verification fails - never a
    partially-trusted list.
  - One contents page per call - never the whole front matter, which is
    exactly the size that triggers Ollama's silent truncation.
  - The running-head path is 100% local (PDF text layer, then Tesseract).
    Only the contents-page fallback needs a provider, and it still refuses
    to run against a cloud one.
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass

import ai
import readaloud

_CONTENTS_HEADING = re.compile(r"^\s*(table of )?contents\s*$", re.I)
# A page number at the very start or end of a page's extracted text - either
# alone on its own line, or tacked onto the end of a running head like
# "3.4 Metallic Crystal Structures  *  51" (very common: number, title and
# page all share one header line rather than the number sitting alone).
_TRAILING_NUM = re.compile(r"(\d{1,4})\s*$")

_SYSTEM = (
    "You read the OCR'd or extracted text of ONE page from a book's table "
    "of contents and turn it into a JSON chapter list. Reply with ONLY a "
    "JSON array - no prose, no markdown fences, nothing before or after it. "
    'Each element: {"title": "<title, no leading number>", "page": <printed '
    "page number as a plain integer>}. Only include rows whose number is a "
    "whole integer with no decimal point (a chapter, like '3 Phase "
    "Diagrams'), never a decimal-numbered row (a section, like '3.4 The "
    "Phase Rule') - skip every decimal-numbered row entirely. Skip rows "
    "with no page number. Keep the array in the same top-to-bottom order "
    "the rows appear on the page. If this text is not actually a table of "
    "contents, reply with []."
)


def _find_contents_page(doc, first_page=0, last_page=None):
    """A contents page is short, near the front, and either says "Contents"
    at the top or is dense with lines ending in a page number. Checked
    against the PDF's own text layer only - cheap, and a scanned contents
    page still gets picked up here even without OCR, since the density
    check works on OCR'd text too once chapters.detect() OCRs it below.

    "Near the front" means near the front of what the reader opened. A
    selection that starts after the front matter simply has no contents page
    in it, and finds none - which is the right answer, not a reason to go
    reading pages the document does not contain."""
    if last_page is None:
        last_page = doc.page_count - 1
    for i in range(first_page, min(first_page + 20, last_page + 1)):
        text = doc[i].get_text()
        if not text.strip():
            continue
        lines = text.strip().splitlines()
        head = lines[0].strip() if lines else ""
        dotted = sum(1 for l in lines if re.search(r"\.{2,}\s*\d+\s*$", l))
        trailing_num = sum(1 for l in lines if re.search(r"\s\d{1,4}\s*$", l))
        if _CONTENTS_HEADING.match(head) or dotted >= 3 or trailing_num >= 5:
            return i
    return None


def _calibrate_offset(doc, samples=14, first_page=0, last_page=None):
    """pdf_page_index - printed_page_number, taken as the mode across
    several sampled body pages (their own text layer, no OCR - this only
    needs to be fast). Take the mode, not the mean - one OCR/
    extraction misread produces a wild outlier that would poison an average.
    Returns None (meaning "give up, don't guess") if nothing reliable
    turns up - a wrong offset is worse than no chapter list at all."""
    offsets = Counter()
    if last_page is None:
        last_page = doc.page_count - 1
    n = last_page - first_page + 1
    step = max(1, n // samples)
    for i in range(first_page, last_page + 1, step):
        text = doc[i].get_text()
        if not text.strip():
            continue
        lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
        if not lines:
            continue
        for line in (lines[0], lines[-1]):
            m = _TRAILING_NUM.search(line)
            if m:
                num = int(m.group(1))
                if 0 < num < n + 300:
                    offsets[i - num] += 1
    if not offsets:
        return None
    (offset, count), = offsets.most_common(1)
    return offset if count >= 3 else None


def _contents_page_text(doc, idx):
    page = doc[idx]
    text = page.get_text()
    if text.strip():
        return text
    lines = readaloud.ocr_lines(page)
    return "\n".join(l["text"] for l in lines)


def _parse_model_json(raw):
    match = re.search(r"\[.*\]", raw, re.S)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return None
    if not isinstance(data, list):
        return None
    out = []
    for it in data:
        if not isinstance(it, dict):
            continue
        title, page = it.get("title"), it.get("page")
        if not title or not isinstance(page, (int, float)):
            continue
        out.append({"title": str(title).strip(), "page": int(page)})
    return out


def _verify(pages, page_count):
    if len(pages) < 2:
        return False
    if any(p < 0 or p >= page_count for p in pages):
        return False
    if pages != sorted(pages) or len(set(pages)) != len(pages):
        return False
    if pages[-1] - pages[0] < len(pages) - 1:      # not all crammed onto ~1 page
        return False
    return True


def _detect_toc(path, cancelled=None, pages=None):
    """The contents-page fallback. Needs a *local* language model; with only a
    cloud key configured it declines rather than shipping a private book's
    front matter to a metered API that the reader never asked for.

    Chapter detection for every document shares one serial worker (see
    _SerialAnalysis in talkify.py), so a slow ai.ask() here holds up the next
    document's chapters too - closing/reopening a book can't cancel a request
    already in flight, but checking `cancelled` right before making it means
    the common case (the document moved on while we were still reading the
    PDF, not while the network call was already running) doesn't spend its
    full 150s timeout for nothing."""
    info = ai.provider_info()
    if not info.get("available") or not info.get("local"):
        # Background chapter detection runs automatically on PDF open without
        # user disclosure. Private document text must never
        # leave the machine to a metered cloud API without explicit user action.
        return []
    import pymupdf
    doc = pymupdf.open(path)
    try:
        page_count = doc.page_count
        first_page, last_page = _page_window(page_count, pages)
        offset = _calibrate_offset(doc, first_page=first_page,
                                   last_page=last_page)
        if offset is None:
            return []
        idx = _find_contents_page(doc, first_page, last_page)
        if idx is None:
            return []
        rows = _contents_page_text(doc, idx)
    finally:
        doc.close()

    if not rows.strip():
        return []
    if cancelled and cancelled():
        return []
    try:
        answer, _ = ai.ask(_SYSTEM, rows, temperature=0.2, timeout=150)
    except Exception:
        return []

    items = _parse_model_json(answer)
    if not items:
        return []

    chapters = [{"title": it["title"], "page": it["page"] + offset,
                 "method": "toc", "confidence": "estimated"} for it in items]
    # A contents page lists the whole book. A reader who opened part of it has
    # no use for a mark pointing at a page they do not have, and following one
    # would seek into nothing.
    chosen = _page_set(page_count, pages)
    chapters = [c for c in chapters
                if first_page <= c["page"] <= last_page
                and (chosen is None or c["page"] in chosen)]
    if not _verify([c["page"] for c in chapters], page_count):
        return []
    return chapters


# ================================================== running heads (the good one)

# Fraction of the page height cropped for the head. Measured
# figure: the top ~9% is enough on a textbook page and small enough that OCR
# of one strip is fast enough to scan a whole book.
TOP_FRACTION = 0.09

# Odd stride: it samples both PDF parities as it walks, so
# nothing has to assume "even PDF page" means "physical verso" after however
# many pages of front matter this particular book happens to have.
COARSE_STRIDE = 7

# How far either side of a binary-search midpoint to look for *some* head. A
# missing head is not evidence of anything (chapter openers are often full-page
# photographs, and some pages are figures), so a blank probe steps outward
# rather than moving a bound.
PROBE_RADIUS = 3

# A chapter opener is usually a page or two before the first page that
# actually carries a running head - watch for full-page
# opening photographs. Conservative and bounded; the head page is kept
# alongside so nothing pretends to a precision it does not have.
OPENER_MARGIN = 2

MIN_CHAPTERS = 2
MIN_CHAPTER_PAGES = 3        # two "chapters" two pages apart are a misread
MIN_OFFSET_SAMPLES = 3       # the mode, and only with real support
MAX_REFINE_PROBES = 40       # per boundary; caps the cost of an unreadable stretch

# The separator between the printed page number and the chapter, and the
# one between the chapter number and its title, are both whatever the book
# set and whatever the reader of the page could recover. A bullet is the
# usual first one, and it arrives as: a real bullet, a middle dot (which is
# what a Helvetica-substituted bullet extracts as), a ring, a circled dot,
# a star, or - when the font has no usable ToUnicode entry at all - a
# replacement character. None of that is evidence about whether the line is
# a running head, so all of it is accepted.
HEAD = re.compile(
    r"^\s*(?:(?P<printed>\d{1,4})\s*"
    r"[•·°©∙+*�\-]\s*)?"
    r"chapter\s*(?P<number>\d{1,2})\s*(?:[/|–—\-:])\s*"
    r"(?P<title>[A-Za-z][A-Za-z ,'\-]{3,60})\s*$",
    re.I,
)


@dataclass(frozen=True)
class HeadHit:
    page: int                 # 0-based PDF page index the head was read from
    number: int               # the chapter number it names
    title: str
    printed: int | None       # the printed page number, when the head carries one


def normalize_head(text):
    """Whitespace, control characters and the curly apostrophes OCR emits,
    flattened - but nothing more. Deliberately *not* an OCR corrector:
    "improving" a misread into a chapter match is how a hallucinated chapter
    gets manufactured out of a page of body text."""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("’", "'").replace("‘", "'")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", " ", text)
    return re.sub(r"[ \t]+", " ", text).strip()


def parse_head(text, page):
    """A HeadHit if this line is a running head, else None. Anchored at both
    ends on purpose: a sentence that happens to contain "Chapter 3 / results"
    is prose, and prose is exactly what must not become a chapter."""
    for line in normalize_head(text).splitlines():
        m = HEAD.match(line.strip())
        if not m:
            continue
        try:
            number = int(m.group("number"))
        except (TypeError, ValueError):
            continue
        if number < 1:
            continue
        printed = m.group("printed")
        title = re.sub(r"\s+", " ", m.group("title")).strip(" -,")
        if not title:
            continue
        return HeadHit(page, number, title,
                       int(printed) if printed is not None else None)
    return None


def mode_offset(pairs):
    """pdf_page_index - printed_page_number, as the mode over (pdf, printed)
    pairs. The mode, never the mean - one OCR misread of a page
    number produces a wild outlier that would drag an average with it. On the
    measured book this was a clean, consistent 19.

    None unless there is real support for the answer: at least three
    observations agreeing, and a strong plurality of everything seen."""
    counts = Counter(pdf - printed for pdf, printed in pairs
                     if printed is not None)
    if not counts:
        return None
    offset, count = counts.most_common(1)[0]
    total = sum(counts.values())
    if count < MIN_OFFSET_SAMPLES or count * 2 <= total:
        return None
    return offset


# ------------------------------------------------------------------- probing

class _Prober:
    """Reads one page's top strip, and remembers what it found - including
    that it found nothing.

    Caching the misses is the point. Binary search returns to the same pages
    repeatedly, and a page with no head is the common case; without this, a
    980-page scan would OCR the same blank strip a dozen times. `calls` is
    what smoke.py asserts on."""

    def __init__(self, doc=None, cache_dir=None, read=None, allowed=None):
        self.doc = doc
        self.cache_dir = cache_dir
        self.read = read                  # injected by tests; real reader below
        self.allowed = allowed            # pages this document actually has
        self.seen = {}
        self.calls = 0
        self.ocr_broken = False           # no Tesseract: stop trying, stay quiet

    def __call__(self, page):
        if page in self.seen:
            return self.seen[page]
        if self.allowed is not None and page not in self.allowed:
            # A page this document does not contain reads as silence, which is
            # exactly how the search already treats a page with no head on it.
            # Doing it here rather than at the bounds is what makes a
            # non-contiguous selection - "pages 20, 400-410" - safe: the
            # binary search between two selected pages can no longer OCR the
            # hundreds of unselected ones in between.
            self.seen[page] = None
            return None
        hit = None
        try:
            text = self.read(page) if self.read else self._read_page(page)
            self.calls += 1
            hit = parse_head(text, page)
        except Exception:
            hit = None
        self.seen[page] = hit
        return hit

    # ---- the real reader
    def _cache_path(self, page):
        if not self.cache_dir:
            return None
        return os.path.join(self.cache_dir, f"{page:05d}.json")

    def _read_page(self, page):
        path = self._cache_path(page)
        if path:
            try:
                with open(path, encoding="utf-8") as f:
                    return json.load(f).get("text", "")
            except (OSError, ValueError):
                pass                      # absent or corrupt: just read it again
        text = self._strip_text(page)
        if path:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    json.dump({"text": text}, f)
            except OSError:
                pass                      # an unwritable cache is only slower
        return text

    def _strip_text(self, page):
        import pymupdf
        p = self.doc[page]
        r = p.rect
        clip = pymupdf.Rect(r.x0, r.y0, r.x1, r.y0 + r.height * TOP_FRACTION)
        text = p.get_text("text", clip=clip)
        if text.strip():
            return text
        if self.ocr_broken:
            return ""
        return self._ocr_strip(p, clip)

    def _ocr_strip(self, p, clip):
        """--psm 7: one line of text. That is what a running head is, and
        telling Tesseract so is most of why this works on a strip when
        full-page OCR of the same book is slow and noisy."""
        try:
            pytesseract, Image = readaloud.setup_ocr()
        except Exception:
            self.ocr_broken = True        # missing Tesseract is not an error here
            return ""
        try:
            dpi = readaloud.OCR_DPI
            native = readaloud._native_dpi(p)
            if native:
                dpi = max(72, min(dpi, round(native)))
            pix = p.get_pixmap(clip=clip, dpi=dpi)
            # Straight over the pixmap's own bytes - see readaloud.pixmap_image.
            # Coarse-scanning a 980-page book probes hundreds of these strips,
            # and a PNG round-trip per probe is pure overhead for pixels that
            # come out identical either way.
            img = readaloud.pixmap_image(Image, pix)
            return pytesseract.image_to_string(img, config="--psm 7", timeout=20)
        except Exception:
            return ""


def _probe_near(probe, mid, low, high, radius=PROBE_RADIUS):
    """Some head near `mid`, or None. A blank probe steps outward instead of
    concluding anything - a page with no head is silence, not evidence."""
    for d in range(radius + 1):
        for page in ([mid] if d == 0 else [mid - d, mid + d]):
            if low <= page <= high:
                hit = probe(page)
                if hit is not None:
                    return hit
    return None


def _first_head_page(probe, number, low, high, budget=MAX_REFINE_PROBES):
    """The earliest page in (low, high] confirmed to carry chapter `number`.

    Binary search, with one rule: only a verified head moves a
    bound. A head for an earlier chapter rules out everything at or below it;
    a head for a later one rules out everything at or above it; `number`
    itself pulls the answer down and keeps looking below.

    A blank probe is the interesting case, and it is why this is a worklist
    rather than a plain loop. No head near the midpoint says nothing at all
    about which chapter those pages belong to - a run of full-page plates
    reads exactly like the front matter before chapter one - so neither bound
    may move. Both halves are pushed instead, the earlier one examined first,
    and `best` prunes anything that can no longer improve the answer. `budget`
    stops a book with a large unreadable stretch from turning this into a
    page-by-page walk; probes are cached, so a bounded search that revisits
    is cheap."""
    best = high
    work = [(low + 1, high - 1)]
    while work and budget > 0:
        lo, hi = work.pop()
        if lo > hi or lo >= best:
            continue                       # nothing here could beat what we have
        mid = (lo + hi) // 2
        hit = _probe_near(probe, mid, lo, hi)
        budget -= 1
        if hit is None:
            # Silence. Split, and look in the earlier half first - it is the
            # half that could still improve the answer.
            work.append((mid + PROBE_RADIUS + 1, hi))
            work.append((lo, mid - PROBE_RADIUS - 1))
            continue
        if hit.number == number:
            best = min(best, hit.page)
            work.append((lo, hit.page - 1))
        elif hit.number > number:
            work.append((lo, hit.page - 1))
        else:
            work.append((hit.page + 1, hi))
    return best


def _collect(probe, first_page, last_page, stride=COARSE_STRIDE, cancelled=None,
             pages=None):
    """The coarse pass. Every `stride`th page, read the strip; keep every head.

    Walks the pages the reader actually opened, not the size of the file.
    Opening pages 20-100 of a 980-page scan and then probing all 980 of them
    is minutes of Tesseract for chapter marks that could only ever apply to
    eighty-one pages - and every `stride`th *selected* page rather than every
    `stride`th number keeps the sample proportional to the document even when
    the selection has a gap in it."""
    walk = sorted(pages) if pages is not None else range(first_page, last_page + 1)
    hits = []
    for page in walk[::stride]:
        if cancelled and cancelled():
            return []
        hit = probe(page)
        if hit is not None:
            hits.append(hit)
    return hits


def _verify_running_heads(chapters, page_count):
    """All-or-nothing. A single OCR hallucination must never be
    able to produce a partial chapter list, so anything that fails here
    discards the whole result rather than the offending row."""
    if len(chapters) < MIN_CHAPTERS:
        return False
    numbers = [c["number"] for c in chapters]
    pages = [c["page"] for c in chapters]
    heads = [c["head_page"] for c in chapters]
    if numbers != sorted(numbers) or len(set(numbers)) != len(numbers):
        return False
    if pages != sorted(pages) or len(set(pages)) != len(pages):
        return False
    if any(p < 0 or p >= page_count for p in pages + heads):
        return False
    # Spacing is checked on the *head* pages, not the published starts. The
    # starts have the opener margin backed out of them and are clamped to stay
    # after the previous chapter, which would manufacture the very gap this is
    # supposed to be testing for - two heads one page apart are a misread, and
    # spreading them three pages apart before checking would hide that.
    if heads != sorted(heads) or len(set(heads)) != len(heads):
        return False
    if any(b - a < MIN_CHAPTER_PAGES for a, b in zip(heads, heads[1:])):
        return False
    if any(not c.get("title") for c in chapters):
        return False
    # Repeated compatible evidence, for *every* chapter, not for most of them.
    # A running head repeats on every page of its chapter, so one sighting is
    # not a chapter - it is a line of body text that happened to match, or an
    # exercise header, or an OCR misread. Half the list being well-supported
    # is no comfort when the other half is invented, and this result is
    # all-or-nothing by design: rejecting a real book whose last chapter was
    # only ever sampled once is the cheaper mistake.
    return all(c.get("observations", 0) >= 2 for c in chapters)


def _page_set(page_count, pages=None):
    """The pages this document actually contains, as a frozenset, or None for
    a document that is the whole file.

    Anything nonsensical reads as None - the whole book - rather than as an
    empty document that could never be scanned at all."""
    last = max(0, (page_count or 0) - 1)
    if not pages:
        return None
    try:
        kept = frozenset(int(p) for p in pages if 0 <= int(p) <= last)
    except (TypeError, ValueError):
        return None
    return kept or None


def _page_window(page_count, pages=None):
    """The inclusive first/last page to probe. The ends of the selection, or
    the ends of the book."""
    last = max(0, (page_count or 0) - 1)
    chosen = _page_set(page_count, pages)
    return (min(chosen), max(chosen)) if chosen else (0, last)


def _detect_running_heads(path, probe=None, page_count=None, cancelled=None,
                          pages=None):
    """The whole running-head pass: coarse scan, boundary refinement, modal
    offset, verification. Returns [] on anything at all going wrong - a
    missing file, no Tesseract, no heads, a sequence that fails to verify.

    `pages` is the exact set of PDF pages the reader opened. Every probe stays
    inside it - the coarse pass, and the binary search that refines each
    boundary - because a page outside it is one this document does not
    contain and no chapter mark can ever point at. A set rather than a span,
    because "pages 20, 400-410" is two pages and eleven, not three hundred
    and ninety-one.

    `probe` and `page_count` exist so the logic can be tested against a fake
    book, with no PDF, no Tesseract and no OCR at all."""
    doc = None
    try:
        if probe is None:
            import pymupdf
            if not path or not os.path.exists(path):
                return []
            doc = pymupdf.open(path)
            page_count = doc.page_count
            cache = os.path.join(readaloud.cache_dir_for(path), "running-heads")
            probe = _Prober(doc=doc, cache_dir=cache,
                            allowed=_page_set(doc.page_count, pages))
        if not page_count:
            return []

        chosen = _page_set(page_count, pages)
        if chosen is not None and getattr(probe, "allowed", None) is None:
            # An injected probe (a test, or a caller that built its own) still
            # has to be held to the selection.
            probe.allowed = chosen
        first_page, last_page = _page_window(page_count, pages)
        hits = _collect(probe, first_page, last_page, cancelled=cancelled,
                        pages=chosen)
        if not hits:
            return []

        first_seen = {}
        for h in hits:
            if h.number not in first_seen or h.page < first_seen[h.number].page:
                first_seen[h.number] = h

        # The lower bound of the refinement, too. Starting it at -1 would send
        # the first chapter's binary search down through every page before the
        # window - which on a selection that begins at page 20 of a scanned
        # book is exactly the OCR this is here to avoid.
        starts, prev_page = [], first_page - 1
        for n in sorted(first_seen):
            if cancelled and cancelled():
                return []
            head_page = _first_head_page(probe, n, max(prev_page, first_page - 1),
                                         min(first_seen[n].page, last_page))
            starts.append((n, head_page))
            prev_page = head_page

        # Every page ever probed counts as evidence, refinement included -
        # which is what lets a short chapter reach two sightings at all.
        every = [h for h in getattr(probe, "seen", {}).values() if h is not None]
        if not every:
            every = hits
        observed, titles = Counter(), {}
        for h in every:
            observed[h.number] += 1
            titles.setdefault(h.number, Counter())[h.title] += 1
        offset = mode_offset([(h.page, h.printed) for h in every
                              if h.printed is not None])

        def printed_for(start):
            """The printed page number of a chapter start, or None. A number
            below 1 is not a page number - it means the estimated start ran
            back past the book's own page 1 - and publishing it would be worse
            than publishing nothing."""
            if offset is None:
                return None
            value = start - offset
            return value if value >= 1 else None

        # The floor starts at the first page the reader actually has. The
        # opener margin walks an estimated start a page or two back from the
        # head that proved it, and on a document opened as a page range that
        # would put chapter one's start before the beginning of the document -
        # a mark pointing at a page it does not contain.
        chapters, floor = [], first_page
        for n, head_page in starts:
            winning_title, title_obs = titles[n].most_common(1)[0]
            # head_page - OPENER_MARGIN is the estimate; the floor only ever
            # matters if verification is about to reject this list anyway.
            start = max(floor, head_page - OPENER_MARGIN)
            chapters.append({
                "title": winning_title,
                "number": n,
                "page": start,
                "head_page": head_page,
                # Paired with "page", the start this actually publishes - not
                # with head_page, which is one or two pages further in. A UI
                # showing "chapter 3 begins on printed page 81" must not send
                # the reader to 83.
                "printed_page": printed_for(start),
                "method": "running_head",
                "confidence": "estimated" if start != head_page else "verified",
                "observations": title_obs,
            })
            floor = start + 1

        if not _verify_running_heads(chapters, page_count):
            return []
        for c in chapters:
            c.pop("observations", None)
        return chapters
    except Exception:
        return []
    finally:
        if doc is not None:
            try:
                doc.close()
            except Exception:
                pass


def detect(path, cancelled=None, pages=None):
    """Running heads first - local, fast, and the method that
    actually worked in testing. The contents-page/LLM route is only the
    fallback for books that print no heads, and it is the only half that
    needs a language model at all.

    `pages` is the set of PDF pages the document actually holds. A reader who
    opened pages 20-100 opened a book of eighty-one pages, and nothing here
    may read outside it."""
    heads = _detect_running_heads(path, cancelled=cancelled, pages=pages)
    if heads:
        return heads
    if cancelled and cancelled():
        return []
    try:
        return _detect_toc(path, cancelled=cancelled, pages=pages)
    except Exception:
        # Same contract as the running-head half: this is called from a
        # background thread and returns [] for everything, including a PDF
        # that will not open and a provider that dies mid-request.
        return []
