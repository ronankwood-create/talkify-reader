"""
Show the figure when the text references it.

This is an optional enhancement, and it is built to the project's rule
rather than to what would be easiest: *if detection fails, returns nothing, or
throws, the reader must behave exactly as if the feature did not exist.* So
nothing here runs on the critical path. `build_index()` is called from a
background daemon thread after the document is already open and readable, it
writes its results to a sidecar cache, and every failure mode - no PyMuPDF, no
text layer, no captions, no images, a caption that cannot be matched to a
placement, a corrupt cache - ends the same way: an empty index, no assets, no
bindings, and a reader that never learns anything was attempted.

    build_index(pdf_path, blocks, block_pages, cache_root) -> manifest dict

The manifest is the whole public contract:

    {"state": "ready" | "none",
     "revision": "figure-v1:<source fingerprint>",
     "items": [{"id", "kind", "label", "page", "caption", "width", "height",
                "confidence"}],
     "refs":  {"<block index>": [{"figure_id", "label", "page",
                                  "start_word", "end_word"}]}}

Two decisions worth keeping:

  * **Captions are decided by readaloud.caption_starts(), not by a second,
    looser detector.** That function is the same "isolated visual block" rule
    group_paragraphs() uses to drop captions from the narration. If the two
    ever disagreed, the reader would either hear a caption read aloud or see a
    figure bound to a sentence that merely mentions one.

  * **Word indexes, not character offsets.** Python string indexes and the
    browser's UTF-16 indexes part company on astral characters, and the player
    already thinks in words - `{w, s, e}` timings, one highlight per word - so
    a reference that says "reveal at word 3" needs no translation at all.

Only figures that some sentence actually refers to are rendered. A textbook
has hundreds of images and a handful of them get mentioned; extracting all of
them would cost minutes of CPU to produce a cache nobody looks at.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, asdict

INDEX_VERSION = "figure-v1"

# Rendering resolution for the cropped placement. 150 DPI matches what
# /api/page already serves for the split view, so a figure and the page it
# came from look like they belong to the same document.
RENDER_DPI = 150

# Geometry limits, all in PDF points / page-area fractions.
MIN_SIDE_PT = 36.0          # smaller than this is a rule, a logo, or a bullet
MIN_AREA_FRACTION = 0.015   # ...and so is anything under ~1.5% of the page
MAX_AREA_FRACTION = 0.80    # bigger than this is the page itself: a scan
MAX_GAP_ABOVE_PT = 110.0    # a figure sits directly above its caption
MAX_GAP_BELOW_PT = 70.0     # ...or, less often, directly below it
MIN_OVERLAP = 0.35          # share of the narrower box's width they must share
AMBIGUITY_MARGIN_PT = 8.0   # two candidates this close is a tie; ties are dropped


# ------------------------------------------------------------------ labels

# Canonical names for the label kinds readaloud.CAPTION_START recognizes, so
# "Fig. 3.20" and "Figure 3.20" resolve to one key and a sentence can name
# either. Anything not listed here is simply not a figure as far as this
# module is concerned.
_KINDS = {
    "fig": "figure", "figure": "figure",
    "tbl": "table", "table": "table",
    "plate": "plate", "chart": "chart", "exhibit": "exhibit",
    "scheme": "scheme", "photo": "photo", "diagram": "diagram",
    "listing": "listing", "algorithm": "algorithm",
}

_NUMBER = r"\d+(?:[A-Za-z]|\.\d+|[-–—:][A-Za-z0-9]+)*"

_LABEL = re.compile(
    r"^\s*(?P<kind>fig(?:ure)?|tbl|table|plate|chart|exhibit|scheme|photo|"
    r"diagram|listing|algorithm)\.?\s*(?P<number>" + _NUMBER + r")\b",
    re.I,
)

# Narrative references, deliberately conservative: the sentence has to point
# at the figure ("as shown in Figure 3.20", "see Table 2A"), not merely
# contain its name. A bare "Figure 3.20" at the start of a line is a caption,
# which is a different thing and is handled above. Ranges ("Figs. 2-4") are
# not supported and are not meant to be - a range has no single answer to
# "which image do I show", and guessing is how this feature would start
# getting in the way.
REF = re.compile(
    r"\b(?:(?:as\s+)?shown\s+in|see|refer(?:ring)?\s+to|from|in)\s+"
    r"(?P<kind>fig(?:ure)?|tbl\.?|table|plate|chart|exhibit|scheme|photo|diagram)"
    r"\.?\s*(?P<number>" + _NUMBER + r")\b",
    re.I,
)


def _canonical(kind, number):
    """('Fig.', '3.20') -> 'figure:3.20'. None if the kind isn't one we index."""
    base = _KINDS.get(kind.rstrip(".").lower())
    if base is None:
        return None
    return f"{base}:{number.lower()}"


def parse_caption_label(text):
    """The canonical key and the printed label of a caption's opening text.

        parse_caption_label("Figure 3.20 Crystal lattice")
            -> ("figure:3.20", "Figure 3.20")

    None when the text does not open with a label at all - which is most
    lines, and is not an error."""
    m = _LABEL.match(text or "")
    if not m:
        return None
    key = _canonical(m.group("kind"), m.group("number"))
    if key is None:
        return None
    label = re.sub(r"\s+", " ", text[m.start("kind"):m.end("number")]).strip()
    return key, label


# ------------------------------------------------------------- data shapes

@dataclass(frozen=True)
class FigureItem:
    """One extracted figure. `rect` is internal - it names a region of a file
    on this machine, so it never crosses the API boundary."""
    id: str
    kind: str
    label_key: str
    label: str
    page: int
    caption: str = ""
    rect: tuple = (0.0, 0.0, 0.0, 0.0)
    asset_sha256: str = ""
    mime: str = "image/png"
    width: int = 0
    height: int = 0
    confidence: str = "medium"

    def public(self):
        return {"id": self.id, "kind": self.kind, "label": self.label,
                "page": self.page, "caption": self.caption,
                "width": self.width, "height": self.height,
                "confidence": self.confidence}


@dataclass(frozen=True)
class FigureRef:
    """Where in a block a figure gets mentioned, in word indexes."""
    figure_id: str
    label: str
    page: int
    start_word: int
    end_word: int              # exclusive

    def public(self):
        return asdict(self)


@dataclass(frozen=True)
class RawRef:
    """A regex hit, before it is known to name anything that exists."""
    label_key: str
    label: str
    block: int
    page: int | None
    start_word: int
    end_word: int


# --------------------------------------------------------- reference binding

def _word_spans(text):
    """(start, end) character span of every whitespace-separated token, in the
    same order and count as text.split() - which is exactly the word list the
    timings and the highlight are indexed by."""
    return [(m.start(), m.end()) for m in re.finditer(r"\S+", text)]


def _word_at(spans, pos):
    for i, (a, b) in enumerate(spans):
        if a <= pos < b:
            return i
    return None


def _by_label(items):
    """{label_key: [item, ...]}, so resolving a reference is a lookup instead
    of a walk of every figure in the book."""
    out = {}
    for it in items:
        out.setdefault(it.label_key, []).append(it)
    return out


def resolve_reference(raw, items):
    """Which extracted figure a narrative reference means, or None.

    One candidate is the easy case. Several - a book that numbers its figures
    per chapter and has three "Figure 1" - is only resolvable if the
    referencing block's own page makes exactly one of them nearest. A tie, or
    a block whose page is unknown, yields None: no preview at all is much
    better than confidently showing the wrong figure.

    `items` may be the flat list of figures or the {label_key: [...]} index
    _by_label() builds from it; find_references passes the index so a thousand
    blocks do not each re-scan a thousand figures."""
    if isinstance(items, dict):
        candidates = items.get(raw.label_key) or []
    else:
        candidates = [it for it in items if it.label_key == raw.label_key]
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    if raw.page is None:
        return None
    scored = sorted(candidates, key=lambda it: (abs(it.page - raw.page), it.page))
    if abs(scored[0].page - raw.page) == abs(scored[1].page - raw.page):
        return None
    return scored[0]


def find_references(blocks, block_pages, items):
    """{block index: [FigureRef, ...]} for every block that points at a figure
    this index actually holds. Blocks with no reference are absent rather than
    present-and-empty, which keeps the manifest small on a book where two
    paragraphs in a thousand mention a figure."""
    out = {}
    by_label = _by_label(items)
    n_pages = len(block_pages) if block_pages is not None else 0
    for bi, text in enumerate(blocks or []):
        # Almost no paragraph mentions a figure, and the ones that do not have
        # nothing for the rest of this loop to do. Ask the cheap question
        # first: tokenizing every block of a textbook to find word offsets
        # nobody ends up using is most of the cost of building an index.
        if REF.search(text) is None:
            continue
        page = block_pages[bi] if bi < n_pages else None
        spans = _word_spans(text)
        refs = []
        for m in REF.finditer(text):
            key = _canonical(m.group("kind"), m.group("number"))
            if key is None:
                continue
            sw = _word_at(spans, m.start("kind"))
            ew = _word_at(spans, m.end("number") - 1)
            if sw is None or ew is None:
                continue
            label = re.sub(r"\s+", " ",
                           text[m.start("kind"):m.end("number")]).strip()
            raw = RawRef(key, label, bi, page, sw, ew + 1)
            item = resolve_reference(raw, by_label)
            if item is None:
                continue
            if any(r.figure_id == item.id for r in refs):
                continue                      # one preview per figure per block
            refs.append(FigureRef(item.id, item.label, item.page, sw, ew + 1))
        if refs:
            out[bi] = refs
    return out


# ---------------------------------------------------------------- the cache

def fingerprint(pdf_path, options=None):
    """Identifies the source, so a re-opened book reuses its index and an
    edited one does not. Cheap on purpose - stat, not a content hash: a
    scanned textbook is 291MB and this runs on every open."""
    st = os.stat(pdf_path)
    raw = (f"{os.path.abspath(pdf_path)}:{st.st_size}:{int(st.st_mtime)}:"
           f"{json.dumps(options or {}, sort_keys=True)}:{INDEX_VERSION}")
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_dir(cache_root, fp):
    return os.path.join(cache_root, fp, INDEX_VERSION)


def _atomic_write(path, data):
    tmp = f"{path}.{os.getpid()}.part"
    mode = "wb" if isinstance(data, bytes) else "w"
    kw = {} if isinstance(data, bytes) else {"encoding": "utf-8"}
    with open(tmp, mode, **kw) as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_manifest(cache_root, fp):
    """The stored manifest, or None. A corrupt or half-written cache reads as
    absent - it gets rebuilt, and nothing about reading is affected."""
    path = os.path.join(cache_dir(cache_root, fp), "manifest.json")
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "items" not in data or "refs" not in data:
        return None
    if data.get("revision") != f"{INDEX_VERSION}:{fp}":
        return None
    return data


def asset_path(cache_root, fp, figure_id):
    """Where one figure's PNG lives. The caller must have already checked
    `figure_id` against the manifest - this deliberately does no validation of
    its own, so there is exactly one place that decides what is servable."""
    return os.path.join(cache_dir(cache_root, fp), f"{figure_id}.png")


EMPTY = {"state": "none", "revision": None, "items": [], "refs": {}}


# ------------------------------------------------------------- the pipeline

def _caption_text(lines, idx):
    """A caption's opening line plus its wrapped continuations. Same rule
    group_paragraphs uses: a continuation starts mid-sentence, so anything
    beginning with a capital is the body text resuming."""
    parts = [lines[idx]["text"].strip()]
    for l in lines[idx + 1: idx + 5]:
        s = l["text"].strip()
        if not s or s[0].isupper():
            break
        parts.append(s)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def _placements(page):
    """Every raster image actually placed on the page, deduplicated, with the
    decorations and the full-page scan background thrown out.

    A scanned page *is* one big image, and showing it as "the figure" would
    mean popping up a photograph of the page the reader is already looking
    at - which is why the upper bound exists."""
    try:
        area = page.rect.width * page.rect.height
    except Exception:
        return []
    if area <= 0:
        return []
    seen, out = set(), []
    try:
        images = page.get_images(full=True)
    except Exception:
        return []
    for info in images:
        xref = info[0]
        try:
            rects = page.get_image_rects(xref)
        except Exception:
            continue
        for r in rects:
            key = (xref, round(r.x0, 1), round(r.y0, 1), round(r.x1, 1), round(r.y1, 1))
            if key in seen:
                continue
            seen.add(key)
            w, h = r.width, r.height
            if w < MIN_SIDE_PT or h < MIN_SIDE_PT:
                continue
            frac = (w * h) / area
            if frac < MIN_AREA_FRACTION or frac > MAX_AREA_FRACTION:
                continue
            out.append(r)
    return out


def _overlap_ratio(a0, a1, b0, b1):
    span = min(a1, b1) - max(a0, b0)
    narrower = min(a1 - a0, b1 - b0)
    if narrower <= 0:
        return 0.0
    return max(0.0, span) / narrower


class _Edge:
    """One geometrically plausible caption/rect pairing, and how good it is.

    ``key`` is compared lexicographically and lower is better:

        (gap, 1 - caption_coverage, 1 - rect_coverage,
         centre offset / rect width, -specificity)

    Distance first, because a figure sits next to its caption and nothing else
    about the geometry outranks that. Coverage next, because a rect that spans
    the caption *and* is spanned by it is a far better fit than one that
    merely contains it. Specificity last: it is the tie-break that says "this
    rect's own nearest caption is this one", which is what turns two
    independent local guesses into a page-level answer.
    """

    __slots__ = ("caption", "rect", "gap", "key")

    def __init__(self, caption, rect, gap, key):
        self.caption = caption
        self.rect = rect
        self.gap = gap
        self.key = key


def _geometry(caption_line, rect):
    """(gap, caption_coverage, rect_coverage, centre offset) or None.

    None means these two are not a candidate at all - too far apart, on the
    wrong side, or not overlapping horizontally enough to be about each other.
    """
    cx0, cy0, cx1, cy1 = (caption_line["x0"], caption_line["y0"],
                          caption_line["x1"], caption_line["y1"])
    span = min(cx1, rect.x1) - max(cx0, rect.x0)
    if span <= 0:
        return None
    caption_width = max(cx1 - cx0, 1e-6)
    rect_width = max(rect.x1 - rect.x0, 1e-6)
    if span / min(caption_width, rect_width) < MIN_OVERLAP:
        return None
    above = cy0 - rect.y1
    below = rect.y0 - cy1
    if 0 <= above <= MAX_GAP_ABOVE_PT:
        gap = above
    elif 0 <= below <= MAX_GAP_BELOW_PT:
        gap = below
    else:
        return None
    centre = abs((cx0 + cx1) / 2.0 - (rect.x0 + rect.x1) / 2.0)
    return (gap, span / caption_width, span / rect_width,
            centre / max(rect_width, 1.0))


def _ambiguous(candidates):
    """Whether the best of these is too close to the next to be trusted.

    A wrong figure shown confidently is worse than no figure, so the bar is
    "clearly better", not "better". Two candidates within AMBIGUITY_MARGIN_PT
    of each other are a tie unless the nearer one also fits the caption at
    least as well on both coverage measures and strictly better on one -
    which is the difference between a genuine tie and a rect that is a couple
    of points further away but obviously the right one.
    """
    if len(candidates) < 2:
        return False
    best, second = candidates[0], candidates[1]
    if second.gap - best.gap >= AMBIGUITY_MARGIN_PT:
        return False
    # key[1] and key[2] are 1 - coverage, so smaller is a better fit.
    at_least_as_good = (best.key[1] <= second.key[1]
                        and best.key[2] <= second.key[2])
    strictly_better = (best.key[1] < second.key[1]
                       or best.key[2] < second.key[2])
    return not (at_least_as_good and strictly_better)


def _match_placements(caption_lines, rects):
    """{caption index: rect} for every caption with an unambiguous placement.

    Page-level rather than one caption at a time, and that is the whole fix
    for debrief finding 2b. Matching greedily in reading order let a large
    unrelated rect - one that horizontally contains a short caption, so it
    passes the overlap gate whatever its width - win a caption it had nothing
    to do with, purely by sitting a point or two nearer than the real figure.
    Rejecting wide rects would close that and break the ordinary case of a
    short caption under a full-width figure, which is a real layout and a
    tested one.

    What actually resolves it is competition. The unrelated rect is somebody
    else's best match, and a mutual best pair - each is the other's first
    choice - is evidence neither one has on its own. Those are settled first;
    what remains is assigned by ascending cost, deterministically, and any
    caption whose best two candidates are too close is left with nothing.
    """
    edges = []
    for ci, line in enumerate(caption_lines):
        for ri, rect in enumerate(rects):
            measured = _geometry(line, rect)
            if measured is None:
                continue
            gap, caption_coverage, rect_coverage, centre = measured
            edges.append(_Edge(ci, ri, gap,
                               (gap, 1.0 - caption_coverage,
                                1.0 - rect_coverage, centre, 0.0)))
    if not edges:
        return {}

    # Specificity: for each rect, how far down its own preference list this
    # caption sits. Computed across the whole page, which is the only place
    # the information exists.
    for ri in {e.rect for e in edges}:
        mine = sorted((e for e in edges if e.rect == ri),
                      key=lambda e: (e.gap, e.rect, e.caption))
        for rank, edge in enumerate(mine):
            edge.key = edge.key[:4] + (-1.0 / (1.0 + rank),)

    result = {}
    open_captions = set(range(len(caption_lines)))
    open_rects = set(range(len(rects)))

    def live():
        return sorted((e for e in edges
                       if e.caption in open_captions and e.rect in open_rects),
                      key=lambda e: (e.key, e.caption, e.rect))

    def settle(caption, ranked):
        """Take the best candidate for this caption, or give up on it.

        Both directions have to be unambiguous. A caption with one plausible
        rect still gets nothing if that rect is a near-tie between this
        caption and another one still open - the figure would be shown under
        one of two captions on a coin toss, and a wrong figure shown
        confidently is worse than no figure at all."""
        if _ambiguous(ranked):
            open_captions.discard(caption)
            return
        best = ranked[0]
        contenders = sorted((e for e in edges
                             if e.rect == best.rect and e.caption in open_captions),
                            key=lambda e: (e.key, e.caption))
        if _ambiguous(contenders) or (contenders and contenders[0] is not best):
            # The rect is what is ambiguous, so it goes too. Refusing only
            # this caption would hand the same coin toss to the other one on
            # the next round, which is the wrong answer arrived at politely.
            open_captions.discard(caption)
            open_rects.discard(best.rect)
            return
        result[caption] = best.rect
        open_captions.discard(caption)
        open_rects.discard(best.rect)

    # Phase 1: mutual best pairs, repeatedly. Each round can free a rect that
    # makes another pair mutual, so this runs to a fixed point.
    while True:
        current = live()
        if not current:
            break
        by_caption, by_rect = {}, {}
        for edge in current:
            by_caption.setdefault(edge.caption, []).append(edge)
            by_rect.setdefault(edge.rect, []).append(edge)
        mutual = [edges_for[0] for caption, edges_for in by_caption.items()
                  if by_rect[edges_for[0].rect][0].caption == caption]
        if not mutual:
            break
        for edge in sorted(mutual, key=lambda e: (e.key, e.caption, e.rect)):
            if edge.caption not in open_captions or edge.rect not in open_rects:
                continue
            settle(edge.caption, [e for e in by_caption[edge.caption]
                                  if e.rect in open_rects])

    # Phase 2: whatever is left, cheapest first. Deterministic because the
    # ordering is total - cost, then caption index, then rect index.
    for edge in live():
        if edge.caption not in open_captions or edge.rect not in open_rects:
            continue
        settle(edge.caption, [e for e in edges
                              if e.caption == edge.caption
                              and e.rect in open_rects])
    return result


def _match_placement(caption_line, rects):
    """One caption against a list of rects. Kept because it is the shape a
    single-caption question has, and because page-level matching with one
    caption is exactly this."""
    matched = _match_placements([caption_line], list(rects))
    return rects[matched[0]] if 0 in matched else None


def _scan_page(page, page_index):
    """Provisional items for one page: caption, label and placement, but no
    rendered pixels yet. Nothing is rendered until it is known that some
    sentence actually refers to it."""
    import readaloud
    try:
        lines = readaloud.page_lines(page)
    except Exception:
        return []
    # Text layer only. A scanned page has no lines here, and its one full-page
    # raster is rejected below anyway - so a scan quietly yields no figures
    # rather than offering the reader a photograph of the page they are on.
    if not lines:
        return []
    rects = _placements(page)
    if not rects:
        return []
    # Every caption on the page is matched together, so a rect that two of
    # them could plausibly own goes to the one it actually belongs to rather
    # than to whichever was read first. See _match_placements.
    labelled = []
    for idx, line in readaloud.caption_starts(lines):
        parsed = parse_caption_label(line["text"].strip())
        if parsed:
            labelled.append((idx, line, parsed))
    if not labelled:
        return []
    matched = _match_placements([line for _idx, line, _p in labelled], rects)
    out = []
    for position, (idx, line, parsed) in enumerate(labelled):
        key, label = parsed
        if position not in matched:
            continue
        rect = rects[matched[position]]
        fid = "f-" + hashlib.sha256(
            f"{page_index}:{key}:{round(rect.x0, 1)}:{round(rect.y0, 1)}:"
            f"{round(rect.x1, 1)}:{round(rect.y1, 1)}".encode("utf-8")
        ).hexdigest()[:12]
        out.append(FigureItem(
            id=fid, kind=key.split(":")[0], label_key=key, label=label,
            page=page_index, caption=_caption_text(lines, idx),
            rect=(rect.x0, rect.y0, rect.x1, rect.y1),
            confidence="high"))
    return out


def build_index(pdf_path, blocks, block_pages, cache_root, *,
                options=None, cancelled=None):
    """Index one PDF's figures. Returns a manifest dict - never raises.

    `cancelled`, if given, is called between pages; returning true abandons
    the run and publishes nothing, which is what happens when the document is
    closed or reopened while this is still walking a long book."""
    try:
        return _build_index(pdf_path, blocks, block_pages, cache_root,
                            options=options, cancelled=cancelled)
    except Exception:
        return dict(EMPTY)


def _stop(cancelled):
    try:
        return bool(cancelled and cancelled())
    except Exception:
        return True


def _build_index(pdf_path, blocks, block_pages, cache_root, *,
                 options=None, cancelled=None):
    if not pdf_path or not os.path.exists(pdf_path):
        return dict(EMPTY)
    fp = fingerprint(pdf_path, options)
    revision = f"{INDEX_VERSION}:{fp}"
    cached = load_manifest(cache_root, fp)
    if cached is not None:
        return cached
    # No page provenance means no way to tell which page a sentence sits on,
    # which means duplicate figure numbers cannot be resolved. Rather than
    # bind them by guesswork, index nothing.
    if not block_pages:
        return dict(EMPTY)

    import pymupdf
    doc = pymupdf.open(pdf_path)
    items = []
    try:
        valid_pages = [p for p in block_pages
                       if isinstance(p, int) and 0 <= p < doc.page_count]
        if not valid_pages:
            return dict(EMPTY)
        # Scan all pages across the document's range, ensuring pages that contain
        # only figures/captions (which produce no narrative blocks) are still scanned.
        min_p, max_p = min(valid_pages), max(valid_pages)
        wanted = list(range(min_p, min(doc.page_count, max_p + 1)))
        for p in wanted:
            if _stop(cancelled):
                return dict(EMPTY)
            items.extend(_scan_page(doc[p], p))
        if not items:
            _publish(cache_root, fp, revision, [], {}, [])
            return {"state": "none", "revision": revision, "items": [], "refs": {}}

        refs = find_references(blocks, block_pages, items)
        wanted_ids = {r.figure_id for rs in refs.values() for r in rs}
        keep = [it for it in items if it.id in wanted_ids]
        if not keep:
            _publish(cache_root, fp, revision, [], {}, [])
            return {"state": "none", "revision": revision, "items": [], "refs": {}}

        rendered, assets = [], []
        id_map = {}
        for it in keep:
            if _stop(cancelled):
                return dict(EMPTY)
            try:
                pix = doc[it.page].get_pixmap(clip=pymupdf.Rect(*it.rect),
                                              dpi=RENDER_DPI)
                png = pix.tobytes("png")
            except Exception:
                continue                      # one bad crop is not a failed index
            if len(png) < 200:
                continue
            sha = hashlib.sha256(png).hexdigest()
            # Derive the final public ID from the source fingerprint, label key,
            # and rendered byte hash. If the PDF content behind this crop is edited,
            # it receives a new ID, avoiding collisions with immutable browser caches.
            final_id = "f-" + hashlib.sha256(f"{fp}:{it.label_key}:{sha}".encode("utf-8")).hexdigest()[:12]
            id_map[it.id] = final_id
            rendered.append(FigureItem(
                **{**asdict(it),
                   "id": final_id,
                   "asset_sha256": sha,
                   "width": pix.width, "height": pix.height}))
            assets.append((final_id, png))
    finally:
        doc.close()

    kept_ids = {it.id for it in rendered}
    refs = {bi: [FigureRef(id_map[r.figure_id], r.label, r.page, r.start_word, r.end_word)
                 for r in rs if r.figure_id in id_map]
            for bi, rs in refs.items()}
    refs = {bi: rs for bi, rs in refs.items() if rs}
    if not rendered or not refs:
        _publish(cache_root, fp, revision, [], {}, [])
        return {"state": "none", "revision": revision, "items": [], "refs": {}}

    manifest = _publish(cache_root, fp, revision, rendered, refs, assets)
    return manifest


def _publish(cache_root, fp, revision, items, refs, assets):
    """Assets first, each written whole, then the manifest that names them -
    so a manifest never points at a file that is still being written, and a
    crash mid-run leaves an unreferenced PNG rather than a broken index."""
    manifest = {
        "state": "ready" if items else "none",
        "revision": revision,
        "items": [it.public() for it in items],
        "refs": {str(bi): [r.public() for r in rs] for bi, rs in refs.items()},
    }
    try:
        d = cache_dir(cache_root, fp)
        os.makedirs(d, exist_ok=True)
        for fid, png in assets:
            _atomic_write(os.path.join(d, f"{fid}.png"), png)
        _atomic_write(os.path.join(d, "manifest.json"),
                      json.dumps(manifest, ensure_ascii=False))
    except OSError:
        # An unwritable cache is a slower next open, not a broken feature:
        # the manifest still goes back to the caller for this session.
        pass
    return manifest
