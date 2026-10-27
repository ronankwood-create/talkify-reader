"""
Get readable text out of whatever the user opens.

Each extractor returns a list of blocks: paragraph-sized strings, in reading
order, already cleaned of the things that ruin audio (running heads, hyphenated
line breaks, figure captions).
"""
from __future__ import annotations

import os
import re

MAX_BLOCK = 320       # Kokoro does best on short chunks; also keeps latency low

# Titles and initialisms that end in a period without ending the sentence.
# The splitter can't use a lookbehind for these (Python's re needs a fixed
# width there), so it splits first and stitches these back onto what follows.
_ABBREV = {
    "mr", "mrs", "ms", "dr", "prof", "sr", "jr", "st", "vs", "etc",
    "e.g", "i.e", "u.s", "u.k", "no", "fig", "eq", "al", "inc", "ltd",
    "co", "corp", "ave", "blvd", "jan", "feb", "mar", "apr", "jun", "jul",
    "aug", "sep", "sept", "oct", "nov", "dec",
}


def _ends_in_abbrev(sent):
    words = sent.rstrip(".!?").split()
    return bool(words) and words[-1].lower() in _ABBREV


def _split_sentences(para):
    out = []
    for part in re.split(r"(?<=[.!?])\s+(?=[A-Z\"'(])", para):
        if out and _ends_in_abbrev(out[-1]):
            out[-1] = f"{out[-1]} {part}"
        else:
            out.append(part)
    return out


def _pack(pieces, limit):
    """Greedily join pieces into chunks no longer than limit."""
    out, cur = [], ""
    for piece in pieces:
        if cur and len(cur) + len(piece) + 1 > limit:
            out.append(cur.strip())
            cur = piece
        else:
            cur = f"{cur} {piece}".strip()
    if cur.strip():
        out.append(cur.strip())
    return out


def _split_long_sentence(sent, limit):
    """A sentence with too little punctuation to land under limit on its own
    breaks on clause boundaries, then hard word-wraps as a last resort."""
    chunks = _pack(re.split(r"(?<=[,;:])\s+", sent), limit)
    out = []
    for chunk in chunks:
        if len(chunk) <= limit:
            out.append(chunk)
            continue
        cur = ""
        for w in chunk.split(" "):
            if cur and len(cur) + len(w) + 1 > limit:
                out.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            out.append(cur)
    return out


def split_blocks(text, limit=MAX_BLOCK):
    """Paragraphs, then sentences, so no chunk is long enough to stall playback."""
    out = []
    for para in re.split(r"\n\s*\n", text):
        para = re.sub(r"\s+", " ", para).strip()
        if not para:
            continue
        if len(para) <= limit:
            out.append(para)
            continue
        cur = ""
        for sent in _split_sentences(para):
            if len(sent) > limit:
                if cur:
                    out.append(cur.strip())
                    cur = ""
                out.extend(_split_long_sentence(sent, limit))
                continue
            if len(cur) + len(sent) + 1 > limit and cur:
                out.append(cur.strip())
                cur = sent
            else:
                cur = f"{cur} {sent}".strip()
        if cur.strip():
            out.append(cur.strip())
    return out


def from_txt(path):
    with open(path, encoding="utf-8", errors="replace") as f:
        return split_blocks(f.read())


def from_pdf(path, pages=None, math="skip", progress=None):
    """Reuses the readaloud pipeline: reading order, de-hyphenation, running-head
    removal, spoken-form equations and units, OCR for scans.

    `progress`, if given, is called as `progress(done, total)` while OCR runs -
    a scanned book can take minutes, and the caller (talkify.py, for a
    document open the browser is waiting on) needs page counts to show real
    feedback instead of a spinner that looks the same whether it's stuck or
    fifteen minutes from done."""
    import readaloud
    return split_blocks(readaloud.convert(path, pages=pages, math=math, ocr="auto",
                                          progress=progress))


def blockify_pages(pairs, limit=MAX_BLOCK):
    """(paragraph, page) pairs -> (blocks, block_pages).

    The last step of every PDF path, shared so that a book opened a page at a
    time and the same book opened all at once produce identical blocks. A
    paragraph that splits into three blocks carries its page onto all three.
    """
    blocks, block_pages = [], []
    for para, page in pairs:
        pieces = split_blocks(para, limit)
        blocks.extend(pieces)
        block_pages.extend([page] * len(pieces))
    return blocks, block_pages


def from_pdf_with_pages(path, pages=None, math="skip", limit=MAX_BLOCK):
    """Like from_pdf, but also returns which 0-based PDF page each block
    came from. Used only by chapters.py's background job to place a
    detected chapter (a PDF page number) onto a block index - reprocesses
    the PDF a second time, which is fine since it always runs once, off to
    the side, well after the document is already open and readable.

    Must be called with the same (pages, math) the document was originally
    opened with, or the block list here won't line up with doc["blocks"]."""
    import readaloud
    merged, merged_pages = readaloud._convert_paragraphs(path, pages=pages, math=math, ocr="auto")
    return blockify_pages(zip(merged, merged_pages), limit)


def from_epub(path):
    try:
        from ebooklib import epub, ITEM_DOCUMENT
        from bs4 import BeautifulSoup
    except ImportError:
        raise SystemExit("EPUB support needs:  python -m pip install ebooklib beautifulsoup4")

    tags = ["h1", "h2", "h3", "h4", "p", "li", "blockquote"]
    book = epub.read_epub(path)
    blocks = []
    for item in book.get_items_of_type(ITEM_DOCUMENT):
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for tag in soup(["script", "style", "nav"]):
            tag.decompose()
        for el in soup.find_all(tags):
            if el.find_parent(tags):
                continue          # its text is already captured by the ancestor
            t = re.sub(r"\s+", " ", el.get_text(" ", strip=True))
            if len(t) > 1:
                blocks.extend(split_blocks(t))
    return blocks


def load(path, **kw):
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return from_pdf(path, **kw)
    if ext == ".epub":
        return from_epub(path)
    if ext in (".txt", ".md", ".text", ""):
        return from_txt(path)
    raise SystemExit(f"don't know how to read {ext or 'that file'} yet")
