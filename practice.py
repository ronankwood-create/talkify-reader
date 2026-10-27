"""
Locates practice problems the book already contains - never invents any.
The rule: do not generate practice problems, because a wrong worked answer
in a study tool is worse than no tool. This module only points at text
that's already there.

    find(blocks) -> [{"heading_block": i,
                      "items": [{"block": j, "has_answer": bool,
                                 "answer": str|None}, ...]}, ...]

An answer is only ever *quoted from the item itself* - the book printing
"Ans: 4.2 kN" at the end of the question. Nothing here cross-references a
solutions appendix (book-format-dependent and failure-prone) and nothing
here works an answer out. An item whose answer this cannot read is reported
honestly as a thought question, and the reader is told so, rather than being
offered a box to check against something that was guessed.

Finds nothing on a document with no problem-set heading - callers must treat
an empty list as "this feature doesn't apply here", not an error.
"""
from __future__ import annotations

import re

_HEADING = re.compile(
    r"^(problems|exercises|exercise\s*set|review\s*questions|questions|"
    r"discussion\s*questions)\s*[:.]?\s*$", re.I)
_ITEM = re.compile(r"^\(?\d{1,3}[.\)]\s+\S")
_ITEM_NUMBER = re.compile(r"^\(?\d{1,3}[.\)]\s+")

# How far past a heading to keep looking for numbered items before giving up
# on it being a real problem set (a false-positive heading match otherwise
# ties up the scan for the rest of the document).
_MAX_SPAN = 60

# The handful of ways a textbook actually prints an answer beside its
# question:
#
#     3. Find the reaction at A. Ans: 4.2 kN
#     3. Find the reaction at A. (Ans. 4.2 kN)
#     3. Find the reaction at A. [4.2 kN]
#
# The label form has to follow the end of a sentence or an opening bracket,
# which is what keeps "1. Answer the following in one sentence." from being
# read as an answer of "the following in one sentence" - the item number is
# stripped before matching, so a leading "Answer" has nothing in front of it.
# The bare-bracket form must contain a digit, since a bracketed aside without
# one is a citation or an editorial note, not a result.
#
# Deliberately not matched: "(a) ... (b) ...", which in an exercise means
# sub-parts of the question far more often than it means a worked answer.
# Treating it as an answer key would show the reader a "correct answer" that
# is really half the question - exactly the wrong-answer-in-a-study-tool
# failure this module exists to avoid. Anything less explicit than the forms above is
# left as a thought question.
# The label form has to follow the end of a sentence or an opening bracket,
# and requires an explicit delimiter (a colon, dash, equals, or period with
# whitespace), which keeps "1. Answer in one sentence." or "1. Do you agree?
# Answer in one sentence." from being misread as an answer.
# The bare-bracket form must contain a digit, and must not contain citation
# or reference keywords (like 'see', 'chapter', 'fig', 'page') which denote
# cross-references rather than numerical results.
_ANSWER_LABEL = re.compile(
    r"(?:[.?!]\s+|[\[(])\s*(?:ans|answer|soln|solution)s?\b\s*(?:[:=\-–—]\s*|\.\s+)"
    r"(?P<a>\S.*)$", re.I)
_CITATION_WORDS = re.compile(
    r"\b(?:see|chapter|ch|section|sec|page|p|fig|figure|ref|reference|table|note)\b", re.I)
_ANSWER_BRACKET = re.compile(r"\[\s*(?P<a>[^\[\]]*\d[^\[\]]*?)\s*\]\s*$")


def answer_in(item_text):
    """The answer this item prints for itself, or None if it prints none."""
    body = _ITEM_NUMBER.sub("", item_text.strip(), count=1)
    m = _ANSWER_LABEL.search(body)
    if m:
        ans = m.group("a").strip().rstrip(")].,;").strip()
        if ans:
            return ans
    m = _ANSWER_BRACKET.search(body)
    if m:
        raw = m.group("a").strip()
        if not _CITATION_WORDS.search(raw):
            ans = raw.rstrip(")].,;").strip()
            if ans:
                return ans
    return None


# No string _HEADING can match is longer than this - the longest alternative
# is "discussion questions" plus optional punctuation, and blocks arrive from
# extract.split_blocks with runs of whitespace already collapsed. Checking a
# length before stripping and matching turns the scan over a whole library at
# startup (every block of every book) into an integer comparison for all but a
# handful of lines.
_HEADING_MAX = 64


def _is_heading(text):
    return len(text) <= _HEADING_MAX and _HEADING.match(text.strip()) is not None


def find(blocks):
    out = []
    i, n = 0, len(blocks)
    while i < n:
        if _is_heading(blocks[i]):
            j = i + 1
            items = []
            while j < n and j - i < _MAX_SPAN and not _is_heading(blocks[j]):
                if _ITEM.match(blocks[j].strip()):
                    answer = answer_in(blocks[j])
                    items.append({"block": j, "has_answer": answer is not None,
                                  "answer": answer})
                j += 1
            if items:
                out.append({"heading_block": i, "items": items})
            i = j
        else:
            i += 1
    return out
