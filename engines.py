"""
Speech engines for Talkify.

Every engine returns the same thing, and that contract is the whole design:

    synth(text) -> (audio_bytes, mime, [Word(w, start, end), ...])

Times are seconds from the start of that clip. The player never knows or cares
which engine produced them, so a new backend is one class and nothing in the UI
changes.
"""
from __future__ import annotations

import base64
import collections
import difflib
import io
import itertools
import json
import math
import os
import re
import struct
import subprocess
import threading
import wave

import worker_protocol
from dataclasses import dataclass, asdict

HERE = os.path.dirname(os.path.abspath(__file__))


@dataclass
class Word:
    w: str
    s: float          # start, seconds
    e: float          # end, seconds


@dataclass(frozen=True)
class SpokenPlan:
    """The text the models hear, and how it maps back to what is displayed.

    ``source_words`` is the only display/highlight truth - it is exactly
    ``words_of(original_text)`` and nothing here may change it. Everything
    else describes the *speech*, which is deliberately not the same string:
    ``PSI`` is spoken ``P S I``, a citation is not spoken at all, and a URL is
    spoken as one word.

    ``source_for_spoken`` names, for each spoken word, the one displayed token
    it belongs to. It is monotonic and in bounds, and repeats are normal (one
    displayed token can expand to several spoken words). A displayed token
    with no spoken word at all is legal only when it is in ``silent_sources``;
    those get their timings from the gap around them so the highlight still
    crosses them at a sensible moment.
    """
    source_words: tuple[str, ...]       # tuple(words_of(original_text))
    spoken_text: str                    # exact TTS/ASR input
    spoken_words: tuple[str, ...]       # tuple(words_of(spoken_text))
    source_for_spoken: tuple[int, ...]  # one source index per spoken word
    silent_sources: frozenset[int]      # visible tokens deliberately unspoken


def words_of(text):
    """Source-of-truth tokens: what the reader sees and what gets highlighted."""
    return [m.group(0) for m in re.finditer(r"\S+", text)]


# All-caps words whose default English pronunciation is wrong when they're
# actually an abbreviation - "PSI" the pressure unit reads letter-by-letter
# ("P S I"), not as "psi" the Greek letter. Verified against what Whisper
# actually transcribes back (see the PSI investigation): spelling it out with
# spaces flips the heard word from lowercase "psi" to uppercase "PSI" as one
# token, which then matches the source token directly - no special-casing
# needed in align().
SPELL_OUT = {"PSI"}


def _spell_out(token):
    """SPELL_OUT applied inside one source token, never across tokens.

    The hyphen is excluded from the boundary as well as word characters: \b
    treats "PSI-high" as containing the whole word PSI, so spelling it out
    produced "P S I-high", and the fragment "I-high" is not a word anybody
    asked a speech model to say. An abbreviation joined to something by a
    hyphen is left alone."""
    def fix(m):
        w = m.group(0)
        return " ".join(w) if w in SPELL_OUT else w
    return re.sub(r"(?<![\w-])[A-Z]{2,}(?![\w-])", fix, token)


# ------------------------------------------------- acoustic de-noising

# What a reference-heavy page contains that nobody wants read aloud. All three
# patterns are deliberately narrow: one that fires on ordinary prose deletes
# text the reader can see, which is far worse than reading one citation out.
URL_RE = re.compile(r"(?i)(?:https?://|www\.)[^\s<>\[\]{}]+")
NUMERIC_CITATION_RE = re.compile(
    r"\[\s*\d+(?:\s*(?:,|;|[\-–])\s*\d+)*\s*\]"
)
_NAME = (r"[A-ZÀ-ÖØ-Þ]"
         r"[A-Za-zÀ-ÖØ-öø-ÿ'’.-]*")
_ONE_REFERENCE = (
    _NAME
    + r"(?:\s+(?:and|&)\s+" + _NAME + r")?"
    + r"(?:\s+et\s+al\.)?"
    + r"(?:\s*,\s*|\s+)\d{4}[a-z]?"
)
AUTHOR_DATE_RE = re.compile(
    r"\(\s*" + _ONE_REFERENCE
    + r"(?:\s*;\s*" + _ONE_REFERENCE + r")*"
    + r"(?:\s*,\s*(?:(?:p|pp|ch|sec)\.)?\s*\d+(?:\s*[\-–]\s*\d+)?)?"
    + r"\s*\)"
)

# The false-positive guard on AUTHOR_DATE_RE. The grammar above already
# refuses "(Apollo 11 launched in 1969)" - no four-digit year sits where the
# year has to sit - and every lowercase math parenthetical. What it cannot see
# is that a capitalized word may open a cross-reference rather than name a
# person: "(See 2020)" parses as author-year and is not one. Anything listed
# here is spoken normally, which is the safe direction - the cost of a miss is
# one citation read aloud, the cost of a false positive is visible text that
# is never spoken at all.
_NOT_AUTHORS = frozenset("""
    See Also Note Notes Compare Cf Source Sources From Since When While After
    Before Until During Between Under Over Above Below Within Through
    The This That These Those There Here It They We You Our
    In On At By For With And But Or If As Of To
    Figure Fig Figures Table Tbl Tables Section Sec Chapter Ch Equation Eq
    Appendix Page Pages Part Volume Vol Line Step Case Example Problem
    Exercise Version Revision Year Model Type Class Grade Room Suite
    Born Died Founded Established Published Written Composed Built Painted
    Invented Discovered Completed Signed Opened Circa Ca About Around
    January February March April June July August September October
    November December
""".split())

# A publication year, not a room number or a quantity. Wide enough for any
# book Talkify will open, narrow enough that "(Bay 4200)" stays speech.
_YEAR_RANGE = (1400, 2100)

# What a removed citation or a spoken URL may leave behind at either end of
# the source token that owned it. Anything else means the match cut a real
# word in half, and the whole segment is refused.
_TRAILING_PUNCT = re.compile(
    "[.,;:!?)\\]}\"'‘’“”…]*\\Z")
# "(https://example.com)" is one token whose URL starts one character in. The
# opener is not part of the address and not a word either, so it is kept and
# spoken around the link rather than making the whole token fail to match.
_LEADING_PUNCT = re.compile("\\A[(\\[{\"'‘“¡¿]*")

_TOKEN_RE = re.compile(r"\S+")


def _plausible_citation(matched):
    """Reject an author-date match that is really ordinary prose."""
    inner = matched.strip()[1:-1].strip()
    first = re.match(r"[^\s,;]+", inner)
    if first is None:
        return False
    if first.group(0).rstrip(".,;") in _NOT_AUTHORS:
        return False
    # Only the first four-digit number. A locator - "(Smith, 2020, p. 1200)" -
    # puts a page number inside the same parentheses, and checking every
    # number in there rejected the citation because page 1200 is not a
    # plausible year.
    years = re.findall(r"\b(\d{4})[a-z]?\b", inner)
    if years and not (_YEAR_RANGE[0] <= int(years[0]) <= _YEAR_RANGE[1]):
        return False
    return True


# Closing brackets a URL may genuinely end with, and their openers.
_BRACKET_PAIRS = {")": "(", "]": "[", "}": "{"}


def _plausible_numeric_citation(matched):
    """Reject a bracket that is maths or a score rather than a reference list.

    The pattern alone cannot tell "[12-14]" from "[6-4, 7-5]" or "[0, 1]", and
    silencing an array index or a closed interval in an engineering textbook
    is exactly the failure this whole feature must not have. Three cheap
    facts about reference numbers do most of the work: they start at one, they
    are small, and a range runs upward.

    "[1, 2]" as a closed interval is genuinely indistinguishable from
    references 1 and 2 and is read as a citation. That is the residual, and it
    is the right way round for a book with a bibliography.
    """
    numbers = [int(n) for n in re.findall(r"\d+", matched)]
    if not numbers or any(n < 1 or n > 999 for n in numbers):
        return False
    for low, high in re.findall(r"(\d+)\s*[\-\u2013]\s*(\d+)", matched):
        if int(high) <= int(low):
            return False
    return True


# Real text trails a URL with a full stop, maybe a closing quote. The bound
# exists so that a line of OCR noise - "example.com" followed by three hundred
# dots - cannot turn one token into three hundred full regex matches.
MAX_URL_TAIL_TRIM = 8


def _trim_url(text, start, end):
    """The URL without the sentence punctuation the greedy pattern swallowed.

    Returns the new end offset. Only trims while what is left is still a whole
    URL, and never trims a closing bracket the address itself opened -
    Wikipedia's "..._(programming_language)" ends in a parenthesis that is
    part of the address, not punctuation around it."""
    for _ in range(MAX_URL_TAIL_TRIM):
        if end <= start or text[end - 1] not in ".,;:!?)]}\"'’”":
            break
        closing = _BRACKET_PAIRS.get(text[end - 1])
        if closing is not None and text.count(closing, start, end) >= text.count(
                text[end - 1], start, end):
            break
        if URL_RE.fullmatch(text[start:end - 1]) is None:
            break
        end -= 1
    return end


def _candidate_segments(text):
    """Every URL/citation match, left to right, with later overlaps dropped."""
    found = []
    for m in URL_RE.finditer(text):
        found.append((m.start(), _trim_url(text, m.start(), m.end()), "url"))
    for m in NUMERIC_CITATION_RE.finditer(text):
        if _plausible_numeric_citation(m.group(0)):
            found.append((m.start(), m.end(), "citation"))
    for m in AUTHOR_DATE_RE.finditer(text):
        if _plausible_citation(m.group(0)):
            found.append((m.start(), m.end(), "citation"))
    # Earliest start wins, longest wins a tie; anything overlapping something
    # already accepted is dropped rather than truncated, because a half-eaten
    # citation is exactly what would strand a word fragment in the speech.
    found.sort(key=lambda seg: (seg[0], -(seg[1] - seg[0])))
    accepted, reach = [], -1
    for start, end, kind in found:
        if start < reach:
            continue
        accepted.append((start, end, kind))
        reach = end
    return accepted


def _owned_tokens(text, spans, start, end):
    """(first, last, head) for the source tokens a [start, end) span covers.

    `head` is whatever of the first token sits before the match: empty
    normally, or the opening bracket or quote the text wrapped the match in.
    None means the span begins inside a real word, and is not ours to take.
    """
    first = last = None
    for i, (ts, te) in enumerate(spans):
        if te <= start or ts >= end:
            continue
        if first is None:
            first = i
        last = i
    if first is None:
        return None
    head = text[spans[first][0]:start]
    if head and _LEADING_PUNCT.fullmatch(head) is None:
        return None                     # begins mid-word: not ours to take
    return first, last, head


def make_spoken_plan(text, pronunciations=None):
    """Normalize speech while retaining display-word provenance.

    Built by scanning the original text left to right and consuming whole
    source tokens - never by substituting into the string and then guessing
    where the words went. Every emitted spoken word records the one displayed
    token it came from, so a highlight can never land on a word the reader is
    not hearing, and a token that is deliberately not spoken is recorded as
    such rather than silently losing its place.
    """
    spans = [m.span() for m in _TOKEN_RE.finditer(text)]
    source_words = tuple(text[s:e] for s, e in spans)
    segments = {}
    reach = 0                           # first token no segment has claimed
    for start, end, kind in _candidate_segments(text):
        owned = _owned_tokens(text, spans, start, end)
        if owned is None:
            continue
        first, last, head = owned
        tail = text[end:spans[last][1]]
        if _TRAILING_PUNCT.fullmatch(tail) is None:
            continue                    # ends mid-token: leave it as speech
        if kind == "url" and first != last:
            continue                    # a URL owning several tokens is not one
        if first < reach:
            continue                    # already owned by an accepted segment
        segments[first] = (last, kind, head, tail)
        reach = last + 1

    table = _PRONUNCIATIONS if pronunciations is None else pronunciations
    words, sources, deliberately_silent = [], [], set()

    def emit(word, source_index):
        if word:
            words.append(word)
            sources.append(source_index)

    def glue(suffix):
        """Punctuation a removed citation left behind. It joins the previous
        spoken word rather than standing alone, so "See [12-14]." speaks as
        "See." and not as "See" followed by a lone full stop."""
        if suffix and words:
            words[-1] += suffix

    i = 0
    while i < len(source_words):
        segment = segments.get(i)
        if segment is None:
            token = source_words[i]
            spoken = _apply_pronunciations(token, table)
            if spoken == token:
                # The dictionary is the user's explicit instruction, so it
                # wins; SPELL_OUT is only the built-in fallback.
                spoken = _spell_out(token)
            pieces = spoken.split()
            for piece in pieces:
                emit(piece, i)
            if not pieces:
                deliberately_silent.add(i)
            i += 1
            continue
        last, kind, head, tail = segment
        if kind == "url":
            emit(head + "link" + tail, i)
        else:
            for k in range(i, last + 1):
                deliberately_silent.add(k)
            # One separator comma goes with the citation it separated; a
            # sentence stop belongs to the sentence and stays, and so does an
            # opening bracket the citation was written inside.
            glue(head)
            glue(tail[1:] if tail.startswith(",") else tail)
        i = last + 1

    spoken_text = " ".join(words)
    spoken_words = tuple(words)
    source_for_spoken = tuple(sources)
    represented = set(source_for_spoken)
    silent_sources = frozenset(
        i for i in range(len(source_words)) if i not in represented)

    # These are the contract, not paranoia. Every one of them, violated, shows
    # up as the highlight sitting on a word other than the one being spoken -
    # which nothing downstream can detect, and no user can report usefully.
    if len(spoken_words) != len(source_for_spoken):
        raise ValueError("speech normalization lost word provenance")
    if tuple(words_of(spoken_text)) != spoken_words:
        raise ValueError("speech normalization lost word provenance")
    if any(not 0 <= s < len(source_words) for s in source_for_spoken):
        raise ValueError("speech normalization produced an out-of-range source")
    if any(a > b for a, b in zip(source_for_spoken, source_for_spoken[1:])):
        raise ValueError("speech normalization reordered the source")
    if not silent_sources <= deliberately_silent:
        raise ValueError("speech normalization dropped a source word")
    return SpokenPlan(source_words, spoken_text, spoken_words,
                      source_for_spoken, silent_sources)


def spoken_form(text):
    """What gets synthesized - never what's displayed or highlighted. Only
    words_of(text) on the original text is source-of-truth for that."""
    return make_spoken_plan(text).spoken_text


# ------------------------------------------- the pronunciation dictionary

@dataclass(frozen=True)
class Pronunciations:
    """A compiled, immutable pronunciation table.

    One alternation regex rather than a substitution per entry: applying
    entries in sequence would let a later key match inside an earlier one's
    replacement, so an "SQL" -> "S Q L" entry plus an entry for "S" would
    rewrite the expansion it had just produced. One pass cannot do that.
    """
    entries: tuple = ()
    pattern: object = None
    values: tuple = ()

    def __bool__(self):
        return bool(self.entries)


_NO_PRONUNCIATIONS = Pronunciations()

# Replaced wholesale, never mutated - a plan already being built keeps the
# table it started with, and a reader mid-paragraph never sees half an edit.
_PRONUNCIATIONS = _NO_PRONUNCIATIONS
_PRONUNCIATION_FINGERPRINT = object()      # nothing compares equal to this

MAX_PRONUNCIATION_KEY = 128
MAX_PRONUNCIATION_VALUE = 256
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def load_pronunciations(raw):
    """Compile config.json's ``pronunciations`` map, or an empty table.

    Deliberately silent about what it rejected. This is user text from a
    hand-edited file that may sit next to an API key, so no entry is ever
    logged; a bad one simply does not apply and the word is spoken normally.
    A map that is not a map at all yields the empty table, which reads exactly
    like the feature not existing.
    """
    if not isinstance(raw, dict):
        return _NO_PRONUNCIATIONS
    accepted = {}
    for key, value in raw.items():
        if not isinstance(key, str) or not isinstance(value, str):
            continue
        key, value = key.strip(), value.strip()
        if not key or not value:
            continue
        if len(key) > MAX_PRONUNCIATION_KEY or len(value) > MAX_PRONUNCIATION_VALUE:
            continue
        if _CONTROL_CHARS.search(key) or _CONTROL_CHARS.search(value):
            continue
        accepted[key] = value
    if not accepted:
        return _NO_PRONUNCIATIONS
    # Longest key first so "Navier-Stokes" beats "Navier"; lexical after that,
    # so the same config always compiles to the same table.
    entries = tuple(sorted(accepted.items(), key=lambda kv: (-len(kv[0]), kv[0])))
    parts, values = [], []
    for index, (key, value) in enumerate(entries):
        body = re.escape(key)
        if key == key.lower() and key != key.upper():
            # An all-lowercase entry is a request about a word, not about a
            # spelling: "euler" should fix "Euler" too. Anything the user
            # capitalized is taken literally.
            body = f"(?i:{body})"
        parts.append(f"(?P<p{index}>{body})")
        values.append(value)
    try:
        pattern = re.compile(r"(?<!\w)(?:" + "|".join(parts) + r")(?!\w)")
    except re.error:
        return _NO_PRONUNCIATIONS
    return Pronunciations(entries, pattern, tuple(values))


def _apply_pronunciations(token, table=None):
    table = _PRONUNCIATIONS if table is None else table
    if table.pattern is None:
        return token
    values = table.values
    return table.pattern.sub(lambda m: values[int(m.lastgroup[1:])], token)


def refresh_speech_settings(raw):
    """Re-read the pronunciation map at the synthesis/config-read boundary.

    Fingerprinted so the common case - config.json unchanged since the last
    block - costs one canonical dump and a string compare rather than
    recompiling a regex per paragraph. The published table is swapped by a
    single assignment; two callers arriving together may both compile, but
    they compile the same thing and whichever lands last is correct.
    """
    global _PRONUNCIATIONS, _PRONUNCIATION_FINGERPRINT
    try:
        fingerprint = json.dumps(raw, sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        fingerprint = None
    if fingerprint is not None and fingerprint == _PRONUNCIATION_FINGERPRINT:
        return _PRONUNCIATIONS
    table = load_pronunciations(raw)
    _PRONUNCIATIONS = table
    _PRONUNCIATION_FINGERPRINT = fingerprint
    return table


def pronunciation_table():
    """The table in force right now. For tests and diagnostics only."""
    return _PRONUNCIATIONS


def _norm(tok):
    return re.sub(r"[^\w']+", "", tok).lower()


def align(source_tokens, heard):
    """Map recognised words (with timings) back onto the source tokens.

    Whisper does not return the text we fed in — it drops punctuation, splits
    hyphenates, mishears the odd word. Matching on normalised forms and
    interpolating across the gaps keeps the highlight on the right word even
    where recognition slipped.
    """
    if not heard:
        return []
    a = [_norm(t) for t in source_tokens]
    b = [_norm(h.w) for h in heard]
    out: list[Word | None] = [None] * len(source_tokens)

    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for i, j, n in sm.get_matching_blocks():
        for k in range(n):
            h = heard[j + k]
            out[i + k] = Word(source_tokens[i + k], h.s, h.e)

    # fill unmatched runs by spreading them evenly between known anchors
    total = heard[-1].e
    last_end = 0.0
    i = 0
    while i < len(out):
        if out[i] is not None:
            last_end = out[i].e
            i += 1
            continue
        j = i
        while j < len(out) and out[j] is None:
            j += 1
        nxt = out[j].s if j < len(out) else total
        span = max(nxt - last_end, 1e-3)
        step = span / (j - i)
        for k in range(i, j):
            out[k] = Word(source_tokens[k],
                          last_end + step * (k - i),
                          last_end + step * (k - i + 1))
        i = j
    return [w for w in out if w]


def _fill_runs(starts, ends, total):
    """Give every untimed slot an interval taken from the gap around it.

    Used twice: once for spoken words a backend did not mark, and once for
    displayed tokens that are deliberately never spoken (a citation). Both are
    the same problem - a run of blanks between two known anchors - and both
    have to be filled by *position*, never by matching text, or two
    occurrences of the same word swap their timings.
    """
    n = len(starts)
    i = 0
    last_end = 0.0
    while i < n:
        if starts[i] is not None:
            last_end = ends[i]
            i += 1
            continue
        j = i
        while j < n and starts[j] is None:
            j += 1
        nxt = starts[j] if j < n else total
        step = max(nxt - last_end, 1e-3) / (j - i)
        for k in range(i, j):
            starts[k] = last_end + step * (k - i)
            ends[k] = last_end + step * (k - i + 1)
        i = j
    return starts, ends


def _settle(words, starts, ends, audio_seconds):
    """One finite, monotonic, in-audio interval per displayed token.

    The player treats a zero-length or out-of-order interval as a word it can
    never be inside, so the highlight sticks. Epsilon is the smallest nudge
    that satisfies its strict-interval requirement without visibly moving
    anything; the final rescale only runs when the nudges have pushed the last
    word fractionally past the end of the audio.
    """
    if not words:
        return []
    limit = audio_seconds if (isinstance(audio_seconds, (int, float))
                              and math.isfinite(audio_seconds)
                              and audio_seconds > 0) else None
    eps = max(0.001, (limit or 0.0) / 100000.0)
    out = []
    previous = 0.0
    for i, word in enumerate(words):
        s, e = starts[i], ends[i]
        if s is None or not math.isfinite(s):
            s = previous
        if e is None or not math.isfinite(e):
            e = s + eps
        s = max(float(s), previous, 0.0)
        e = max(float(e), s + eps)
        out.append(Word(word, s, e))
        previous = e
    if limit is not None and out[-1].e > limit:
        # Scale rather than clamp: clamping would pile the tail of the block
        # onto one instant, and the words there are the ones a reader is
        # watching when a paragraph ends.
        scale = limit / out[-1].e
        out = [Word(w.w, w.s * scale, w.e * scale) for w in out]
    return out


def _proportional(plan, audio_seconds):
    """Timings with no acoustic evidence at all: every displayed token gets a
    share of the clip proportional to its length.

    The last resort, and it has to exist. A block whose timings failed would
    otherwise never advance, and a reader would sit on a silent paragraph
    with no way to tell what went wrong."""
    words = plan.source_words
    if not words:
        return []
    total = audio_seconds if (isinstance(audio_seconds, (int, float))
                              and math.isfinite(audio_seconds)
                              and audio_seconds > 0) else 0.35 * len(words)
    weights = [max(1, len(w)) for w in words]
    span = sum(weights)
    starts, ends, t = [], [], 0.0
    for weight in weights:
        share = total * weight / span
        starts.append(t)
        ends.append(t + share)
        t += share
    return _settle(list(words), starts, ends, total)


def wav_seconds(data):
    """Playing length of a WAV clip, or None if the header cannot be read.

    The one number every timing decision is clamped against, so it is read
    from the audio actually returned rather than assumed from a sample rate a
    backend was asked for and may not have honoured."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            rate = w.getframerate()
            return w.getnframes() / rate if rate else None
    except Exception:
        return None


def silence_wav(seconds, rate=22050):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * max(seconds, 0.1)))
    return buf.getvalue()


def silent_block(plan, mime="audio/wav"):
    """A playable block for text with no speech in it.

    A paragraph that is nothing but a citation still has to advance: the
    player waits for "ended" on a real element, and a block that never
    produces audio is a reader stuck on a paragraph forever. Short, valid,
    silent, and with a timing for every displayed token - and no model is
    asked for any of it."""
    seconds = max(0.25, 0.06 * len(plan.source_words))
    # The clip is a shade longer than the last timing on purpose. Frame
    # rounding makes a WAV a few tens of microseconds shorter than the length
    # asked for, and "the audio is at least as long as its last word" is an
    # invariant the player and the export both rely on.
    return silence_wav(seconds + 0.05), mime, _proportional(plan, seconds)


def _spoken_marks(plan, heard, spoken_indices, audio_seconds=None):
    """One Word per *spoken* word, or None if the evidence cannot support that.

    Native marks name their own position, so they are placed by index and the
    unmarked ones filled from the gap. Whisper has no positions, so its
    transcript is aligned against the spoken text - and against the spoken
    text only. Aligning it against the displayed text is the bug this whole
    plan exists to prevent: the two are not the same string.
    """
    if not heard:
        return None
    count = len(plan.spoken_words)
    if (spoken_indices is not None and len(spoken_indices) == len(heard)
            and all(isinstance(i, int) and not isinstance(i, bool)
                    and 0 <= i < count for i in spoken_indices)
            and all(a < b for a, b in zip(spoken_indices, spoken_indices[1:]))):
        starts = [None] * count
        ends = [None] * count
        for index, mark in zip(spoken_indices, heard):
            starts[index], ends[index] = mark.s, mark.e
        # The clip, not the last mark. A backend may hand back marks for every
        # word but the last one - the native validator tolerates a few percent
        # of unmarked words - and filling that trailing run up to the final
        # mark gives the closing word a millisecond of highlight while it is
        # still being spoken a second later.
        total = heard[-1].e
        if (isinstance(audio_seconds, (int, float))
                and math.isfinite(audio_seconds) and audio_seconds > total):
            total = audio_seconds
        _fill_runs(starts, ends, total)
        return [Word(plan.spoken_words[i], starts[i], ends[i])
                for i in range(count)]
    # No positional evidence: an older worker, or a Whisper transcript.
    marks = align(list(plan.spoken_words), heard)
    return marks if len(marks) == count else None


def plan_words(plan, heard, *, spoken_indices=None, audio_seconds=None):
    """Timings for every displayed token, collapsed through the plan.

    This is the only path from a backend's marks to what the player
    highlights. Marks describe `plan.spoken_text`; the reader sees
    `plan.source_words`; and after de-noising those are different lengths,
    different words, and in places not even a subsequence of each other. The
    provenance map is what makes the collapse exact instead of a guess.
    """
    if not plan.source_words:
        return []
    if not plan.spoken_words:
        # Nothing was spoken - a citation-only block. There is no evidence to
        # collapse and none is needed.
        return _proportional(plan, audio_seconds)
    spoken = _spoken_marks(plan, heard, spoken_indices, audio_seconds)
    if spoken is None:
        return _proportional(plan, audio_seconds)

    starts = [None] * len(plan.source_words)
    ends = [None] * len(plan.source_words)
    for mark, source_index in zip(spoken, plan.source_for_spoken):
        starts[source_index] = (mark.s if starts[source_index] is None
                                else min(starts[source_index], mark.s))
        ends[source_index] = (mark.e if ends[source_index] is None
                              else max(ends[source_index], mark.e))
    if all(s is None for s in starts):
        return _proportional(plan, audio_seconds)
    # Everything still blank is a silent source - a citation, a bracketed
    # reference - and takes its share of the gap it sits in.
    timed = [e for e in ends if e is not None]
    total = audio_seconds if (isinstance(audio_seconds, (int, float))
                              and math.isfinite(audio_seconds)
                              and audio_seconds > 0) else max(timed)
    _fill_runs(starts, ends, total)
    return _settle(list(plan.source_words), starts, ends, audio_seconds)


def align_native(plan, heard, spoken_indices=None, audio_seconds=None):
    """Backwards-compatible name for plan_words()."""
    return plan_words(plan, heard, spoken_indices=spoken_indices,
                      audio_seconds=audio_seconds)


# --------------------------------------------------------------- stub

VOICES = {
    "stub": [("stub", "Stub (silent)")],
    "kokoro": [
        ("af_heart", "Heart (US Female)"),
        ("af_bella", "Bella (US Female)"),
        ("af_nicole", "Nicole (US Female)"),
        ("af_sarah", "Sarah (US Female)"),
        ("am_adam", "Adam (US Male)"),
        ("am_michael", "Michael (US Male)"),
        ("bf_emma", "Emma (UK Female)"),
        ("bf_isabella", "Isabella (UK Female)"),
        ("bm_george", "George (UK Male)"),
        ("bm_lewis", "Lewis (UK Male)"),
    ],
    "edge": [
        ("en-US-AndrewNeural", "Andrew (US Male)"),
        ("en-US-AriaNeural", "Aria (US Female)"),
        ("en-US-GuyNeural", "Guy (US Male)"),
        ("en-GB-RyanNeural", "Ryan (UK Male)"),
        ("en-GB-SoniaNeural", "Sonia (UK Female)"),
    ],
}


_VOICE_LISTS = {}
_VOICE_IDS = {}


def voices_for(engine_name):
    """Static catalogue, not a live query - kokoro has no voice-listing call
    and edge_tts's list_voices() needs the network on every gallery load.
    Scoped to the engine actually running: switching *between* kokoro and
    edge live would mean juggling two model/process backends with different
    mime types mid-document, which is a bigger architecture change than a
    voice picker warrants. Voice switching stays within the active engine.

    The dicts are built once per engine and shared. The list wrapping them is
    fresh each call, so a caller that sorts or filters its copy cannot disturb
    anyone else's; the dicts themselves are read-only by convention."""
    cached = _VOICE_LISTS.get(engine_name)
    if cached is None:
        cached = _VOICE_LISTS[engine_name] = [
            {"id": vid, "label": label} for vid, label in VOICES.get(engine_name, [])]
    return list(cached)


def voice_ids_for(engine_name):
    """The same catalogue as a frozenset of ids. Every /api/block and
    /api/audio request checks a requested voice against this, so it is on the
    playback path and has no business rebuilding ten dicts to do it."""
    ids = _VOICE_IDS.get(engine_name)
    if ids is None:
        ids = _VOICE_IDS[engine_name] = frozenset(
            vid for vid, _label in VOICES.get(engine_name, []))
    return ids


class StubEngine:
    """Silent audio with plausible timings.

    Exists so the reader, the highlighting and the scrolling can be built and
    checked without downloading a gigabyte of models first.
    """
    name = "stub"
    mime = "audio/wav"

    def __init__(self, rate=1.0, **_):
        self.rate = rate

    def synth(self, text, voice=None):
        toks = words_of(text)
        wps = 2.6 * self.rate           # words per second, roughly read-aloud pace
        timings, t = [], 0.0
        for tok in toks:
            dur = max(0.12, len(tok) / 14.0) / self.rate
            timings.append(Word(tok, t, t + dur))
            t += dur + 0.06 / self.rate
        return self._silence(t), self.mime, timings

    _silence = staticmethod(silence_wav)


# --------------------------------------------------------------- edge

class EdgeEngine:
    """Microsoft neural voices. Emits WordBoundary events, so timings come free
    and exact — no recognition step. Needs the network."""
    name = "edge"
    mime = "audio/mpeg"

    def __init__(self, voice="en-US-AndrewNeural", rate=1.0, **_):
        self.voice = voice
        self.pct = f"{int(round((rate - 1) * 100)):+d}%"

    def synth(self, text, voice=None):
        import asyncio, edge_tts

        plan = make_spoken_plan(text)
        # A block that is nothing but citations has no speech in it. Edge is
        # the one engine whose audio is MP3, and the export encoder is told
        # the document's mime once, so this cannot answer with a WAV the way
        # the local engines do - it asks for a bare stop instead, which is
        # the shortest thing the service will accept.
        spoken = plan.spoken_text or "."

        async def run():
            audio, marks = bytearray(), []
            comm = edge_tts.Communicate(spoken, voice or self.voice, rate=self.pct)
            async for piece in comm.stream():
                if piece["type"] == "audio":
                    audio.extend(piece["data"])
                elif piece["type"] == "WordBoundary":
                    s = piece["offset"] / 1e7          # 100-ns ticks to seconds
                    d = piece["duration"] / 1e7
                    marks.append(Word(piece["text"], s, s + d))
            return bytes(audio), marks

        audio, marks = asyncio.run(run())
        # Word boundaries describe plan.spoken_text, so they are collapsed
        # through the plan - never aligned straight onto the displayed words.
        total = marks[-1].e if marks else None
        return audio, self.mime, plan_words(plan, marks, audio_seconds=total)


# ------------------------------------------------------- kokoro + whisper

# ------------------------------------------------------- worker transports

# The parent half of the two transports. Both present the same three things to
# WorkerClient - read one reply, send one request, send one control message -
# so the client's locking, epochs and pending map are written once and are not
# aware of which framing is underneath.

WORKER_PROTOCOLS = ("jsonl-v1", "binary-v1")


def worker_protocol_name():
    """Which transport a *new* worker epoch will use.

    Read here and nowhere else, and only when an epoch is created: changing it
    under a live worker would mean writing frames to a process reading lines.
    Anything unrecognised is JSONL, because the rollback path has to be the
    one that cannot be reached by a typo.
    """
    name = (os.environ.get("TALKIFY_WORKER_PROTOCOL") or "jsonl-v1").strip()
    return name if name in WORKER_PROTOCOLS else "jsonl-v1"


class JsonlLink:
    """One JSON object per line, audio as base64. The rollback transport."""

    name = "jsonl-v1"
    text_streams = True
    needs_handshake = False

    def __init__(self, proc):
        self.proc = proc

    def read(self):
        """(reply_dict, note) - exactly one of them, or (None, None) at EOF."""
        line = self.proc.stdout.readline()
        if not line:
            return None, None
        line = line.strip()
        if not line:
            return None, ""
        try:
            resp = json.loads(line)
            if not isinstance(resp, dict):
                raise ValueError("response is not an object")
        except Exception:
            # One unreadable line is stray output, not a dead worker. Killing
            # every live request over it would turn a library's warning print
            # into a failed paragraph.
            return None, f"unparseable line: {line[:120]!r}"
        return resp, None

    def _send(self, obj):
        self.proc.stdin.write(json.dumps(obj) + "\n")
        self.proc.stdin.flush()

    def send_synth(self, payload, req_id):
        self._send({**payload, "op": "synth", "req_id": req_id})

    def send_cancel(self, req_id):
        self._send({"op": "cancel", "req_id": req_id})

    def send_shutdown(self):
        self._send({"op": "shutdown"})

    def acknowledge(self, meta):
        return None


class BinaryLink:
    """TKF1 frames, raw WAV bodies.

    The reader always consumes a frame's entire body before it returns, even
    for a request nobody is waiting on any more. Skipping the body of a late
    or cancelled result would leave the samples in the pipe, and the next
    header read would start inside them - which is the one failure this
    protocol has no way to recover from.
    """

    name = "binary-v1"
    text_streams = False
    needs_handshake = True

    def __init__(self, proc):
        self.proc = proc
        self.writer = worker_protocol.FrameWriter(proc.stdin)
        self.hello = None

    def read(self):
        try:
            frame = worker_protocol.read_frame(self.proc.stdout)
        except worker_protocol.StreamClosed:
            return None, None
        if frame.kind == worker_protocol.HELLO:
            self.hello = frame.meta
            return {"__hello__": frame.meta}, None
        if frame.kind == worker_protocol.RESULT:
            meta = frame.meta
            diagnostics = meta.get("diagnostics")
            diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
            return {"req_id": frame.req_id, "ok": True,
                    "audio": frame.body,
                    "heard": meta.get("words") or [],
                    "sample_rate": meta.get("sample_rate"),
                    "mime_type": meta.get("mime_type"),
                    "alignment_method": meta.get("alignment_method"),
                    "alignment_fallback_reason": diagnostics.get(
                        "alignment_fallback_reason"),
                    "native_direct_coverage": diagnostics.get(
                        "native_direct_coverage", 0.0)}, None
        if frame.kind == worker_protocol.ERROR:
            return {"req_id": frame.req_id, "ok": False,
                    "error": frame.meta.get("error")}, None
        raise worker_protocol.ProtocolError(
            f"unexpected {worker_protocol.KIND_NAMES.get(frame.kind)} frame")

    def send_synth(self, payload, req_id):
        self.writer.write(worker_protocol.SYNTHESIZE,
                          {**payload, "req_id": req_id})

    def send_cancel(self, req_id):
        self.writer.write(worker_protocol.CANCEL, {"req_id": req_id})

    def send_shutdown(self):
        self.writer.write(worker_protocol.SHUTDOWN, {})

    def acknowledge(self, meta):
        worker_protocol.check_hello(meta, expect_protocol=self.name)
        self.writer.write(worker_protocol.HELLO_ACK, {"version":
                                                      worker_protocol.VERSION})


LINKS = {"jsonl-v1": JsonlLink, "binary-v1": BinaryLink}


# Alignment methods whose marks carry their own spoken-word position.
NATIVE_ALIGNMENT_METHODS = frozenset({
    "kokoro-pytorch-duration-v1", "kokoro-onnx-duration-v1"})

KOKORO_VENV = os.path.join(HERE, "kokoro_venv")
KOKORO_WORKER = os.path.join(HERE, "kokoro_worker.py")
KOKORO_LOG = os.path.join(HERE, "kokoro_worker.log")


@dataclass(frozen=True)
class SynthMeta:
    """What the coordinator knows about a request and the engine does not need
    to. Carried through purely so a pipelined engine can order and cancel
    sensibly; nothing here reaches the cache key, which is still
    (doc_id, block_index, voice)."""
    request_id: str = ""
    doc_id: str = ""
    block_index: int = -1
    generation: int = 0
    priority: int = 0
    background: bool = False


class SynthError(RuntimeError):
    """One request failed. Subclasses RuntimeError because every existing
    caller - the HTTP layer included - already treats a synthesis failure as
    one, and this must not change what a user sees."""

    def __init__(self, message, *, code="error", stage="", retryable=False,
                 fatal=False):
        super().__init__(message)
        self.code = code
        self.stage = stage
        self.retryable = retryable
        self.fatal = fatal


class CancelledSynth(SynthError):
    """This request was superseded before it produced anything worth keeping."""


class WorkerDied(SynthError):
    """The subprocess went away. Every request that was in flight fails with
    one shared instance, so a caller can tell "the worker died" from "this
    block failed"."""

    def __init__(self, message):
        super().__init__(message, code="worker_died", stage="process",
                         retryable=True, fatal=True)


def _synth_error(name, err):
    if not isinstance(err, dict):
        return SynthError(f"{name} worker: malformed error response")
    code = err.get("code") or "error"
    message = err.get("message") or code
    cls = CancelledSynth if code == "cancelled" else SynthError
    return cls(f"{name} worker: {message}", code=code,
               stage=err.get("stage") or "", retryable=bool(err.get("retryable")),
               fatal=bool(err.get("fatal")))


class WorkerHandle:
    """One outstanding request. `result()` blocks like the old synchronous
    call did; `cancel()` is advisory and idempotent."""

    __slots__ = ("request_id", "_client", "_epoch", "_done", "_value", "_error")

    def __init__(self, request_id, client, epoch):
        self.request_id = request_id
        self._client = client
        self._epoch = epoch
        self._done = threading.Event()
        self._value = None
        self._error = None

    def _complete(self, value=None, error=None):
        if self._done.is_set():
            return                          # a terminal reply already landed
        self._value = value
        self._error = error
        self._done.set()

    def done(self):
        return self._done.is_set()

    def result(self, timeout=None):
        if not self._done.wait(timeout):
            raise TimeoutError(f"request {self.request_id} timed out")
        if self._error is not None:
            raise self._error
        return self._value

    def cancel(self):
        """Ask the worker to drop this if it can. Safe to call at any point,
        including after the result has landed, and safe to call twice."""
        if self._done.is_set():
            return
        self._client.cancel(self.request_id)


class WorkerClient:
    """A long-lived JSON-line subprocess with more than one request in flight.

    What this replaces: one lock held across "write a request, read a line".
    That made the pipe safe by making it single-file - the second request did
    not exist until the first response was complete, so no amount of
    concurrency inside the worker could have helped.

    Now every request carries an id, one reader thread owns stdout, and
    responses are matched by id. Responses may arrive out of order (a cancel,
    a failed stage), so nothing here may ever infer identity from arrival
    order.

    `spawn()` returns (Popen, cleanup) and is the only thing that knows what
    the worker is - which is what lets smoke.py drive this against a fake
    worker with no models in it.

    Three locks, and they are never nested:

      _start_lock   which process is current, and spawning the next one
      _write_lock   exactly one JSON line plus flush() on stdin
      _state        the pending map

    A blocked stdin write holds _write_lock for as long as the worker is
    wedged, so every other path that wants it takes it with a timeout and
    carries on without it. Killing the process is what unblocks that writer,
    and that must never be the thing waiting on it.
    """

    def __init__(self, spawn, *, name="worker", tail=None):
        self._spawn = spawn
        self._name = name
        self._tail = tail
        self._start_lock = threading.Lock()   # which process is current
        self._write_lock = threading.Lock()   # exactly one JSON line + flush
        self._state = threading.Lock()        # the pending map
        self._proc = None
        self._cleanup = None
        self._reader = None
        self._link = None
        self._ready = None          # this epoch's handshake, or None if it needs none
        self._epoch = 0
        self._pending = {}
        self._stopped = False
        self._counter = itertools.count(1)
        self.notes = collections.deque(maxlen=50)
        self.identity = ""          # what the live worker said it is

    def _note(self, message):
        """Protocol oddities that must not become exceptions: a stray line on
        stdout, a reply for a request nobody is waiting on. Kept in memory so
        a test can assert they happened and were ignored."""
        self.notes.append(message)

    def _death_message(self):
        tail = ""
        if self._tail is not None:
            try:
                tail = self._tail() or ""
            except Exception:
                tail = ""
        return f"{self._name} worker died: {tail}" if tail else f"{self._name} worker died"

    # ------------------------------------------------------------ lifecycle
    def _detach(self, epoch):
        """Give up the process for `epoch` if it is still the current one.
        Returns what has to be cleaned up, or (None, None) if somebody else
        already took it."""
        with self._start_lock:
            if self._epoch != epoch or self._proc is None:
                return None, None
            dead, cleanup = self._proc, self._cleanup
            self._proc = self._cleanup = None
            self._link = None
            ready, self._ready = self._ready, None
        if ready is not None:
            # Anything waiting on this epoch's handshake has to be released,
            # or a submit() blocks until its timeout on a process that is
            # already being buried.
            ready.set()
        return dead, cleanup

    def _fail(self, epoch, error):
        """Complete every future belonging to `epoch` (or to any epoch, when
        `epoch` is None) with one shared error.

        Futures are completed with no lock held: a waiter woken here runs its
        own continuation, and doing that under the pending-map lock is how a
        deadlock gets built.
        """
        with self._state:
            doomed = [h for h in self._pending.values()
                      if epoch is None or h._epoch == epoch]
            for h in doomed:
                self._pending.pop(h.request_id, None)
        for h in doomed:
            h._complete(error=error)

    def _reap(self, dead, cleanup):
        """Stop a detached process and let go of its handles.

        Order matters on Windows and this is the order: terminate first, so a
        thread blocked writing to a wedged worker gets an error and drops
        _write_lock; wait for the process to actually be gone, because
        TerminateProcess is asynchronous and the next spawn will reopen the
        log file this one still has open; only then close anything.
        """
        if dead is None:
            return
        try:
            dead.terminate()
        except Exception:
            pass
        try:
            dead.wait(timeout=5)
        except Exception:
            pass
        # A writer holding this is inside a write to a process that no longer
        # exists, so it is about to fail and let go. If it somehow does not,
        # closing under it would corrupt its buffer - skip the close instead
        # and let the object's own finaliser deal with the descriptors.
        held = self._write_lock.acquire(timeout=2)
        try:
            for stream in (dead.stdin, dead.stdout):
                try:
                    stream.close()
                except Exception:
                    pass
        finally:
            if held:
                self._write_lock.release()
        if cleanup is not None:
            try:
                cleanup()
            except Exception:
                pass

    def _retire(self, epoch, error):
        """The process for `epoch` is gone: detach it, fail everything that
        belonged to it, and let the next submit start a fresh one lazily."""
        dead, cleanup = self._detach(epoch)
        self._fail(epoch, error)
        self._reap(dead, cleanup)

    def _ensure(self):
        """The current live process, starting one if there is not one.

        A dead process is retired here rather than merely replaced. Leaving it
        for its own reader thread to notice is what would strand its futures:
        once a newer epoch exists, that reader's retirement no longer owns the
        process, and a shutdown that only looked at the current epoch would
        never complete them at all.
        """
        for _ in range(2):
            with self._start_lock:
                if self._stopped:
                    raise RuntimeError(f"{self._name} client is stopped")
                proc, epoch = self._proc, self._epoch
                if proc is not None and proc.poll() is None:
                    return proc, epoch
                if proc is None:
                    return self._start_locked()
                # A corpse. Drop the claim under the lock, then bury it
                # outside - terminate() and wait() are far too slow to hold a
                # lock every submit takes.
                dead, cleanup = proc, self._cleanup
                self._proc, self._cleanup = None, None
                self._link = None
                ready, self._ready = self._ready, None
            if ready is not None:
                # This process is being buried, so nothing is going to
                # acknowledge its hello. Release anyone waiting for one rather
                # than making them sit out the whole handshake timeout.
                ready.set()
            self._fail(epoch, WorkerDied(self._death_message()))
            self._reap(dead, cleanup)
        raise WorkerDied(f"{self._name} worker could not be started")

    def _start_locked(self):
        """Caller holds _start_lock. Spawning stays inside it so two callers
        arriving together cannot start two workers.

        The transport is decided here and only here - when the epoch is
        created. Reading the flag anywhere else would let it change under a
        live worker, which means writing frames to a process reading lines."""
        try:
            spawned = self._spawn()
            proc, cleanup = spawned[0], spawned[1]
            link_name = spawned[2] if len(spawned) > 2 else "jsonl-v1"
        except Exception as e:
            # A missing venv or an unreadable interpreter is not one bad
            # block - every request would fail the same way - so it is
            # reported as fatal and the warmer gives up on the document
            # rather than failing once per paragraph for the rest of a book.
            raise WorkerDied(f"{self._name} worker could not start: {e}") from e
        self._epoch += 1
        self._proc = proc
        self._cleanup = cleanup
        link = self._link = LINKS.get(link_name, JsonlLink)(proc)
        self._ready = threading.Event() if link.needs_handshake else None
        if self._ready is None:
            self.identity = link_name
        epoch = self._epoch
        self._reader = threading.Thread(
            target=self._read, args=(proc, epoch, link), daemon=True,
            name=f"{self._name}-reader")
        self._reader.start()
        return proc, epoch

    # --------------------------------------------------------------- reader
    def _read(self, proc, epoch, link):
        try:
            while True:
                resp, note = link.read()
                if resp is None:
                    if note is None:
                        break               # clean EOF between frames/lines
                    if note:
                        self._note(note)
                    continue
                hello = resp.get("__hello__")
                if hello is not None:
                    if not self._handshake(epoch, link, hello):
                        return
                    continue
                if self._dispatch(resp):
                    # A fatal reply. The worker stops itself, but not waiting
                    # to find that out means the next request starts a healthy
                    # process instead of going to a corpse.
                    self._retire(epoch, WorkerDied(
                        f"{self._name} worker reported a fatal error"))
                    return
        except Exception as e:
            # A framing failure lands here too, and is treated exactly like a
            # dead worker: there is no safe resynchronisation point on a
            # stream whose bodies can contain the frame magic.
            self._retire(epoch, WorkerDied(f"{self._name} worker read failed: {e}"))
            return
        self._retire(epoch, WorkerDied(self._death_message()))

    def _handshake(self, epoch, link, hello):
        """Validate the worker's HELLO and acknowledge it. False means the
        epoch has been retired and this reader must stop.

        The acknowledgement goes out under the frame writer's own lock rather
        than the client's: no request can be in flight yet, because submit()
        blocks until this event is set, and reaching for the client's write
        lock from the reader thread is how a wedged writer would deadlock the
        one thread that could notice it."""
        try:
            link.acknowledge(hello)
        except Exception as e:
            self._retire(epoch, WorkerDied(
                f"{self._name} worker handshake failed: {e}"))
            return False
        with self._start_lock:
            if self._epoch != epoch:
                return True                 # a newer worker already owns this
            self.identity = "{}:{}:{}".format(
                hello.get("protocol"), hello.get("backend"),
                hello.get("model_identity"))
            ready = self._ready
        if ready is not None:
            ready.set()
        return True

    def _dispatch(self, resp):
        """Complete one future. Returns True if the reply was fatal.

        The transport has already consumed the whole message, body included,
        before this is called - a result nobody is waiting on still has to be
        drained off the pipe, or the next read starts inside its audio."""
        req_id = resp.get("req_id")
        if not isinstance(req_id, str):
            self._note("reply with no request id")
            return False
        with self._state:
            handle = self._pending.pop(req_id, None)
        if handle is None:
            # Never guess. A reply whose id nobody is holding is dropped -
            # attaching it to some other caller is how the wrong audio gets
            # played for the wrong paragraph.
            self._note(f"reply for unknown request {req_id}")
            return False
        if resp.get("ok"):
            handle._complete(value=resp)
            return False
        error = _synth_error(self._name, resp.get("error"))
        handle._complete(error=error)
        return error.fatal

    # --------------------------------------------------------------- submit
    HANDSHAKE_TIMEOUT = 30.0

    def submit(self, payload):
        proc, epoch = self._ensure()
        with self._start_lock:
            link, ready = self._link, self._ready
        if ready is not None and not ready.wait(self.HANDSHAKE_TIMEOUT):
            error = WorkerDied(f"{self._name} worker never said hello")
            self._retire(epoch, error)
            raise error
        with self._start_lock:
            if self._epoch != epoch or self._link is not link:
                # The handshake failed and the epoch was retired while we
                # waited. Do not send into a process that is being buried.
                raise WorkerDied(f"{self._name} worker was replaced")
        req_id = f"{epoch:x}-{next(self._counter):x}-{os.urandom(3).hex()}"
        # Reject an unencodable payload before registering: a request that
        # cannot be sent must not leave a future in the pending map that no
        # reply will ever reach.
        json.dumps(payload)
        handle = WorkerHandle(req_id, self, epoch)
        with self._state:
            if self._stopped:
                raise RuntimeError(f"{self._name} client is stopped")
            self._pending[req_id] = handle
        try:
            with self._write_lock:
                if self._proc is not proc:
                    raise OSError("worker was replaced before the request was sent")
                link.send_synth(payload, req_id)
        except Exception as e:
            # Half a line may already be in the pipe, so this process cannot
            # be reused and the request cannot be silently retried - a retry
            # that the worker also received would synthesize twice, which is
            # invisible and expensive. Fail, and let the caller's own
            # deduplication decide whether to ask again.
            error = WorkerDied(f"{self._name} worker write failed: {e}")
            self._retire(epoch, error)
            with self._state:
                self._pending.pop(req_id, None)
            handle._complete(error=error)
            raise error
        return handle

    def cancel(self, req_id):
        """Best effort, and never fatal to the caller. A cancel that does not
        arrive costs one wasted block, which is exactly what it was trying to
        save - but a cancel that half-arrives corrupts the line the next
        request writes, so a failed write retires the process."""
        with self._start_lock:
            proc, epoch, link = self._proc, self._epoch, self._link
        if proc is None or link is None or proc.poll() is not None:
            return
        try:
            with self._write_lock:
                if self._proc is not proc:
                    return
                link.send_cancel(req_id)
        except Exception as e:
            self._retire(epoch, WorkerDied(f"{self._name} worker write failed: {e}"))

    def inflight(self):
        with self._state:
            return len(self._pending)

    def shutdown(self, timeout=2.0):
        with self._start_lock:
            self._stopped = True
            proc, epoch, link = self._proc, self._epoch, self._link
            reader = self._reader
        if proc is not None and link is not None and proc.poll() is None:
            # A courtesy, not a requirement. If a wedged worker is holding the
            # write lock, skip the polite goodbye rather than joining it in
            # the deadlock - terminate() below is what actually ends this.
            if self._write_lock.acquire(timeout=1):
                try:
                    link.send_shutdown()
                except Exception:
                    pass
                finally:
                    self._write_lock.release()
        dead, cleanup = self._detach(epoch)
        # Every epoch, not just the current one: an older process's reader may
        # still be alive, and nothing else will ever complete its futures.
        self._fail(None, WorkerDied(f"{self._name} client shut down"))
        self._reap(dead, cleanup)
        if reader is not None:
            reader.join(timeout)


class KokoroEngine:
    """Fully offline. Kokoro's own duration predictor supplies normal word
    timings; faster-whisper listens back only when native timings are invalid.

    Kokoro's English phonetics need spaCy, which has no wheels for this
    project's Python version yet, so the actual synthesis
    runs in a dedicated Python 3.12 venv (kokoro_venv/, see
    setup_kokoro.ps1) via kokoro_worker.py, talked to as a long-lived
    subprocess so models load once instead of once per block. Only the raw
    (audio, heard) crosses that boundary. Native spoken-word timings are
    collapsed through a provenance map here; Whisper fallback timings still
    go through align() exactly as before.

    synth() is unchanged for every caller. submit() is the extra capability
    the coordinator uses to keep two blocks moving through the worker's TTS
    and ASR stages at once; it is deliberately not part of the engine contract
    that the player, the routes or the export path see.
    """
    name = "kokoro"
    mime = "audio/wav"

    _client = None
    _client_lock = threading.Lock()
    _log = None

    def __init__(self, voice="af_heart", rate=1.0, whisper_model="base.en", **_):
        self.voice = voice
        self.rate = rate
        self.whisper_model = whisper_model

    # ------------------------------------------------------------ the process
    @classmethod
    def _spawn(cls):
        python = os.path.join(KOKORO_VENV, "Scripts", "python.exe")
        if not os.path.exists(python):
            raise RuntimeError(
                f"kokoro venv not found at {KOKORO_VENV} - run setup_kokoro.ps1 first")
        # The worker's stray stdout (kokoro/torch/spaCy warnings, HF Hub
        # notices) is redirected to its own stderr - see kokoro_worker.py.
        # That has to go to a file, not subprocess.PIPE: nothing here drains a
        # stderr pipe on the happy path, and over a long session enough small
        # warnings eventually fill the OS pipe buffer and block the worker's
        # next write() forever.
        # A restart races the previous process's own handle on this file -
        # TerminateProcess is asynchronous, and Windows will not reopen a file
        # the corpse still holds. Reaping waits for that, but the log is a
        # convenience and synthesis is not: if it still cannot be opened, run
        # without it rather than refusing to speak.
        try:
            log = open(KOKORO_LOG, "w", encoding="utf-8")
        except OSError:
            log = None
        cls._log = log
        # The transport is named on the worker's own command line, never
        # inferred from what turns up on the pipe. binary-v1 opens the pipes
        # in binary mode: a text wrapper would translate a lone newline inside
        # a WAV body into CRLF on Windows and corrupt audio that still plays.
        protocol = worker_protocol_name()
        binary = protocol == "binary-v1"
        proc = subprocess.Popen(
            [python, KOKORO_WORKER, "--protocol", protocol],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=log if log is not None else subprocess.DEVNULL,
            **({} if binary else
               {"text": True, "encoding": "utf-8", "bufsize": 1}))
        return proc, (log.close if log is not None else None), protocol

    @staticmethod
    def _log_tail():
        try:
            with open(KOKORO_LOG, encoding="utf-8", errors="replace") as f:
                return f.read()[-2000:]
        except OSError:
            return ""

    @classmethod
    def client(cls):
        with cls._client_lock:
            if cls._client is None:
                cls._client = WorkerClient(cls._spawn, name="kokoro",
                                           tail=cls._log_tail)
            return cls._client

    def audio_identity(self):
        """What made the audio now being cached: transport, backend, model.

        A cache entry filled by one backend must never be served as if the
        other had made it - the bytes are different, and so is the timing
        method. Empty until a worker has actually said hello, which is
        correct: nothing has been made yet."""
        client = type(self)._client
        return getattr(client, "identity", "") if client is not None else ""

    def warm(self):
        """Spawn the worker and load its models now, in the background - so
        the ~30s cold-start cost lands while the browser tab is opening,
        not on the first paragraph someone actually tries to hear."""
        def run():
            try:
                self.synth("Hello.")
            except Exception:
                pass
        threading.Thread(target=run, daemon=True).start()

    # ------------------------------------------------------------ synthesis
    def submit(self, text, voice=None, *, meta=None):
        plan = make_spoken_plan(text)
        if not plan.spoken_words:
            # Nothing to say: a paragraph that is only a citation, or only a
            # bracketed reference. Waking Kokoro and Whisper to produce eight
            # seconds of nothing would cost as much as a real block, so this
            # answers straight away - and still returns a timing for every
            # displayed token, so the highlight and the block advance behave
            # exactly as they do everywhere else.
            return _SettledHandle(silent_block(plan, self.mime))
        payload = {"text": plan.spoken_text, "voice": voice or self.voice,
                   "rate": self.rate, "whisper_model": self.whisper_model,
                   "priority": meta.priority if meta is not None else 0}
        inner = self.client().submit(payload)
        return _KokoroHandle(inner, plan, self.mime)

    def synth(self, text, voice=None):
        return self.submit(text, voice=voice).result()

    @classmethod
    def shutdown(cls, timeout=2.0):
        with cls._client_lock:
            client, cls._client = cls._client, None
        if client is not None:
            client.shutdown(timeout)


class _SettledHandle:
    """A result that existed before anyone asked for it. Same three methods
    the coordinator uses on a real handle, so nothing upstream has to know
    that this block never reached the worker."""

    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    @property
    def request_id(self):
        return ""

    def done(self):
        return True

    def result(self, timeout=None):
        return self._value

    def cancel(self):
        pass


class _KokoroHandle:
    """Decodes the worker's reply in the *caller's* thread.

    Base64 decoding a WAV and running align() are not free, and doing either
    in the single stdout reader would put them between two responses. Here
    they cost only the thread that was waiting anyway.
    """

    __slots__ = ("_inner", "_plan", "_mime", "_value")

    def __init__(self, inner, plan, mime):
        self._inner = inner
        self._plan = plan
        self._mime = mime
        self._value = None

    @property
    def request_id(self):
        return self._inner.request_id

    def done(self):
        return self._inner.done()

    def result(self, timeout=None):
        if self._value is None:
            resp = self._inner.result(timeout)
            wav = resp.get("audio")
            if wav is None:
                wav = base64.b64decode(resp["audio_b64"])
            raw_heard = resp.get("heard", [])
            heard = [Word(h["w"], h["s"], h["e"]) for h in raw_heard]
            indices = None
            if resp.get("alignment_method") in NATIVE_ALIGNMENT_METHODS:
                indices = [h.get("i") for h in raw_heard]
            # Marks describe the plan's spoken text either way - Whisper heard
            # that text, and the duration predictor produced it. Collapsing
            # through the plan is the only thing that gets them back onto the
            # words on screen.
            words = plan_words(self._plan, heard, spoken_indices=indices,
                               audio_seconds=wav_seconds(wav))
            self._value = (wav, self._mime, words)
        return self._value

    def cancel(self):
        self._inner.cancel()


ENGINES = {e.name: e for e in (StubEngine, EdgeEngine, KokoroEngine)}


def build(name, **kw):
    if name not in ENGINES:
        raise SystemExit(f"unknown engine {name!r}; try one of {', '.join(ENGINES)}")
    return ENGINES[name](**kw)
