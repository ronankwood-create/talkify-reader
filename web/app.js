/* Talkify reader: plays one block at a time and highlights the spoken word.
   Timings arrive as [{w, s, e}] per block; the player is engine-agnostic.

   Three views (library/reader/voices) share one page so the player bar and
   the <audio> element never get torn down when you switch views - that's
   what makes playback "persistent" across navigation. Views that are not
   current are just display:none; their DOM (including the rendered reader
   text) stays alive underneath.

   SECTION INDEX (for token-efficient sliced viewing):
     - State & Elements:          ~L41
     - Reader DOM & Highlight:    ~L199
     - Audio Element Standby:     ~L684
     - Gapless Web Audio:         ~L838
     - Block & Audio Loader:      ~L1420
     - Controls & Scrubbing:      ~L1775
     - Position Heartbeat:        ~L1857
     - Speed Controls:            ~L1893
     - Voice Switching Barrier:   ~L1916
     - Voice Gallery:             ~L2174
     - Popovers:                  ~L2357
     - Theme:                     ~L2372
     - Sleep Timer:               ~L2394
     - In-Doc Search:             ~L2422
     - Chapters Navigation:       ~L2538
     - Figures & Illustrations:   ~L2608
     - AI Explanations:           ~L2781
     - Back to Position:          ~L2929
     - Split View & PDF Pages:    ~L2953
     - View Switching:            ~L3131
     - File Open, Upload & OCR:   ~L3157
     - Library View & Recents:    ~L3340
     - Background Export:         ~L3586
     - Application Boot:          ~L3931
*/

const $ = (id) => document.getElementById(id);
const reader = $("reader"), audio = $("audio"), previewAudio = $("previewAudio");

const state = {
  docId: null,       // the document currently shown/playing
  blocks: [],
  words: {},          // block index -> [{w,s,e}]
  audioUrls: {},       // block index -> object URL for its synthesized audio
  audioInflight: {},   // block index -> {voice, promise}; coalesces duplicate fetches
  /* The same, for the timings half. Without it every paragraph boundary sent
     /api/block twice for the same block: handoff() calls primeStandby(next+1)
     and prefetch(next+1) one after the other, and both begin by loading that
     block's words. The server deduplicates the synthesis, so it never cost
     two runs of Kokoro - it cost a second request that blocks a server thread
     and one of the browser's six connections to this origin for as long as
     the synthesis takes, which is exactly the window the next seek or
     heartbeat has to get through. */
  blockInflight: {},   // block index -> {voice, promise}; the same, for timings
  /* Which voice each cached block was made in. A voice switch purges the
     cache, but it cannot purge a fetch that is already in flight: the
     prefetcher fires two blocks ahead, so a request sent in the old voice
     routinely resolves after the switch and would otherwise write an
     old-voice blob straight back into the cache it was just cleared from.
     That is what made a switched voice alternate old/new block by block.
     Tagging every entry lets both the writer and the reader tell stale from
     current. */
  wordVoices: {},      // block index -> the voice state.words[i] was timed in
  audioVoices: {},     // block index -> the voice state.audioUrls[i] was spoken in
  /* The server names the exact synthesis behind each half of a block -
     document, block, voice - and both halves carry the same name. Timings and
     audio arrive over two separate responses, and they must never come from two different synthesis calls; comparing these
     is how the player can tell rather than assume. */
  blockKeys: {},       // block index -> asset_key reported with its timings
  audioKeys: {},       // block index -> asset_key reported with its audio
  /* The decoded form of that same audio, for the Web Audio path below. Kept
     beside the object URLs rather than instead of them: the URL is what the
     <audio> fallback plays and it is also what the decode reads from, so a
     block can be scheduled sample-exactly *and* still be playable by an
     element that has never heard of an AudioBuffer. Voice-tagged for the
     same reason every other half-block table here is. */
  audioBuffers: {},    // block index -> decoded AudioBuffer
  bufferVoices: {},    // block index -> the voice state.audioBuffers[i] was decoded from
  /* And the asset_key its bytes came from. The voice alone is not enough: the
     same block can be synthesized twice in the same voice - the keysAgree
     retry path below does exactly that - and a buffer kept across that would
     be audio from one synthesis under timings from another, which is the drift
     nothing downstream can detect. */
  bufferKeys: {},      // block index -> asset_key state.audioBuffers[i] was decoded from
  bufferInflight: {},  // block index -> {voice, doc, promise}; coalesces decoding
  i: -1,              // block currently loaded in the audio element
  wi: -1,              // index of the highlighted word
  playing: false,
  /* Two <audio> elements, ping-ponged, so the next paragraph is already
     decoded and buffered when this one ends - see the slots section below. */
  activeSlot: "A",
  /* Goes up on anything that changes what should be playing next: a seek, a
     skip, a word click, a chapter jump, a voice switch, opening another
     document. Every asynchronous slot load carries the epoch it was started
     under and refuses to touch a slot once that epoch is spent, which is what
     stops a fetch begun three paragraphs ago from arming the standby with
     audio nobody is waiting for any more. */
  playbackEpoch: 0,
  generation: 0,        // the server's session generation; see /api/voice
  figures: { state: "none", items: {}, refs: {} },
  figureShown: null,
  figuresGone: {},      // figure id -> dismissed or failed, for this session
  voice: null,          // active voice id for the current document (null = engine default)
  /* A voice switch that has been asked for but not yet confirmed. It is a
     *reservation*, never the active voice: state.voice still names what is
     coming out of the speaker, because that is what the block now sounding
     was made in. What this adds is a barrier - see beyondVoiceBarrier() -
     so that between the click and the server's answer, nothing starts any
     new work in the voice the user has just left. Without it the standby
     slot, the prefetcher and the ended-handler each had their own chance to
     begin one more old-voice paragraph while the POST was in flight, and the
     reader heard the switch take effect a paragraph late.
     {id, epoch, voice, doc, at, wasPlaying, barrier, resumeAt} */
  pendingVoice: null,
  /* Monotonic. Only the newest request may be applied, and a late answer to
     an older one must not clear a newer reservation. */
  voiceRequestEpoch: 0,
  /* The last voice the server confirmed it is on. Normally the same as
     state.voice, and it exists for the case where they part: click B, click C
     before B answers. B commits server-side and its reply is discarded as
     superseded; C is then refused as stale, because it was sent against the
     generation B has just moved past. Without this the client would go on
     believing the document is still in the voice it had before either click,
     and fetch blocks in a voice the server no longer has. */
  serverVoice: null,
  voices: [],           // catalogue for the running engine, from /api/voices
  defaultVoice: null,
  resumeSeek: null,     // {block, within} queued by opening a doc from the library, consumed on first Play
  lastHeartbeat: 0,      // Date.now() of the last /api/position send, for throttling
  search: { matches: [], idx: -1, query: "" },
  sleep: { mode: "off", deadline: null },
  autoFollow: true,     // false once the reader scrolls the page itself - see keepInView()
  /* A scanned book is readable from its first page while the rest is still
     being OCR'd, so its text arrives over time. contentRevision is the
     server's count of appends; ingestion is what it says about how far the
     scan has got. waitingForText is set only when playback has reached the
     last block that exists yet and the reader still wants to hear more. */
  contentRevision: 0,
  ingestion: { state: "complete" },
  practiceMarked: null,   // block indices that already carry a practice card
  ingestPollTimer: null,
  waitingForText: false,
  splitView: false,
  /* The page pane follows the reading position until the reader turns a page
     themselves - then it stays where they put it, the same bargain
     autoFollow makes for the reader's own scrolling. */
  pdfAutoFollow: true,
  hasPages: false,
  pageCount: 0,
  blockPages: [],
  /* Inclusive [start, end] runs of the PDF pages this document was opened
     with, or null for a whole file. Runs rather than one span, because
     "pages 20, 400-410" is thirteen pages and not three hundred and
     ninety-one. The page view is scoped to it - see loadedPages(). */
  pageRuns: null,
  currentPage: 0,
  activeExports: {},    // docId -> a background export job; see the export section
};

/* keepInView()'s own scrollBy/scrollIntoView calls fire the same "scroll"
   event a manual wheel/drag does, with nothing in the event to tell them
   apart. Without this, the very next auto-scroll after a manual one would
   look like more manual scrolling and immediately cancel autoFollow again -
   or worse, a manual scroll during the ~400ms a smooth auto-scroll animation
   takes would go undetected. Mark our own scrolls right before making them;
   the scroll listener only treats a scroll as user-initiated once that
   window has passed. */
let programmaticScrollUntil = 0;
function markProgrammaticScroll(ms = 500) {
  programmaticScrollUntil = Date.now() + ms;
}

/* ------------------------------------------------------------- rendering */

/* One paragraph, spans and all. Pulled out of render() because a book being
   read off a scanner grows after it is rendered, and a block appended later
   has to be built exactly the way the first ones were - same classes, same id
   scheme, same _words index the highlighter walks. Two builders would drift,
   and the drift would show up as highlighting that stops working somewhere
   past the middle of a long book. */
function makeBlock(text, bi) {
  const p = document.createElement("p");
  p.className = "blk";
  p.id = `b-${bi}`;
  const words = [];
  for (const piece of text.split(/(\s+)/)) {
    if (!piece) continue;
    if (!piece.trim()) { p.appendChild(document.createTextNode(piece)); continue; }
    const s = document.createElement("span");
    s.className = "w";
    s.textContent = piece;
    s._b = bi;
    s._w = words.length;
    words.push(s);
    p.appendChild(s);
  }
  p._words = words;
  return p;
}

function render(docId, blocks, practiceSets) {
  releaseAllSlots();          // before revoking: never leave a slot on a dead blob
  /* Opening another book leaves this one's lookahead in flight, holding
     connections to serve blocks of a document that is no longer on screen.
     Nothing pinned is touched, so a paragraph somebody is still waiting on
     finishes and is discarded by its own docId check as before. */
  abortStale();
  hideFigure();
  Object.values(state.audioUrls).forEach((u) => URL.revokeObjectURL(u));
  state.docId = docId;
  state.blocks = blocks;
  state.words = {};
  state.audioUrls = {};
  state.audioInflight = {};
  state.blockInflight = {};
  state.wordVoices = {};
  state.audioVoices = {};
  state.blockKeys = {};
  state.audioKeys = {};
  state.audioBuffers = {};
  state.bufferVoices = {};
  state.bufferKeys = {};
  state.bufferInflight = {};
  /* Or the boundary heartbeat could report the previous book's position for
     this book's first block - see currentWithin(). */
  wa.lastPos = { i: -1, within: 0 };
  state.i = -1;
  state.wi = -1;
  /* A freshly registered document starts at generation 0 server-side (see
     session.PlaybackSession). Carrying the previous document's number over
     would get the next voice switch refused as stale for no reason the user
     could have caused; openDocForReading overwrites this with the real one
     from /api/doc immediately after. */
  state.generation = 0;
  state.resumeSeek = null;
  state.autoFollow = true;
  state.pdfAutoFollow = true;
  state.search = { matches: [], idx: -1, query: "" };
  state.lowerBlocks = null;
  state.contentRevision = 0;
  state.ingestion = { state: "complete" };
  state.waitingForText = false;
  stopIngestionPoll();
  state.chapters = [];
  state.hasPages = false;
  state.pageCount = 0;
  state.blockPages = [];
  state.pageRuns = null;
  state.currentPage = 0;
  stopChaptersPoll();
  updateChaptersUI();
  updateSplitViewUI();
  setExplainOpen(false);
  $("selectionToolbar").hidden = true;
  $("explainHoverBtn").hidden = true;
  reader.innerHTML = "";

  /* One pass, into a fragment, and the word index is built as the spans are
     made rather than by re-finding them afterwards.

     This is the single most expensive thing the reader does: a 900-page book
     is a few hundred thousand of these spans, and it used to walk them three
     times over - once to create them, once more to look each paragraph back
     up by id and re-query its spans, and a third implicit time to write two
     data- attributes per word. Attributes are the costly part (each one is a
     string parse and a live attribute-map insert), and nothing styles or
     selects on them: only the click handler reads them back, so they are
     plain properties now. Appending one fragment also means the page is laid
     out once instead of once per paragraph. */
  const frag = document.createDocumentFragment();
  blocks.forEach((text, bi) => frag.appendChild(makeBlock(text, bi)));
  reader.appendChild(frag);

  state.practiceMarked = new Set();
  applyPractice(practiceSets);

  setEnabled(blocks.length > 0);
  status(blocks.length ? `${blocks.length} paragraphs` : "nothing open");
}

/* ------------------------------------------------- a document that grows

   A scanned book is opened from its first readable pages and the rest arrives
   while it is being read. Everything here is append-only, and that is not a
   style choice: state.i, the highlight, the audio slots, the saved position,
   chapter marks and every id in the DOM name blocks by index, so moving one
   is not a redraw - it is silent corruption of all of them.

   The poll asks for "everything from block N onward" rather than "what
   changed", so a request that is retried, duplicated or arrives late returns
   the same thing and nothing can fall between two of them. */

function stopIngestionPoll() {
  if (state.ingestPollTimer) clearTimeout(state.ingestPollTimer);
  state.ingestPollTimer = null;
}

/* Appends, and only appends. Refuses anything that does not line up exactly
   with what is on screen: another document, a gap, an overlap. Touches
   nothing about playback - not state.i, not the highlight, not the scroll
   position, not a loaded slot - because text arriving is not an event the
   reader should be able to feel. */
function appendBlocks(docId, from, blocks, blockPages, revision) {
  if (docId !== state.docId) return 0;
  if (from !== state.blocks.length) return 0;
  if (!Array.isArray(blocks) || !blocks.length) return 0;
  const frag = document.createDocumentFragment();
  blocks.forEach((text, k) => frag.appendChild(makeBlock(text, from + k)));
  reader.appendChild(frag);
  state.blocks = state.blocks.concat(blocks);
  if (Array.isArray(blockPages) && blockPages.length === blocks.length) {
    state.blockPages = (state.blockPages || []).concat(blockPages);
  }
  state.lowerBlocks = null;          // the search index is one block shorter now
  if (typeof revision === "number") state.contentRevision = revision;
  setEnabled(state.blocks.length > 0);
  return blocks.length;
}

/* Quiet, and on its own line. A book still being scanned is still a book you
   can listen to; this is provenance, not an alert. It goes away when the scan
   finishes, and a scan that ended short leaves a plain count rather than an
   error nobody can act on. */
function updateIngestNote() {
  const el = $("ingestNote");
  if (!el) return;
  const ing = state.ingestion || { state: "complete" };
  const total = ing.pages_total || 0;
  if (ing.state === "complete" || !total) {
    el.hidden = true;
    el.textContent = "";
    return;
  }
  el.hidden = false;
  el.classList.toggle("ingest-settled", ing.state !== "running");
  if (ing.state === "running") {
    el.textContent = `scanning — ${ing.pages_processed || 0} of ${total} pages ready`;
  } else {
    const got = ing.pages_succeeded != null ? ing.pages_succeeded : (ing.pages_processed || 0);
    el.textContent = `${got} of ${total} pages available`;
  }
}

/* Playback has reached the last block that exists yet. Stay in the playing
   state and wait: the suffix that arrives next will start the paragraph
   after this one. If the reader pauses in the meantime, the text still
   arrives - it just does not start speaking again. */
function waitForMoreText() {
  state.waitingForText = true;
  status("waiting for the next scanned page…");
  pollIngestion(state.docId, true);
}

function onTextArrived() {
  if (!state.waitingForText) return;
  const next = state.i + 1;
  if (next >= state.blocks.length) return;
  state.waitingForText = false;
  if (state.playing) playBlock(next);
  else status(`${Math.max(state.i, 0) + 1} of ${state.blocks.length}`);
}

/* 1.5s while a scan is running, backing off to 5s once several replies in a
   row bring nothing - a page of a scanned textbook takes a second or two, and
   a stretch of full-page figures can bring nothing for a while. 600ms only
   while playback is actually waiting at the frontier, which is the one moment
   the delay is audible. */
function pollIngestion(docId, urgent) {
  stopIngestionPoll();
  updateIngestNote();
  if (!state.ingestion || state.ingestion.state !== "running") return;
  if (docId !== state.docId) return;
  let quiet = 0;
  const tick = async () => {
    state.ingestPollTimer = null;
    if (docId !== state.docId) return;
    let reply = null;
    try {
      const r = await fetch(`/api/doc/blocks?doc=${docId}&from=${state.blocks.length}`);
      if (r.status === 409 || r.status === 404) {
        /* The server no longer has the document we are holding, or has a
           different one under this id. Stop asking; what is on screen still
           reads, and the library view is where a reload belongs. */
        state.ingestion = { state: "complete" };
        updateIngestNote();
        return;
      }
      if (r.ok) reply = await r.json();
    } catch { /* offline for a moment; ask again */ }
    if (docId !== state.docId) return;
    if (reply) {
      const added = appendBlocks(docId, reply.from, reply.blocks,
                                 reply.block_pages, reply.revision);
      if (reply.ingestion) state.ingestion = reply.ingestion;
      /* The book's own problem sets are recomputed once the scan settles, so
         the list that arrives with a late poll can cover paragraphs the first
         one did not know about. */
      applyPractice(reply.practice);
      updateIngestNote();
      if (added) { quiet = 0; onTextArrived(); } else { quiet += 1; }
      if (state.ingestion.state !== "running") {
        state.waitingForText = false;
        if (state.playing && state.i >= 0 && state.i + 1 >= state.blocks.length) stop();
        return;
      }
    } else {
      quiet += 1;
    }
    const wait = state.waitingForText ? 600 : (quiet >= 3 ? 5000 : 1500);
    state.ingestPollTimer = setTimeout(tick, wait);
  };
  state.ingestPollTimer = setTimeout(tick, urgent ? 300 : 1500);
}

/* Called from both places a document is put on screen. Navigating away stops
   the browser's poll only - the server keeps reading the book, because that
   is useful work whether or not anyone is looking at it. */
function startIngestion(docId, ingestion, revision) {
  state.ingestion = ingestion && ingestion.state ? ingestion : { state: "complete" };
  state.contentRevision = typeof revision === "number" ? revision : 0;
  state.waitingForText = false;
  pollIngestion(docId, false);
}

/* Marks the book's own practice-problem runs (see practice.py) and puts an
   attempt-it card under each one. Additive in the strictest sense: the cards
   are siblings of the paragraph, never inside it, so the word spans that
   highlighting walks are untouched, and a document with no problem set renders
   exactly as it did before.

   Idempotent, because a scanned book's problem sets are only fully known once
   the scan finishes - a set straddling the boundary between two appends is
   not recognisable until both halves exist - so this runs again with a fuller
   list and must not put a second card under a paragraph that already has one.
   Blocks that have not been appended yet are simply skipped; the next run
   catches them. */
function applyPractice(sets) {
  if (!state.practiceMarked) state.practiceMarked = new Set();
  (sets || []).forEach((set) => {
    const h = $(`b-${set.heading_block}`);
    if (h) h.classList.add("practice-heading");
    (set.items || []).forEach((item) => {
      if (state.practiceMarked.has(item.block)) return;
      const el = $(`b-${item.block}`);
      if (!el) return;
      state.practiceMarked.add(item.block);
      el.classList.add("practice-item");
      el.after(practiceCard(item));
    });
  });
}

/* One card per problem the book printed.

   Two shapes, and which one appears is decided by the book, not by us: if
   the item printed its own answer there is somewhere to type and something
   to check against; if it did not, the card says so in as many words rather
   than offering a box that could only ever mark you wrong. Nothing here
   invents an answer, and nothing here interrupts playback - narration runs
   straight past a card. */
function practiceCard(item) {
  const card = document.createElement("div");
  card.className = "practice-card";
  if (!item.has_answer) {
    card.classList.add("practice-thought");
    const banner = document.createElement("p");
    banner.className = "practice-banner";
    banner.textContent = "Thought question — no answer key provided in text";
    card.appendChild(banner);
    return card;
  }

  const row = document.createElement("div");
  row.className = "practice-row";
  const input = document.createElement("input");
  input.type = "text";
  input.className = "practice-input";
  input.placeholder = "Your answer…";
  const checkBtn = document.createElement("button");
  checkBtn.className = "ghost practice-check";
  checkBtn.textContent = "Check";
  const revealBtn = document.createElement("button");
  revealBtn.className = "ghost practice-reveal";
  revealBtn.textContent = "Skip / Reveal";
  row.append(input, checkBtn, revealBtn);

  const verdict = document.createElement("p");
  verdict.className = "practice-verdict";
  verdict.hidden = true;
  card.append(row, verdict);

  const reveal = () => {
    verdict.classList.remove("is-right");
    verdict.textContent = "The book's answer: ";
    const strong = document.createElement("strong");
    strong.textContent = item.answer;
    verdict.appendChild(strong);
    verdict.hidden = false;
  };
  checkBtn.onclick = () => {
    const mine = normaliseAnswer(input.value);
    if (!mine) { input.focus(); return; }
    const right = mine === normaliseAnswer(item.answer);
    verdict.classList.toggle("is-right", right);
    // A mismatch is reported as a mismatch, not as "wrong": these answers
    // are free text, so "4.2 kN" and "4.2 kilonewtons" are the same answer
    // and this cannot tell. Reveal is one click away either way.
    verdict.textContent = right
      ? "Matches the book's answer."
      : "That doesn't match the book's answer — reveal it?";
    verdict.hidden = false;
  };
  revealBtn.onclick = reveal;
  input.addEventListener("keydown", (e) => {
    if (e.key === "Enter") { e.preventDefault(); checkBtn.click(); }
  });
  return card;
}

/* Enough to forgive case, spacing and trailing punctuation, and nothing
   more - anything cleverer starts guessing at equivalence. */
const normaliseAnswer = (s) =>
  (s || "").toLowerCase().replace(/[\s]+/g, " ").replace(/[.,;:!?]+$/, "").trim();

function setEnabled(on) {
  for (const id of ["playBtn", "prevBtn", "nextBtn", "voiceBtn"]) $(id).disabled = !on;
}
const status = (t) => { $("playerSub").textContent = t; };
const setDocTitle = (t) => { $("title").textContent = t; $("playerTitle").textContent = t; };

/* ---------------------------------------------------------------- timing */

function clearHighlight() {
  if (state.i < 0) return;
  const p = $(`b-${state.i}`);
  if (p) p._words.forEach((el) => el.classList.remove("on", "done"));
}

function highlight(wi) {
  const p = $(`b-${state.i}`);
  if (!p) return;
  const prev = state.wi;
  if (prev >= 0 && p._words[prev]) {
    p._words[prev].classList.remove("on");
    p._words[prev].classList.add("done");
  }
  const el = p._words[wi];
  if (el) {
    el.classList.add("on");
    el.classList.remove("done");
    keepInView(el);
  }
  state.wi = wi;
}

/* Only scroll when the word drifts out of a comfortable band. Scrolling on
   every word makes the page seasick. And only while autoFollow is on - once
   the user scrolls away on their own (to read ahead, or back to re-read),
   this stays out of their way until they explicitly navigate again (a word
   click, prev/next, a chapter jump, or the "back to position" button all
   turn it back on). Without that check this fights any manual scroll during
   playback: the very next highlighted word finds itself outside the band
   and snaps the page straight back. */
function keepInView(el) {
  if (!state.autoFollow) return;
  const r = el.getBoundingClientRect();
  const top = reader.getBoundingClientRect().top;
  const h = reader.clientHeight;
  const y = r.top - top;
  if (y < h * 0.25 || y > h * 0.72) {
    markProgrammaticScroll();
    reader.scrollBy({ top: y - h * 0.42, behavior: "smooth" });
  }
}

function wordAt(t) {
  const ws = state.words[state.i];
  if (!ws || !ws.length) return -1;
  let lo = 0, hi = ws.length - 1, best = 0;
  while (lo <= hi) {
    const mid = (lo + hi) >> 1;
    if (ws[mid].s <= t) { best = mid; lo = mid + 1; } else { hi = mid - 1; }
  }
  return best;
}

/* ------------------------------------------------------------ audio slots */

/* One <audio> element can only ever start the next paragraph *after* the
   current one has ended: fetch, decode and play() all happen after the
   "ended" event, and that is the stutter between paragraphs.

   So there are two. While block i plays in the active slot, block i + 1 is
   fetched into the standby slot and load()ed with preload="auto", so by the
   time the boundary arrives the browser has already decoded it and play() is
   a scheduling call rather than a network round trip.

   Being honest about the limit: this is a handoff on the "ended" event, so it
   costs one turn of the JavaScript event loop - inaudible in practice, but it
   is not sample-exact gaplessness and nothing here pretends otherwise. What
   it removes is the seconds-long hitch, not the milliseconds.

   Everything else in this file still talks about "the" audio element; it just
   asks activeEl() which one that currently is. */
const audioB = (() => {
  const el = document.createElement("audio");
  el.id = "audioB";
  el.preload = "auto";
  audio.parentNode.insertBefore(el, audio.nextSibling);
  return el;
})();

const slots = {
  A: { el: audio, i: -1, voice: null, url: null, epoch: -1 },
  B: { el: audioB, i: -1, voice: null, url: null, epoch: -1 },
};
const active = () => slots[state.activeSlot];
const standby = () => slots[state.activeSlot === "A" ? "B" : "A"];
const activeEl = () => active().el;

/* Empties a slot without touching the object URL it was pointing at - the
   URL belongs to state.audioUrls, and seeking back to a block that is still
   cached must not find its blob revoked out from under it. */
function releaseSlot(s) {
  try { s.el.pause(); } catch {}
  s.el.removeAttribute("src");
  try { s.el.load(); } catch {}
  s.i = -1; s.voice = null; s.url = null; s.epoch = -1;
}

/* "Nothing is holding audio for this document any more", which since the
   gapless path exists means the scheduled graph as well as the two elements.
   Every caller of this means both: a document being replaced, or a paused
   voice switch that must leave Play with nothing to resume. Missing one of
   them would leave a node in the audio thread's hands that no bookkeeping
   here can still reach. */
function releaseAllSlots() {
  waStopAll();
  releaseSlot(slots.A);
  releaseSlot(slots.B);
}

/* Idempotent on purpose: re-arming a slot that already holds exactly this
   clip must not reset src, because that throws away the buffering this whole
   mechanism exists to build up. */
function armSlot(s, i, url, voice, epoch) {
  if (s.url === url && s.i === i && s.voice === voice) { s.epoch = epoch; return; }
  s.i = i; s.voice = voice; s.url = url; s.epoch = epoch;
  s.el.preload = "auto";
  s.el.src = url;
  s.el.playbackRate = state.speed || 1;
  try { s.el.load(); } catch {}
}

const voiceParam = (v) => (v ? `&voice=${encodeURIComponent(v)}` : "");

/* Read ahead, nobody waiting. Everything the player fetches for a block it is
   not yet on carries this: it is what puts the request in the server's
   prefetch tier and what lets a voice switch abandon it. */
const SPECULATIVE = { speculative: true };

/* Both halves of block i name the same synthesis. A server too old to report
   the key says nothing, and nothing is asserted about it - this can only ever
   reject a pair it has positive evidence against. */
const keysAgree = (i) => !state.blockKeys[i] || !state.audioKeys[i]
                         || state.blockKeys[i] === state.audioKeys[i];

/* True for any block a pending voice switch has already invalidated.

   The block that is *sounding* when the switch is asked for keeps its old
   voice and is allowed to finish - the server keeps its audio for exactly
   that reason. Everything after it belongs to the new voice, from the moment
   the user clicks and not from the moment the server answers. */
const beyondVoiceBarrier = (i) =>
  !!state.pendingVoice && i > state.pendingVoice.barrier;

const slotHolds = (s, i) =>
  s.i === i && s.voice === state.voice && !!s.url && keysAgree(i)
  && !beyondVoiceBarrier(i);

/* Ready enough to start without stalling. canplaythrough is the ideal signal
   and some browsers never fire it, so HAVE_CURRENT_DATA is accepted too -
   waiting for a promise that may never resolve would reintroduce exactly the
   gap this is here to remove. */
const slotReady = (s) => !!s.url && s.el.readyState >= 2;

/* Get block `next` ready to follow the one now sounding, while it is still
   sounding. Nothing here ever touches the active slot.

   Two ways to be ready, and this asks for the better one first: if the graph
   below is driving, `next` is *scheduled* - handed to the audio thread with
   the exact context time block N runs out, so the join is sample-exact and no
   main-thread event is involved in it at all. If the graph is not driving, or
   this clip will not decode, the standby element is armed exactly as before
   and the boundary costs its old event-loop turn. Both are "ready"; one is
   simply better, and the reader cannot tell which one it got except by
   listening very carefully at a paragraph end. */
async function primeStandby(next) {
  if (next < 0 || next >= state.blocks.length) { releaseSlot(standby()); return; }
  /* Never fill the standby with a paragraph the user has already switched
     away from. This is the one that made the switch audible a paragraph
     late: the standby was armed with old-voice audio before the POST
     resolved, and the boundary handed straight off to it. */
  if (beyondVoiceBarrier(next)) {
    releaseSlot(standby());
    waDropAfter(next - 1);
    return;
  }
  if (waScheduled(next)) return;
  if (!wa.on && slotHolds(standby(), next)) return;
  const epoch = state.playbackEpoch, v = state.voice, doc = state.docId;
  const pending = state.pendingVoice;
  const spent = () => epoch !== state.playbackEpoch || v !== state.voice
                      || doc !== state.docId || next !== state.i + 1
                      || state.pendingVoice !== pending
                      || beyondVoiceBarrier(next);
  try {
    if (!await loadBlock(next, SPECULATIVE) || spent()) return;
    const url = await loadAudio(next, SPECULATIVE);
    if (!url || spent()) return;
    /* Decode before the last check, schedule after it and without an await in
       between: the schedule has to be the same synchronous run as the guard,
       or a boundary that passes while a decode is outstanding could queue a
       second copy of a block that is already sounding. */
    if (wa.on) {
      const buffer = await ensureBuffer(next);
      if (spent()) return;
      if (buffer && waScheduleNext(next, buffer)) {
        /* Scheduled, so the standby element has no job here - and an element
           left armed with this block would be holding a second decoded copy
           of audio the graph is already holding. Nothing is lost by dropping
           it: if the node is later stopped (a voice switch), advancePastEnd
           reaches playBlock exactly as it does for a block that was never
           ready. */
        releaseSlot(standby());
        return;
      }
    }
    if (spent()) return;
    armSlot(standby(), next, url, v, epoch);
  } catch { /* the standby simply stays empty; onEnded falls back to waiting */ }
}

function markHere(i, { scroll = true } = {}) {
  document.querySelectorAll(".blk.here").forEach((e) => e.classList.remove("here"));
  const here = $(`b-${i}`);
  if (here) {
    here.classList.add("here");
    if (scroll && state.autoFollow) {
      markProgrammaticScroll();
      here.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }
}

/* The boundary. Called from "ended" on the active element, with the standby
   already holding the next block: flip which slot is active, start it, and
   immediately begin filling the retired slot with the block after that. */
function handoff(s, next) {
  const retiring = active();
  /* An element is about to speak, so the graph is not. Normally already true -
     this is reached when the chain ran dry - but stated rather than assumed:
     the one failure neither path recovers from is both of them sounding. */
  waStopAll();
  state.activeSlot = state.activeSlot === "A" ? "B" : "A";
  clearHighlight();
  state.i = next;
  state.wi = -1;
  s.el.playbackRate = state.speed || 1;
  try { s.el.currentTime = 0; } catch {}
  s.el.play().catch(() => {});
  releaseSlot(retiring);
  markHere(next);
  syncPdfToCurrentBlock();
  updateFigureForBlock();
  status(`${next + 1} of ${state.blocks.length}`);
  heartbeat(true);
  primeStandby(next + 1);
  prefetch(next + 1);
  checkSleep();
}

function onSlotEnded(el) {
  if (el !== activeEl()) return;     // a retired slot finishing; not our business
  advancePastEnd();
}

/* The paragraph that was sounding has finished and nothing is playing now.

   Shared by both paths deliberately: the slots reach it from the element's
   "ended" event, and the scheduled graph reaches it when its chain runs dry -
   which is the same situation described from the other side, and every branch
   below (end of the book, a book still being scanned, an unresolved voice
   switch, a block that is still synthesizing) is decided by state that has
   nothing to do with how the last block was played. Two copies of this would
   be two chances for one of them to forget the scanned-book case. */
function advancePastEnd() {
  heartbeat(true);
  if (!state.playing) return;
  const next = state.i + 1;
  if (next >= state.blocks.length) {
    /* The end of what exists is not the end of the book while a scan is
       still running. Stay in the playing state and wait for the next page. */
    if (state.ingestion && state.ingestion.state === "running") {
      waitForMoreText();
      return;
    }
    stop();
    return;
  }
  if (beyondVoiceBarrier(next)) {
    /* The paragraph that was sounding when the switch was asked for has just
       finished, and the server has not answered yet. Wait: handing off to
       anything now would speak the voice the user just left. state.playing
       stays true, so the apply path picks it up from here. */
    state.pendingVoice.resumeAt = next;
    status("switching voice…");
    return;
  }
  const s = standby();
  if (slotHolds(s, next) && slotReady(s)) { handoff(s, next); return; }
  /* The next block is still synthesizing. Wait for it in a clean state -
     never replay what is already in a slot, which after a voice switch would
     mean speaking in the voice the user just moved away from. */
  status(`waiting for ${next + 1} of ${state.blocks.length}…`);
  playBlock(next);
}

function onSlotTime(el) {
  if (el !== activeEl() || state.i < 0) return;
  const wi = wordAt(el.currentTime);
  if (wi !== state.wi) { highlight(wi); updateFigureForWord(wi); }
  heartbeat();
}

for (const s of [slots.A, slots.B]) {
  s.el.addEventListener("timeupdate", () => onSlotTime(s.el));
  s.el.addEventListener("ended", () => onSlotEnded(s.el));
}

/* -------------------------------------------------- gapless: the Web Audio path

   The two elements above removed the seconds-long hitch at a paragraph
   boundary. They cannot remove the last few milliseconds of it, and the
   comment up there says so: the handoff happens on the "ended" event, so it
   costs a turn of the JavaScript event loop plus whatever the element's own
   pipeline adds on play() - 15-50ms, and it lands as a click on a book read
   end to end, because two independently synthesized clips almost never meet
   at the same sample value.

   Web Audio can do better, because a source node is *scheduled* rather than
   started: block N+1 is told to begin at exactly the context time block N
   runs out, the audio thread stitches the two, and no main-thread event is
   involved in the join at all. That is sample-exact, which the slots never
   were and never claimed to be.

   It is a second path, not a replacement, and that is the whole design. Every
   function here is allowed to return false, return null, or throw, and when
   one does the player is on the elements again with no state to unwind - no
   AudioContext (an old browser, a hardened one, no audio device), a clip
   decodeAudioData refuses, a context the autoplay policy will not resume out
   of "suspended": each of those is a reader who hears exactly what they heard
   before, gap and all. Which is the project's main rule applied to the thing
   most likely to break: if the enhancement fails, the reader behaves as if it
   did not exist. Nothing on the reading path may check whether it worked.

   The one thing that is *not* optional: the two paths must never sound at
   once. waBegin() empties both elements before it schedules anything, and
   releaseAllSlots() stops the graph - so whichever side is speaking, the
   other side is holding nothing. */

/* 6ms. Long enough to swallow the DC step where two clips meet - the click
   this section exists for - and far too short to touch a phoneme: the seam
   between two paragraphs is trailing silence against leading silence in every
   voice Kokoro has. Blocks still abut exactly (startAt(N+1) === endAt(N)), so
   nothing is shortened and no word timing moves; what the ramps remove is the
   step, not the join. */
const WA_RAMP = 0.006;

/* Below this a clip would be fading in and straight back out with no middle,
   so it is left at full gain instead. A block that is nothing but a citation
   returns brief silence, and this is that block. */
const WA_MIN_RAMPABLE = WA_RAMP * 4;

/* Finer than a word and cheap. The elements' own "timeupdate" fires about
   four times a second; this is the highlighter getting *better*, not merely
   different. Nothing about the audio depends on it - see waTick(). */
const WA_TICK_MS = 40;

let audioCtx = null;
let waMaster = null;
let audioCtxBroken = false;      // constructing it failed once; stop trying

/* What is scheduled, in play order. One entry per block:
     {i, voice, epoch, src, gain, buffer, offset, startAt, endAt}
   startAt and endAt are AudioContext clock seconds, and startAt of the block
   now sounding is the zero the highlighter measures from. The list is short -
   the block sounding plus the one queued behind it - and always contiguous in
   block index, because a paragraph boundary has no other shape. */
const wa = {
  on: false,          // the graph is driving playback right now
  rate: 1,            // playbackRate the scheduled nodes were built at
  queue: [],
  offset: 0,          // media seconds into wa.pausedAt, while paused
  paused: false,
  pausedAt: -1,
  timer: null,
  /* The last position the graph reported. The boundary heartbeat fires at the
     instant a block ends, when there is no sounding node left to ask, and
     "block i, zero seconds in" saved there would make a reopened book replay
     the paragraph it had just finished. */
  lastPos: { i: -1, within: 0 },
};

/* Lazily, and only ever from inside a user gesture. A context constructed
   anywhere else is created "suspended", resume() outside a gesture is refused,
   and a suspended context plays perfect silence while reporting nothing at
   all - which would look like the engine having gone quiet rather than like a
   policy. Hence the arming listeners below. */
function getAudioContext() {
  if (audioCtx) {
    if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
    return audioCtx;
  }
  if (audioCtxBroken) return null;
  const Ctor = window.AudioContext || window.webkitAudioContext;
  if (!Ctor) { audioCtxBroken = true; return null; }
  try {
    const ctx = new Ctor();
    const master = ctx.createGain();
    master.gain.value = 1;
    master.connect(ctx.destination);
    audioCtx = ctx;
    waMaster = master;
    ctx.onstatechange = () => {
      if (ctx.state === "suspended" && state.playing) {
        ctx.resume().catch(() => {});
      }
    };
  } catch {
    audioCtxBroken = true;
    audioCtx = null;
    waMaster = null;
    return null;
  }
  if (audioCtx.state === "suspended") audioCtx.resume().catch(() => {});
  return audioCtx;
}

/* Armed from the gesture itself rather than from each playback path, because
   the policy is satisfied by a real user event and not by an intention: by the
   time playBlock() gets to its first await it is no longer inside one. Capture
   phase, so this runs before whatever the click was actually for, and once -
   getAudioContext() is idempotent. A context the reader never uses costs one
   idle audio graph, which is cheaper than the alternative of guessing which
   handlers count as gestures and missing one. */
for (const ev of ["pointerdown", "keydown", "touchend"]) {
  window.addEventListener(ev, () => { getAudioContext(); },
                          { capture: true, passive: true });
}

/* Resume AudioContext when returning from a background tab or window suspension
   if playback was actively sounding. */
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && state.playing && audioCtx) {
    if (audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
  }
});

/* Running this instant, with nothing awaited - the synchronous half of
   waReady(), for the two callers that cannot await one. A context that exists
   but is suspended is the dangerous case: decodeAudioData still works on it and
   src.start() still queues without error, so everything looks like success
   while currentTime never advances. Treating that as playable is how a reader
   ends up silent with "Pause" showing, which is the worst outcome available
   here - it has no path back, because the elements have been emptied. */
const waLive = () => !!audioCtx && !!waMaster && audioCtx.state === "running";

/* Usable *now*, awaiting a resume if one is outstanding. Called from
   playBlock, which is already async and is downstream of a gesture, so this
   is the one place the resume can actually be waited for. A context that will
   not leave "suspended" is reported as unusable and the elements take it. */
async function waReady() {
  if (!audioCtx || !waMaster) return false;
  if (audioCtx.state === "suspended") {
    try { await audioCtx.resume(); } catch { return false; }
  }
  return !!audioCtx && audioCtx.state === "running";
}

/* Both spellings. Older WebKit's decodeAudioData takes callbacks and returns
   undefined; everything current returns a promise. */
function decodeAudio(ctx, raw) {
  return new Promise((resolve, reject) => {
    let ret;
    try { ret = ctx.decodeAudioData(raw, resolve, reject); } catch (e) { reject(e); return; }
    if (ret && typeof ret.then === "function") ret.then(resolve, reject);
  });
}

/* The decoded form of block i's audio, or null - and null is a perfectly
   ordinary answer, meaning "play this one on an element".

   Decoded from the object URL the fetch already produced rather than from a
   second request to the server: reading a blob: URL is a copy inside the
   browser, so this costs no network, no second synthesis, and - the part
   that matters most - the audio still comes from the one synth()
   call its timings came from. A second /api/audio could be served by a
   different synthesis and nothing downstream could tell.

   decodeAudioData detaches the ArrayBuffer it is handed, which is why it is
   handed a fresh one read back from the blob and never anything the fallback
   path might still want. */
function ensureBuffer(i) {
  if (!audioCtx) return Promise.resolve(null);
  const v = state.voice;
  if (state.audioBuffers[i] && state.bufferVoices[i] === v
      && state.bufferKeys[i] === state.audioKeys[i]) {
    return Promise.resolve(state.audioBuffers[i]);
  }
  const doc = state.docId;
  const prior = state.bufferInflight[i];
  if (prior && prior.voice === v && prior.doc === doc) return prior.promise;
  const url = state.audioUrls[i];
  if (!url || state.audioVoices[i] !== v) return Promise.resolve(null);
  const record = { voice: v, doc };
  record.promise = (async () => {
    try {
      const raw = await (await fetch(url)).arrayBuffer();
      const buffer = await decodeAudio(audioCtx, raw);
      /* Arrived into a document or a voice we have since left. Same rule as
         loadAudio: dropped rather than cached, so nothing can pair it with
         timings that belong to a different synthesis. */
      if (state.docId !== doc || state.voice !== v) return null;
      state.audioBuffers[i] = buffer;
      state.bufferVoices[i] = v;
      state.bufferKeys[i] = state.audioKeys[i];
      return buffer;
    } catch {
      /* Undecodable, or the blob was revoked under us. Neither is an error the
         reader should ever see: an element plays the same bytes. */
      return null;
    } finally {
      if (state.bufferInflight[i] === record) delete state.bufferInflight[i];
    }
  })();
  state.bufferInflight[i] = record;
  return record.promise;
}

/* Freeze an automated param at whatever it is right now, so a ramp can be
   rebuilt from here without the jump that cancelScheduledValues alone leaves. */
function waHold(param, t) {
  try {
    if (param.cancelAndHoldAtTime) { param.cancelAndHoldAtTime(t); return; }
    const v = param.value;
    param.cancelScheduledValues(t);
    param.setValueAtTime(v, t);
  } catch { try { param.cancelScheduledValues(t); } catch {} }
}

/* Up at the start, down at the end, full gain in between. Two of these
   back to back at an abutting boundary make a ~12ms dip across the seam -
   the micro-crossfade - rather than the discontinuity that clicks. The
   fade-in is also what makes seeking and resuming mid-word silent. */
function waEnvelope(e) {
  const g = e.gain.gain;
  try {
    if (e.endAt - e.startAt <= WA_MIN_RAMPABLE) { g.setValueAtTime(1, e.startAt); return; }
    g.setValueAtTime(0, e.startAt);
    g.linearRampToValueAtTime(1, e.startAt + WA_RAMP);
    g.setValueAtTime(1, e.endAt - WA_RAMP);
    g.linearRampToValueAtTime(0, e.endAt);
  } catch {}
}

/* The same envelope re-hung on a block whose end has moved - a speed change
   mid-paragraph, and nothing else. */
function waRearmEnvelope(e, t) {
  const g = e.gain.gain;
  waHold(g, t);
  try {
    if (e.endAt - t <= WA_MIN_RAMPABLE) {
      /* Almost nothing left to play, and quite possibly already part-way down
         its own fade-out - so ride whatever waHold froze it at down to zero,
         rather than snapping back to full gain, which at the bottom of a fade
         is a click followed by an unramped cutoff. */
      g.linearRampToValueAtTime(0, Math.max(e.endAt, t + 0.001));
      return;
    }
    g.linearRampToValueAtTime(1, t + WA_RAMP);
    g.setValueAtTime(1, e.endAt - WA_RAMP);
    g.linearRampToValueAtTime(0, e.endAt);
  } catch {}
}

/* Builds and starts one scheduled block. Returns the entry, or null if the
   graph refused it - in which case the caller has an element to fall back on
   and this never happened. */
function waStart(i, buffer, startAt, offset) {
  if (!audioCtx || !waMaster) return null;
  const off = Math.min(Math.max(0, offset || 0), Math.max(0, buffer.duration - 0.01));
  let src, gain;
  try {
    src = audioCtx.createBufferSource();
    src.buffer = buffer;
    src.playbackRate.value = wa.rate;
    gain = audioCtx.createGain();
    src.connect(gain);
    gain.connect(waMaster);
  } catch { return null; }
  const e = {
    i, voice: state.voice, epoch: state.playbackEpoch,
    src, gain, buffer, offset: off, startAt,
    endAt: startAt + Math.max(0, buffer.duration - off) / wa.rate,
  };
  waEnvelope(e);
  src.onended = () => waEnded(e);
  try {
    src.start(startAt, off);
  } catch {
    try { src.disconnect(); gain.disconnect(); } catch {}
    return null;
  }
  return e;
}

const waScheduled = (i) => wa.queue.some((e) => e.i === i);
const waHead = () => (wa.queue.length ? wa.queue[wa.queue.length - 1] : null);

/* Decoded audio is PCM held in RAM - a few megabytes for a 17-second block,
   against about one for the WAV it was decoded from - so unlike the object
   URLs beside it, this cannot be allowed to accumulate for the length of a
   book. A window around the reading position is kept and the rest goes; a
   seek back into a dropped block decodes it again from the blob, which was
   never released, so that costs milliseconds and no synthesis. Nothing the
   audio thread is already holding is dropped, and nor is the block a paused
   reader is sitting on - that one is what Play resumes from. */
const WA_KEEP_BACK = 2, WA_KEEP_AHEAD = 3;
const URL_KEEP_BACK = 8, URL_KEEP_AHEAD = 16;

function trimAudioUrls(at) {
  for (const k of Object.keys(state.audioUrls)) {
    const i = +k;
    if (i >= at - URL_KEEP_BACK && i <= at + URL_KEEP_AHEAD) continue;
    if (i === state.i || waScheduled(i) || (wa.paused && wa.pausedAt === i)) continue;
    if (slots.A && slots.A.i === i) continue;
    if (slots.B && slots.B.i === i) continue;
    if (state.audioInflight && state.audioInflight[i]) continue;
    if (state.bufferInflight && state.bufferInflight[i]) continue;
    const u = state.audioUrls[k];
    if (u) URL.revokeObjectURL(u);
    delete state.audioUrls[k];
    delete state.audioVoices[k];
    delete state.audioKeys[k];
  }
}

function waTrimBuffers(at) {
  for (const k of Object.keys(state.audioBuffers)) {
    const i = +k;
    if (i >= at - WA_KEEP_BACK && i <= at + WA_KEEP_AHEAD) continue;
    if (waScheduled(i) || (wa.paused && wa.pausedAt === i)) continue;
    delete state.audioBuffers[k];
    delete state.bufferVoices[k];
    delete state.bufferKeys[k];
  }
  trimAudioUrls(at);
}

/* Media seconds into this block's own audio - the number word timings are in,
   and the number /api/position stores. */
const waWithin = (e) => Math.min(
  Math.max(0, (audioCtx.currentTime - e.startAt) * wa.rate + e.offset),
  e.buffer ? e.buffer.duration : 0);

/* Which block is sounding, decided by the clock rather than by bookkeeping:
   the last entry that has started. Nothing leaves the queue until its node has
   reported ending, so this cannot fall between two blocks - which is exactly
   what it is for, since the bookkeeping is what it is used to correct. */
function waSounding() {
  if (!audioCtx) return null;
  const t = audioCtx.currentTime;
  let cur = null;
  for (const e of wa.queue) if (e.startAt <= t) cur = e;
  return cur;
}

/* Ramped down, not cut. Stopping a source mid-waveform is the same click this
   section exists to remove, and a seek, a pause and a voice switch would each
   have produced one. The node is disconnected from its own "ended" rather
   than immediately, because it still has a ramp left to play. */
function waStopEntry(e) {
  const t = audioCtx ? audioCtx.currentTime : 0;
  try { e.src.onended = null; } catch {}
  waHold(e.gain.gain, t);
  try { e.gain.gain.linearRampToValueAtTime(0, t + WA_RAMP); } catch {}
  const drop = () => { try { e.src.disconnect(); e.gain.disconnect(); } catch {} };
  try { e.src.addEventListener("ended", drop, { once: true }); } catch {}
  try { e.src.stop(t + WA_RAMP); } catch { try { e.src.stop(); } catch {} drop(); }
  /* And unconditionally, a little later. A node told to stop at or before its
     own start time never plays, and a node that never plays is not guaranteed
     to report ending - so the "ended" listener above cannot be the only thing
     that disconnects it, or a voice switch that catches a block before it
     started would leave the node (and the megabytes of PCM it holds) attached
     to the graph forever. disconnect() twice is a no-op, so both may fire. */
  setTimeout(drop, 250);
}

/* Everything, now. wa.paused goes with it: a torn-down graph has no position
   left to resume from, and Play must find that out here rather than by
   starting silence. */
function waStopAll() {
  for (const e of wa.queue) waStopEntry(e);
  wa.queue = [];
  wa.on = false;
  wa.paused = false;
  wa.pausedAt = -1;
  waStopTimer();
}

/* Everything scheduled past `afterIndex`; whatever is sounding stays. This is
   the voice switch's lever on the graph - the paragraph now sounding keeps the
   voice it was made in, and every future node belongs to the new one. A
   scheduled node is already in the audio thread's hands, so stopping it is the
   only way to take it back. */
function waDropAfter(afterIndex) {
  /* Never the block that is sounding, even if it is past the index. Stopping
     it here would leave wa.on true with an empty queue and no node left to
     report ending - silence, with "Pause" showing and nothing to restart it.
     It is also never the right thing to want: the barrier exists precisely to
     let the sounding paragraph finish in the voice it was made in, and every
     caller passes an index at or after it. Anything that really means "stop
     everything" calls waStopAll(). */
  const cur = waSounding();
  const keep = [];
  for (const e of wa.queue) {
    if (e.i > afterIndex && e !== cur) waStopEntry(e); else keep.push(e);
  }
  wa.queue = keep;
  if (wa.paused && wa.pausedAt > afterIndex) { wa.paused = false; wa.pausedAt = -1; }
}

/* Starts block i under the graph, `offset` media-seconds in, and takes the
   elements out of the picture. Whatever was scheduled goes first: this is a
   deliberate start, and anything queued was queued to follow a block the
   reader is no longer on. */
function waBegin(i, buffer, offset) {
  const ctx = getAudioContext();
  if (!ctx) return false;
  waStopAll();
  wa.rate = state.speed || 1;
  /* One ramp into the future, not currentTime: a source told to start in the
     past starts at once and skips its fade-in, which is the click on a seek. */
  const e = waStart(i, buffer, ctx.currentTime + WA_RAMP, offset);
  if (!e) return false;
  /* Nothing in an element while the graph sounds. Two paths playing at once is
     the one failure neither of them can recover from. */
  releaseSlot(slots.A);
  releaseSlot(slots.B);
  wa.queue = [e];
  wa.on = true;
  wa.offset = 0;
  wa.lastPos = { i, within: e.offset };
  waTrimBuffers(i);
  waStartTimer();
  return true;
}

/* The gapless join. `startAt` is the sounding block's own endAt, which is
   startAt(N) + duration(N) - the whole point of the section. Refuses anything
   that is not the block immediately behind the last one scheduled, so a
   duplicate or an out-of-order call cannot put two nodes on the same
   paragraph; and refuses to schedule against a head that has already run out,
   because that is not a join, it is a gap, and advancePastEnd() owns it. */
function waScheduleNext(i, buffer) {
  if (!wa.on || !audioCtx) return false;
  /* Checked here as well as at the caller, for the reason loadBlock gives for
     doing the same thing: a guard at the callers alone is a guard somebody
     adds a caller past. This one would speak the voice the user just left. */
  if (beyondVoiceBarrier(i) || !keysAgree(i)) return false;
  const head = waHead();
  if (!head || head.i !== i - 1) return false;
  /* One ramp of margin, not zero. A start time that elapses between this call
     and the node actually starting is a start in the past: the fade-in is
     skipped and the first phoneme clipped, which is the click this section
     exists to remove, reintroduced at the one boundary where synthesis
     finished at the wire. Refusing hands that boundary to advancePastEnd,
     which is a short gap - the thing this replaced, not the thing it fixed. */
  if (head.endAt <= audioCtx.currentTime + WA_RAMP) return false;
  const e = waStart(i, buffer, head.endAt, 0);
  if (!e) return false;
  wa.queue.push(e);
  return true;
}

/* Pause, in a graph that has no pause. Suspending the context would suspend
   the voice previews with it, so the chain is torn down and where it had got
   to is remembered; waResume() rebuilds it from there, and the fade-in on
   every start is what stops that being audible mid-syllable. Returns false
   when the graph was not driving, so callers pause the element instead. */
function waPause() {
  if (!wa.on) return false;
  /* Or the head, if nothing has started yet: waBegin schedules one ramp into
     the future, and a reader who clicks a word 25 seconds in and then pauses
     inside those 6ms must not have the seek thrown away. A node that has not
     started has spoken none of its clip, so its position is the offset it was
     about to start from. */
  const cur = waSounding() || wa.queue[0] || null;
  const at = cur ? cur.i : state.i;
  const within = !cur ? 0
    : (cur.startAt > audioCtx.currentTime ? cur.offset : waWithin(cur));
  waStopAll();
  wa.offset = within;
  wa.pausedAt = at;
  wa.paused = true;
  wa.lastPos = { i: at, within };
  return true;
}

/* True when Play can pick up exactly where Pause left off - same block, same
   voice, buffer still cached, and not on the far side of a voice barrier. */
const waCanResume = () =>
  wa.paused && wa.pausedAt === state.i && state.i >= 0 && waLive()
  && !!state.audioBuffers[state.i] && state.bufferVoices[state.i] === state.voice
  && state.bufferKeys[state.i] === state.audioKeys[state.i]
  && keysAgree(state.i) && !beyondVoiceBarrier(state.i);

function waResume() {
  if (!waCanResume()) return false;
  return waBegin(state.i, state.audioBuffers[state.i], wa.offset);
}

/* Speed, without a seam. An element retimes itself; a scheduled chain has to
   be re-anchored - the block sounding takes the new rate on its own
   playbackRate param, its end moves, and everything queued behind it is torn
   down and re-scheduled against that new end (primeStandby does the
   re-scheduling, from the buffer it already has, so it costs nothing).
   Re-anchoring startAt so that (now - startAt) * rate is still the offset
   already spoken is what keeps the highlighter honest across the change. */
function waRetime(v) {
  /* The head rather than only the sounding node, so that a speed change inside
     waBegin's 6ms lead-in retimes the node it is about to start. Leaving it at
     the old rate while wa.rate moved would have the audio at one speed and the
     highlighter walking at another - the drift would grow all block. */
  const cur = wa.on && audioCtx ? (waSounding() || wa.queue[0] || null) : null;
  if (!cur) { wa.rate = v; return; }
  const t = audioCtx.currentTime;
  const running = cur.startAt <= t;
  const spoken = running ? Math.max(0, (t - cur.startAt) * wa.rate) : 0;
  wa.rate = v;
  for (const e of wa.queue) if (e !== cur) waStopEntry(e);
  wa.queue = [cur];
  try { cur.src.playbackRate.value = v; } catch {}
  /* A node already running is re-anchored so that (now - startAt) * rate still
     comes to what it has spoken. One that has not started keeps the start it
     was given: moving it up to now would throw away the lead-in its fade-in
     needs, which is the click all over again. */
  if (running) cur.startAt = t - spoken / v;
  cur.endAt = cur.startAt + Math.max(0, cur.buffer.duration - cur.offset) / v;
  if (running) waRearmEnvelope(cur, t);
  else { waHold(cur.gain.gain, t); waEnvelope(cur); }
  primeStandby(cur.i + 1);
}

function waStartTimer() {
  if (wa.timer) return;
  wa.timer = setInterval(waTick, WA_TICK_MS);
}

function waStopTimer() {
  if (wa.timer) clearInterval(wa.timer);
  wa.timer = null;
}

/* The highlighter's clock. A scheduled chain has no "timeupdate" and the
   elements' currentTime means nothing here, so the position is read off the
   context clock: (now - startAt) * rate + offset is exactly how far into this
   block's own audio the speaker is, and it is monotonic by construction
   because it is one subtraction from a clock that only goes forward.

   Nothing about the audio depends on this timer. Block N+1 was handed to the
   audio thread when N started, so a throttled background tab, or a tick that
   never runs at all, changes when the *highlight* moves and never when the
   sound does. */
function waTick() {
  if (!wa.on || !audioCtx) return;
  const cur = waSounding();
  if (!cur) return;
  if (cur.i !== state.i) waCrossedInto(cur);
  if (state.i < 0) return;
  const within = waWithin(cur);
  wa.lastPos = { i: cur.i, within };
  const wi = wordAt(within);
  if (wi !== state.wi) { highlight(wi); updateFigureForWord(wi); }
  heartbeat();
}

/* The boundary, from the bookkeeping side. Note what is *not* here: no play()
   and no element. The sound crossed over inside the audio thread at a sample
   boundary before this ran; everything below is only what the reader sees.
   Otherwise it is handoff(), branch for branch, and for the same reasons -
   including not touching autoFollow, because an automatic advance must not
   undo a manual scroll made to read ahead. */
function waCrossedInto(e) {
  clearHighlight();
  state.i = e.i;
  state.wi = -1;
  markHere(e.i);
  syncPdfToCurrentBlock();
  updateFigureForBlock();
  status(`${e.i + 1} of ${state.blocks.length}`);
  heartbeat(true);
  primeStandby(e.i + 1);
  prefetch(e.i + 1);
  waTrimBuffers(e.i);
  checkSleep();
}

/* A scheduled node has run out. If the next block was already scheduled the
   audio thread has long since stitched them and there is nothing to do here
   but let the bookkeeping catch up. If it was not, the chain has run dry, and
   that is the same event the elements report as "ended" - so it goes to the
   same place. */
function waEnded(e) {
  const at = wa.queue.indexOf(e);
  if (at === -1) return;              // stopped deliberately; already accounted for
  wa.queue.splice(at, 1);
  try { e.src.disconnect(); e.gain.disconnect(); } catch {}
  if (!wa.on) return;
  wa.lastPos = { i: e.i, within: e.buffer ? e.buffer.duration : 0 };
  if (wa.queue.length) { waTick(); return; }
  wa.on = false;
  waStopTimer();
  advancePastEnd();
}

/* How far into the block now sounding, whoever is sounding it. The saved
   reading position is this number, so it has to be right at three awkward
   moments: while the graph plays (the context clock), while it is paused (the
   offset Pause kept), and at the instant a block ends and there is no sounding
   node left to ask (lastPos - "block i, zero seconds in" saved there would
   make a reopened book replay the paragraph it had just finished). Anything
   else is an element, and the element's own currentTime is the answer. */
function currentWithin() {
  if (wa.paused && wa.pausedAt === state.i) return wa.offset || 0;
  if (wa.on && audioCtx) {
    const cur = waSounding();
    if (cur) return waWithin(cur);
  }
  if (wa.lastPos.i === state.i && !active().url) return wa.lastPos.within;
  return activeEl().currentTime || 0;
}

/* --------------------------------------------------------------- loading */

/* Speculative requests say so. The server queues those a tier below the block
   somebody is actually waiting on, so clicking a word four pages down is not
   served behind the two blocks the player had just decided to read ahead -
   see _requested_priority in talkify.py. Omitted on anything the user is
   waiting for, which is what an older cached copy of this file sends for
   everything. */
const prefetchParam = (speculative) => (speculative ? "&prefetch=1" : "");

/* Every in-flight request is registered so a second caller joins it instead of
   opening its own. `pinned` is the one thing a joiner changes: a speculative
   request that somebody has since started waiting on may no longer be
   abandoned, which is what keeps abortStale() from cancelling the paragraph
   the player is about to speak. */
function joinInflight(table, i, doc, voice, speculative) {
  const prior = table[i];
  if (prior && prior.doc === doc && prior.voice === voice) {
    if (!speculative) prior.pinned = true;
    return prior;
  }
  return null;
}

function trackInflight(table, i, record, promise) {
  record.promise = promise;
  table[i] = record;
  const done = () => { if (table[i] === record) delete table[i]; };
  promise.then(done, done);
  return promise;
}

/* Drop speculative requests that can no longer be used: a voice switch has
   invalidated everything past `afterIndex`, or the reader has moved to another
   document entirely.

   Aborting does not stop the synthesis - the server thread is already blocked
   inside the engine and finds out only when it writes - and it is not meant
   to. What it frees is the browser's connection, of which there are six to
   this origin: a voice switch with three blocked requests in flight left the
   first request in the *new* voice queued in the browser behind audio nobody
   would ever hear, which is precisely the delay the switch is trying to
   avoid. What the server does finish stays in its cache, so nothing is
   wasted twice. */
function abortStale(afterIndex) {
  for (const table of [state.blockInflight, state.audioInflight]) {
    for (const k of Object.keys(table)) {
      const record = table[k];
      if (!record || record.pinned) continue;
      if (afterIndex !== undefined && +k <= afterIndex
          && record.doc === state.docId) continue;
      try { record.controller.abort(); } catch {}
      if (table[k] === record) delete table[k];
    }
  }
}

/* True for the one rejection this file expects and does not want reported:
   a fetch this player itself gave up on. */
const isAbort = (e) => !!e && (e.name === "AbortError" || e.code === 20);

async function loadBlock(i, { speculative = false } = {}) {
  const v = state.voice;
  const doc = state.docId;
  /* The barrier is checked here as well as at every caller, because this is
     the only place a request actually leaves for the server. A guard at the
     callers alone is a guard somebody adds a caller past. */
  if (beyondVoiceBarrier(i)) return false;
  if (state.words[i] && state.wordVoices[i] === v) return true;
  const joined = joinInflight(state.blockInflight, i, doc, v, speculative);
  if (joined) return joined.promise;
  const record = { doc, voice: v, pinned: !speculative,
                   controller: new AbortController() };
  /* The requested voice goes with the request. With two slots in flight the
     document's "current" voice at the moment the server handles this is not
     reliably the voice this slot is being loaded for. */
  return trackInflight(state.blockInflight, i, record, (async () => {
    let r, j;
    try {
      r = await fetch(
        `/api/block?doc=${doc}&i=${i}${voiceParam(v)}${prefetchParam(speculative)}`,
        { signal: record.controller.signal });
      j = await r.json();
    } catch (e) {
      /* Swallowed rather than rethrown: playBlock awaits this without a
         catch, and an abandoned prefetch must not surface as a failure on a
         paragraph the user did ask for. */
      if (!isAbort(e)) status("synthesis failed");
      return false;
    }
    if (!r.ok) { status(j.error || "synthesis failed"); return false; }
    if (state.voice !== v || state.docId !== doc) return false;   // switched while in flight
    if (beyondVoiceBarrier(i)) return false;   // ...or asked to, while this was in flight
    state.words[i] = j.words;
    state.wordVoices[i] = v;
    if (j.asset_key) state.blockKeys[i] = j.asset_key;
    return true;
  })());
}

/* Fetched as a blob and kept as an object URL rather than left to the browser's
   HTTP cache - the server sends Cache-Control: no-store on every response, on
   purpose, so a stale block index can't outlive the document that made it. */
async function loadAudio(i, { speculative = false } = {}) {
  const v = state.voice;
  const doc = state.docId;
  if (beyondVoiceBarrier(i)) return null;
  if (state.audioUrls[i] && state.audioVoices[i] === v) return state.audioUrls[i];
  const joined = joinInflight(state.audioInflight, i, doc, v, speculative);
  if (joined) return joined.promise;
  const record = { doc, voice: v, pinned: !speculative,
                   controller: new AbortController() };
  return trackInflight(state.audioInflight, i, record, (async () => {
    let response;
    try {
      response = await fetch(
        `/api/audio?doc=${doc}&i=${i}${voiceParam(v)}${prefetchParam(speculative)}`,
        { signal: record.controller.signal });
    } catch (e) {
      /* An abandoned prefetch is not a failure. Anything else still is, and
         still reaches playBlock's catch exactly as it did. */
      if (isAbort(e)) return null;
      throw e;
    }
    if (!response.ok) throw new Error("audio fetch failed");
    const key = response.headers.get("X-Talkify-Asset-Key");
    let blob;
    try {
      blob = await response.blob();
    } catch (e) {
      if (isAbort(e)) return null;
      throw e;
    }
    /* Arrived too late: this is the voice the user just switched away from.
       Drop it rather than caching it - callers treat null as nothing to play. */
    if (state.docId !== doc || state.voice !== v || beyondVoiceBarrier(i)) return null;
    const url = URL.createObjectURL(blob);
    if (state.audioUrls[i] && state.audioUrls[i] !== url) {
      URL.revokeObjectURL(state.audioUrls[i]);
    }
    state.audioUrls[i] = url;
    state.audioVoices[i] = v;
    if (key) state.audioKeys[i] = key;
    trimAudioUrls(i);
    return url;
  })());
}

/* Drops any client-side cached audio/timings for blocks after `afterIndex` -
   the mirror of the server invalidating its own cache on a voice switch
   (see talkify.py's _voice route). Without this the player would keep
   playing blobs fetched in the old voice for blocks it already prefetched. */
function invalidateAhead(afterIndex) {
  /* Before any URL is revoked: a slot still pointing at one of these blobs
     would be left holding a dead src. The standby is the usual case - it is
     one block ahead, which is exactly the range being purged. */
  for (const s of [slots.A, slots.B]) {
    if (s.i > afterIndex) releaseSlot(s);
  }
  /* And the same for the graph, which is the harder half: a node past the
     barrier is not merely "loaded", it is scheduled - the audio thread will
     play it at a known time whether or not anything here is still watching.
     Stopping it is the only way to take it back, and it has to happen here,
     with the URLs, rather than at the next boundary that will never come. */
  waDropAfter(afterIndex);
  /* Before the tables are cleared, not after: abortStale reads them to find
     what to let go of. */
  abortStale(afterIndex);
  Object.keys(state.audioUrls).forEach((k) => {
    if (+k > afterIndex) {
      URL.revokeObjectURL(state.audioUrls[k]);
      delete state.audioUrls[k];
      delete state.audioVoices[k];
      delete state.audioInflight[k];
    }
  });
  Object.keys(state.audioBuffers).forEach((k) => {
    if (+k > afterIndex) {
      delete state.audioBuffers[k];
      delete state.bufferVoices[k];
      delete state.bufferKeys[k];
    }
  });
  Object.keys(state.bufferInflight).forEach((k) => {
    if (+k > afterIndex) delete state.bufferInflight[k];
  });
  Object.keys(state.blockInflight).forEach((k) => {
    if (+k > afterIndex) delete state.blockInflight[k];
  });
  Object.keys(state.words).forEach((k) => {
    if (+k > afterIndex) {
      delete state.words[k];
      delete state.wordVoices[k];
      delete state.blockKeys[k];
      delete state.audioKeys[k];
    }
  });
}

/* Kept a couple of blocks ahead of playback, not just one - short blocks
   synthesize in a few seconds each on CPU, and a single block of lead time
   isn't always enough to outrun playback. The server-side background warmer
   (new_doc/_warm in talkify.py) is usually already further ahead than this
   by the time it matters; this just covers the gap right after opening. */
const PREFETCH_AHEAD = 2;

const prefetch = (i) => {
  for (let k = i; k < i + PREFETCH_AHEAD && k < state.blocks.length; k++) {
    if (beyondVoiceBarrier(k)) return;
    loadBlock(k, SPECULATIVE).catch(() => {});
    loadAudio(k, SPECULATIVE).catch(() => {});
  }
};

/* Bumped on every playBlock() call. A fast-forward (prev/next, or a second
   next before the first has finished loading) should never wait on the
   block it's leaving - a stale call's synthesis can keep running server-side
   (it gets cached either way) but must not be allowed to act on the player
   once a newer call has superseded it, e.g. suddenly starting playback on a
   paragraph the user already skipped past. */
let playToken = 0;

async function playBlock(i, seek = 0, { explicit = false } = {}) {
  if (i < 0 || i >= state.blocks.length) { stop(); return; }
  const token = ++playToken;
  /* Any deliberate start is a new epoch: whatever the standby was being
     filled with was chosen to follow a block we are no longer on. */
  const epoch = ++state.playbackEpoch;
  const doc = state.docId, v = state.voice;

  /* Re-sync to the reading position on any *explicit* jump (the user asked
     to go here) - but never on the automatic advance to the next block at
     the end of one (see the "ended" handler below), which must not override
     a manual scroll the user made mid-paragraph to read ahead. */
  if (explicit) state.autoFollow = true;

  /* The reading position moves right away, synthesized or not - that's what
     "next paragraph" means. Silence until the audio catches up is fine (the
     user's call); staying stuck on the old paragraph until the new one
     finishes synthesizing is not, and is what made the button feel locked. */
  /* The standby may already hold exactly this block - pressing Next during
     playback is precisely that case - in which case promote it rather than
     re-fetching and re-buffering what is already decoded. */
  if (!seek && slotHolds(standby(), i) && state.i !== i) {
    const retiring = active();
    state.activeSlot = state.activeSlot === "A" ? "B" : "A";
    releaseSlot(retiring);
  } else {
    releaseSlot(standby());
  }
  activeEl().pause();
  /* The graph, for the same reason and in the same breath: a deliberate start
     means what is sounding now stops now. Anything scheduled behind it was
     scheduled to follow a block the reader has just left. */
  waStopAll();
  clearHighlight();
  state.i = i;
  state.wi = -1;
  syncPdfToCurrentBlock();
  updateFigureForBlock();
  markHere(i);
  if (beyondVoiceBarrier(i)) {
    /* Next pressed while a switch is unresolved. Move the reading position -
       that is what the button means - but do not synthesize this paragraph in
       the voice being left. Remember it, and the apply path starts it. */
    state.pendingVoice.resumeAt = i;
    status("switching voice…");
    heartbeat(true);
    return;
  }
  status(`preparing ${i + 1} of ${state.blocks.length}…`);
  heartbeat(true);

  const ok = await loadBlock(i);
  if (token !== playToken) return;             // superseded by a newer skip while this was loading
  if (!ok) return;
  let audioUrl;
  try {
    audioUrl = await loadAudio(i);
  } catch {
    if (token === playToken) status("synthesis failed");
    return;
  }
  if (token !== playToken) return;
  if (!audioUrl) return;                       // fetched in a voice we have since left
  if (!keysAgree(i)) {
    /* Timings and audio keys disagreed - state.voice changed in the narrow
       window between the two awaits above, so neither half is safe to pair.
       Drop them and retry against whatever voice is current now, rather than
       leaving the player stuck on "preparing…" forever: a review caught
       that the old code just returned here with no recovery path at all. */
    delete state.words[i];
    delete state.audioUrls[i];
    delete state.audioBuffers[i];
    delete state.bufferVoices[i];
    delete state.bufferKeys[i];
    if (token === playToken) playBlock(i, seek);
    return;
  }

  /* Gapless if it can be. A decoded buffer can be scheduled sample-exactly
     against the block that follows it; an element cannot. Everything below is
     allowed to say no - no context, a suspended one, a clip that will not
     decode - and saying no lands on the element path underneath it, which is
     the same code, in the same order, that ran before any of this existed. */
  let started = false;
  /* These two awaits are the only ones in this function that sit *after* the
     audio URL has been validated, and that is a window the old code did not
     have: before, everything from here to play() was one synchronous run, so
     nothing could move the player between the check and the arming. Anything
     that has moved it since - another document, a confirmed voice switch, a
     newer epoch, a newer skip - means this URL must not be armed on an element
     either, so this returns instead of falling through to one.

     Wrapped as well as guarded, so that "the graph declined" and "the graph
     threw" are the same thing to the element path below. An exception escaping
     here would skip the fallback entirely, which is the one outcome the
     degrade-to-nothing rule does not allow. */
  const stale = () => token !== playToken || epoch !== state.playbackEpoch
                      || doc !== state.docId || v !== state.voice
                      || beyondVoiceBarrier(i);
  try {
    if (await waReady() && !stale()) {
      const buffer = await ensureBuffer(i);
      if (stale()) return;
      if (buffer) started = waBegin(i, buffer, seek);
    }
  } catch { started = false; }
  if (stale()) { if (started) waStopAll(); return; }
  if (!started) {
    waStopAll();
    armSlot(active(), i, audioUrl, state.voice, epoch);
    activeEl().playbackRate = state.speed || 1;
    if (seek) { try { activeEl().currentTime = seek; } catch {} }
    await activeEl().play();
    if (token !== playToken) { activeEl().pause(); return; }   // superseded again while play() was resolving
  }
  state.playing = true;
  $("playBtn").textContent = "Pause";
  status(`${i + 1} of ${state.blocks.length}`);
  /* Fill the other slot now, not at the boundary - having it decoded before
     this paragraph ends is the entire point. */
  primeStandby(i + 1);
  prefetch(i + 1);
  checkSleep();
}

function stop() {
  activeEl().pause();
  waStopAll();
  state.playing = false;
  $("playBtn").textContent = "Play";
  status("done");
  heartbeat(true);
}

/* --------------------------------------------------------------- controls */

$("playBtn").onclick = async () => {
  /* Inside the gesture, before the first await. The arming listeners above
     already do this; saying it here as well is what documents that the whole
     gapless path depends on it. */
  getAudioContext();
  if (state.playing) {
    /* Pause keeps the standby armed. Nothing is scheduled ahead with plain
       <audio> elements, so a buffered next paragraph is pure benefit on
       resume rather than something stale to invalidate. Under the graph there
       *is* something scheduled ahead, and waPause takes it down along with
       the position it had reached - the buffers stay, so re-scheduling it on
       resume costs a promise and no synthesis. */
    activeEl().pause();
    waPause();
    state.playing = false;
    $("playBtn").textContent = "Play";
    heartbeat(true);
  } else if (waResume()) {
    /* Straight back in at the sample Pause stopped on, fading in over 6ms so
       that resuming inside a word is silent rather than a click. */
    state.playing = true;
    $("playBtn").textContent = "Pause";
    status(`${state.i + 1} of ${state.blocks.length}`);
    primeStandby(state.i + 1);
  } else if (wa.paused && wa.pausedAt === state.i && state.i >= 0) {
    /* Paused under the graph, and the graph will not take it back: the context
       was suspended while we were paused - a device change, a sleep, a tab the
       browser decided to quiet. Go through playBlock at the offset Pause kept,
       which awaits the resume and reaches an element if the resume never
       comes. What must not happen is the thing waBegin would do on a suspended
       context: return true, empty both slots, and leave the reader silent with
       "Pause" showing and no path back. (wa.offset is read here, before
       playBlock's own waStopAll clears it.) */
    playBlock(state.i, wa.offset, { explicit: true });
  } else if (state.i >= 0 && slotHolds(active(), state.i)) {
    activeEl().play();
    state.playing = true;
    $("playBtn").textContent = "Pause";
    primeStandby(state.i + 1);
  } else if (state.i >= 0) {
    /* Loaded on a block, but the element holds nothing playable in the
       current voice - Play pressed during a switch, before the new voice
       finished synthesizing. Resuming here is what made the old voice
       reappear, so play the block properly instead: it waits for the new
       audio and starts the paragraph again from the top. */
    playBlock(state.i, 0, { explicit: true });
  } else {
    const startBlock = state.resumeSeek ? state.resumeSeek.block : 0;
    const seekWithin = state.resumeSeek ? state.resumeSeek.within : 0;
    state.resumeSeek = null;
    /* A little extra wait right at the start buys a real head start - by the
       time these resolve, the document is often already partly warmed
       server-side anyway (new_doc/_warm in talkify.py), so this is usually
       fast. Worth it: it's the difference between one stall up front and a
       stall after every paragraph. */
    status("buffering…");
    $("playBtn").disabled = true;
    const ahead = [];
    for (let k = startBlock; k < startBlock + PREFETCH_AHEAD && k < state.blocks.length; k++) ahead.push(loadAudio(k));
    await Promise.allSettled(ahead);
    $("playBtn").disabled = false;
    playBlock(startBlock, seekWithin, { explicit: true });
  }
};

$("prevBtn").onclick = () => playBlock(Math.max(0, state.i - 1), 0, { explicit: true });
$("nextBtn").onclick = () => playBlock(state.i + 1, 0, { explicit: true });

/* click any word to start reading from there */
reader.addEventListener("click", (e) => {
  const el = e.target.closest(".w");
  if (!el || el._b === undefined) return;
  const b = el._b, w = el._w;
  loadBlock(b).then((ok) => {
    if (!ok) return;
    const t = (state.words[b][w] || { s: 0 }).s;
    playBlock(b, t, { explicit: true });
  });
});

/* ------------------------------------------------------------ heartbeat */

/* Tells the server where playback actually is, so the library's progress
   bars and "resume where you left off" both stay accurate across a reload
   or a restart. Throttled on the frequent timeupdate path; sent immediately
   on anything that actually changes play state. */
function heartbeat(force = false) {
  if (state.i < 0 || !state.docId) return Promise.resolve();
  const now = Date.now();
  if (!force && now - state.lastHeartbeat < 3000) return Promise.resolve();
  state.lastHeartbeat = now;
  const within = currentWithin();
  return fetch("/api/position", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    /* `settled` marks the beats that came from something actually happening -
       a paragraph boundary, a pause, a seek, leaving the reader, closing the
       tab - as against the drift beat that fires every three seconds while a
       paragraph plays. The server writes the settled ones through immediately
       and coalesces the rest, so a resume stays exact at every boundary
       without rewriting the whole document to disk every three seconds. */
    body: JSON.stringify({ doc: state.docId, block: state.i, word: state.wi,
                           within, settled: force }),
    keepalive: true,
  }).then(async (r) => {
    /* A seek bumps the server's session generation, and the response to the
       seek is where the new number comes back. Keeping it current here is
       what stops the next voice switch being refused as stale. */
    if (r && r.ok) {
      const j = await r.json().catch(() => null);
      if (j && typeof j.generation === "number") state.generation = j.generation;
    }
  }).catch(() => {});
}
window.addEventListener("beforeunload", () => heartbeat(true));

/* ------------------------------------------------------------------ speed */

state.speed = parseFloat(localStorage.getItem("talkify.speed") || "1");

function setSpeed(v) {
  state.speed = v;
  /* Both slots: the standby is already loaded and would otherwise start the
     next paragraph at the old rate. */
  slots.A.el.playbackRate = v;
  slots.B.el.playbackRate = v;
  /* And the graph, which cannot simply be told: see waRetime. */
  waRetime(v);
  localStorage.setItem("talkify.speed", String(v));
  $("speedBtn").textContent = `${v.toFixed(2)}×`;
  document.querySelectorAll("#speedPanel .pop-item").forEach((b) =>
    b.classList.toggle("active", parseFloat(b.dataset.speed) === v));
}
setSpeed(state.speed);

$("speedBtn").onclick = () => togglePop("speedPanel");
document.querySelectorAll("#speedPanel .pop-item").forEach((b) =>
  b.onclick = () => { setSpeed(parseFloat(b.dataset.speed)); closePop("speedPanel"); });

/* ------------------------------------------------------------------ focus mode */

function setFocusMode(active) {
  state.focusMode = !!active;
  document.body.classList.toggle("focus-mode", state.focusMode);
  const btn = $("focusBtn");
  if (btn) btn.classList.toggle("active", state.focusMode);
  try { localStorage.setItem("talkify.focusMode", state.focusMode ? "1" : "0"); } catch {}
}

const savedFocus = localStorage.getItem("talkify.focusMode") === "1";
setFocusMode(savedFocus);

const focusBtn = $("focusBtn");
if (focusBtn) {
  focusBtn.onclick = () => setFocusMode(!state.focusMode);
}

window.addEventListener("keydown", (e) => {
  if (e.key === "f" || e.key === "F") {
    const tag = (e.target && e.target.tagName) || "";
    if (tag !== "INPUT" && tag !== "TEXTAREA" && !e.ctrlKey && !e.metaKey && !e.altKey) {
      setFocusMode(!state.focusMode);
    }
  }
});

/* ------------------------------------------------------------------ voice */

async function loadVoices() {
  const r = await fetch("/api/voices");
  const j = await r.json();
  state.voices = j.voices || [];
  state.defaultVoice = j.default;
  state.engineName = j.engine;
  return j;
}

function voiceLabel(id) {
  if (!id) return "Default";
  const v = state.voices.find((v) => v.id === id);
  return v ? v.label : id;
}

/* Labels are "Heart (US Female)". The parenthetical is a convention of the
   catalogue in engines.py, not a guarantee - the stub engine's one voice is
   just "Stub (silent)" - so anything unparsed simply yields no badges, and no
   filter but All matches it. Degrades to nothing, same rule as everywhere. */
function voiceMeta(v) {
  const m = /\(([^)]*)\)\s*$/.exec(v.label || "");
  const parts = m ? m[1].split(/\s+/).filter(Boolean) : [];
  const name = (v.label || v.id).replace(/\s*\([^)]*\)\s*$/, "");
  const accent = parts.find((p) => /^(US|UK|AU|CA|IE)$/i.test(p)) || null;
  const gender = parts.find((p) => /^(female|male)$/i.test(p)) || null;
  return { name, accent, gender, badges: [accent, gender].filter(Boolean) };
}

function updateVoiceBtn() {
  /* A pending switch shows the voice that is coming, not the one still being
     spoken. The reader asked for it and the request is live; showing the old
     label until a cold worker answers reads as the click having been lost. */
  const pending = state.pendingVoice;
  $("voiceBtn").textContent = pending
    ? voiceLabel(pending.voice) + "…" : voiceLabel(state.voice);
}

/* The one place a document's voice changes, whichever control asked for it -
   the player-bar popover and the gallery's "Use here" button must not drift
   apart on what a mid-read switch means.

   The server keeps the block that is currently sounding and starts the new
   voice at the next one (see _voice in talkify.py), so the client mirrors
   exactly that: the paragraph you are hearing finishes in the old voice, and
   everything after it belongs to the new one.

   What makes that true rather than nearly true is that the barrier goes up
   *before* the POST, not after the reply. The request is a reservation from
   the moment it is sent: state.pendingVoice names the last block the old
   voice may still own, and primeStandby, prefetch, loadBlock, loadAudio,
   slotHolds, playBlock and the ended-handler all refuse to touch anything
   past it. Between click and answer - which on a cold worker is seconds -
   nothing can begin one more paragraph in the voice being left, and nothing
   already begun can be handed to a slot.

   Only the newest request may be applied, and a late answer to an older one
   clears nothing: state.pendingVoice is compared by identity, so a superseded
   reply cannot take down the reservation that replaced it. */
async function applyVoiceToDocument(voiceId) {
  if (!state.docId || voiceId === state.voice) return false;
  const wasPlaying = state.playing;
  const doc = state.docId;
  /* Playing: the block now sounding keeps the old voice, so the barrier sits
     on it. Paused: nothing is sounding, so the current block goes too and
     Play speaks the new voice at once. */
  const barrier = wasPlaying ? state.i : state.i - 1;
  const epoch = ++state.voiceRequestEpoch;
  const pending = {
    id: `${doc}:${epoch}`, epoch, voice: voiceId, doc,
    at: state.i, wasPlaying, barrier, resumeAt: null,
  };
  const superseded = () => state.pendingVoice !== pending || state.docId !== doc;
  /* The barrier, and the invalidation, before a byte leaves. */
  state.pendingVoice = pending;
  state.playbackEpoch++;
  invalidateAhead(barrier);
  updateVoiceBtn();

  /* The block index goes with the request. The server would otherwise take
     it from the last /api/position heartbeat, which is throttled and may not
     have landed yet - and if it names an earlier block, the server keeps
     audio for a paragraph the reader has already left. Nothing loaded and no
     saved position means we genuinely don't know better than the session
     does, so the field is left out rather than sent as a guess. */
  const payload = { doc, voice: voiceId, playing: wasPlaying,
                    expected_generation: state.generation };
  const at = state.i >= 0 ? state.i
    : (state.resumeSeek ? state.resumeSeek.block : null);
  if (at !== null) payload.block = at;

  /* Give the reservation back and let the active voice stand. Only ever the
     reservation this call made: a newer one has already re-armed the barrier
     and must not be torn down by an older request's failure. */
  const release = () => {
    if (state.pendingVoice !== pending) return;
    state.pendingVoice = null;      // before anything below, or it still blocks
    if (state.serverVoice !== state.voice && state.docId === doc) {
      /* An earlier switch of ours did land, even though its reply arrived too
         late to apply. The server is the authority on what this document's
         voice is; catch up to it rather than fetching blocks in a voice it
         has already moved off. */
      state.voice = state.serverVoice;
      invalidateAhead(barrier);
    }
    updateVoiceBtn();
    if (!state.playing) return;
    if (pending.resumeAt !== null) {
      /* Playback stopped at the barrier waiting for a switch that is not
         going to happen. Start that block in the voice we still have, or the
         reader is left with Pause showing and nothing sounding. */
      playBlock(pending.resumeAt, 0);
      return;
    }
    if (state.i >= 0) {
      primeStandby(state.i + 1);
      prefetch(state.i + 1);
    }
  };

  let r;
  try {
    r = await fetch("/api/voice", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
  } catch {
    /* The network, not the server. Same rule: keep the voice that is
       actually being spoken and let the user ask again. */
    release();
    status("could not switch voice");
    return false;
  }
  if (r.status === 409) {
    /* Someone else moved the session on (another tab, or a seek that landed
       between this click and this request). Take the server's number and let
       the user try again rather than applying a switch decided against state
       that no longer exists. */
    const j = await r.json().catch(() => null);
    if (j && typeof j.generation === "number" && state.docId === doc) {
      state.generation = j.generation;
    }
    release();
    return false;
  }
  if (!r.ok) { release(); return false; }
  const okBody = await r.json().catch(() => null);
  /* Recorded even when this reply is history: the switch happened on the
     server, and its generation is the server's real count. Taking both is
     never wrong - what the epoch check gates is *acting* on the reply. */
  if (state.docId === doc) {
    if (okBody && typeof okBody.generation === "number") {
      state.generation = okBody.generation;
    }
    state.serverVoice = voiceId;
  }
  if (superseded()) {
    /* A newer switch was asked for while this was in flight. Its barrier is
       the live one; this reply is history and must not move the player. */
    return false;
  }

  state.voice = voiceId;
  state.pendingVoice = null;          // the barrier comes down here, and only here
  updateVoiceBtn();
  markGalleryState();
  /* Again, now definitively: anything fetched between the POST and this
     reply was refused by the barrier, but a blob that was already decoded
     when the switch was asked for is still sitting in the cache. */
  invalidateAhead(barrier);

  if (pending.resumeAt !== null) {
    /* Playback reached the boundary while the switch was unresolved and
       stopped there rather than speaking one more old-voice paragraph. This
       is the block it was waiting for, and it is now a new-voice block. */
    playBlock(pending.resumeAt, 0);
    return true;
  }
  if (wasPlaying) {
    /* The block now sounding keeps the old voice, so the switch is only
       audible at the next chunk boundary - and it is audible as a stall
       unless the next block is already made. Waiting for "ended" to ask for
       it means 4-8s of silence there, because that is what Kokoro plus
       Whisper costs. Ask now, while the current block still has a paragraph
       left to play. */
    primeStandby(state.i + 1);
    prefetch(state.i + 1);
  }
  if (!wasPlaying && state.i >= 0) {
    /* Paused mid-paragraph, the <audio> element is still holding the old
       voice's blob with currentTime part-way through it. Unload it now,
       before the awaits below: until the new audio arrives, Play must have
       nothing to resume, or it quietly speaks the rest of the paragraph in
       the voice the user just switched away from. */
    releaseAllSlots();
    clearHighlight();
    state.wi = -1;
    status("loading new voice…");
    /* Whoever touches the element last wins, so hand it back if anything
       else claims it while these awaits are outstanding: a third voice
       picked from the gallery, or Play pressed before the new audio landed
       (which routes through playBlock and loads the block itself). */
    const token = playToken;
    const mine = () => state.voice === voiceId && playToken === token;
    try {
      const ok = await loadBlock(state.i);
      if (!ok || !mine()) return true;
      const url = await loadAudio(state.i);
      if (!url || !mine()) return true;
      /* Preferring the graph here is what keeps the *next* boundary gapless
         too. "A decoded buffer, paused at the top of this block" is exactly
         what Play resumes from and it needs no element at all - whereas
         arming an element leaves the reader on the ping-pong path until their
         next explicit jump. Falls through to the element when there is no
         context or the clip will not decode, which is the same fallback as
         everywhere else. */
      /* waLive(), not merely a decoded buffer: decodeAudioData succeeds on a
         suspended context, so a buffer is not evidence that the graph can
         play. Without this, a suspended context here would take the graph
         route, arm no element, and leave Play with nothing either side. */
      const buffer = waLive() ? await ensureBuffer(state.i) : null;
      if (!mine()) return true;
      if (buffer) {
        wa.paused = true;
        wa.pausedAt = state.i;
        wa.offset = 0;
        wa.lastPos = { i: state.i, within: 0 };
      } else {
        armSlot(active(), state.i, url, voiceId, state.playbackEpoch);
        try { activeEl().currentTime = 0; } catch {}
      }
      status(`${state.i + 1} of ${state.blocks.length}`);
      primeStandby(state.i + 1);
      prefetch(state.i + 1);
    } catch { status("synthesis failed"); }
  }
  return true;
}

async function renderVoicePanel() {
  if (!state.voices.length) await loadVoices();
  const panel = $("voicePanel");
  panel.innerHTML = "";
  state.voices.forEach((v) => {
    const b = document.createElement("button");
    b.className = "pop-item" + (v.id === state.voice ? " active" : "");
    b.textContent = v.label;
    b.onclick = async () => {
      closePop("voicePanel");
      await applyVoiceToDocument(v.id);
    };
    panel.appendChild(b);
  });
}

$("voiceBtn").onclick = async () => { await renderVoicePanel(); togglePop("voicePanel"); };

/* ---------------------------------------------------------- voice gallery */

const voiceFilter = { chip: "all", query: "" };

function voiceMatches(v) {
  const meta = voiceMeta(v);
  const q = voiceFilter.query.trim().toLowerCase();
  if (q && !(v.label || "").toLowerCase().includes(q) && !v.id.toLowerCase().includes(q)) return false;
  switch (voiceFilter.chip) {
    case "us": return meta.accent === "US";
    case "uk": return meta.accent === "UK";
    case "female": return /female/i.test(meta.gender || "");
    case "male": return /^male$/i.test(meta.gender || "");
    default: return true;
  }
}

function tag(text, cls) {
  const el = document.createElement("span");
  el.className = cls;
  el.textContent = text;
  return el;
}

/* Badges are re-marked rather than re-rendered, so switching the active voice
   from a card does not tear the grid out from under the click that did it,
   and does not cut off a preview clip that is playing. */
function markGalleryState() {
  document.querySelectorAll(".voice-card").forEach((card) => {
    const id = card.dataset.voiceId;
    card.classList.toggle("is-default", id === state.defaultVoice);
    card.classList.toggle("is-active", !!state.docId && id === state.voice);
    const tags = card.querySelector(".v-tags");
    if (tags) {
      tags.innerHTML = "";
      if (id === state.defaultVoice) tags.appendChild(tag("Default", "v-tag"));
      if (state.docId && id === state.voice) tags.appendChild(tag("Active in Reader", "v-tag v-tag-active"));
    }
    const useBtn = card.querySelector(".v-use");
    if (useBtn) {
      useBtn.hidden = !state.docId;
      useBtn.disabled = id === state.voice;
      useBtn.textContent = id === state.voice ? "In use" : "Use here";
    }
    const defBtn = card.querySelector(".v-default");
    if (defBtn) defBtn.disabled = id === state.defaultVoice;
  });
}

function applyVoiceFilter() {
  let shown = 0;
  document.querySelectorAll(".voice-card").forEach((card) => {
    const v = state.voices.find((x) => x.id === card.dataset.voiceId);
    const on = v ? voiceMatches(v) : false;
    card.hidden = !on;
    if (on) shown++;
  });
  $("voiceEmpty").hidden = shown > 0 || !state.voices.length;
}

async function refreshVoiceGallery() {
  await loadVoices();
  const grid = $("voiceGrid");
  grid.innerHTML = "";
  if (!state.voices.length) {
    $("voiceEngineHint").textContent = "No voices available for the running engine.";
    $("voiceEmpty").hidden = true;
    return;
  }
  $("voiceEngineHint").textContent =
    `Voices for the running engine (${state.engineName || "unknown"}). ` +
    (state.docId ? "Preview one, set a default, or switch the open document to it."
                 : "Preview one, or set it as the default for documents you open.");

  state.voices.forEach((v) => {
    const meta = voiceMeta(v);
    const card = document.createElement("div");
    card.className = "voice-card";
    card.dataset.voiceId = v.id;

    const label = document.createElement("div");
    label.className = "v-label";
    label.textContent = meta.name;
    card.appendChild(label);

    const badges = document.createElement("div");
    badges.className = "v-badges";
    meta.badges.forEach((b) => badges.appendChild(tag(b, "v-badge")));
    badges.appendChild(tag(state.engineName || "engine", "v-badge v-badge-engine"));
    card.appendChild(badges);

    const tags = document.createElement("div");
    tags.className = "v-tags";
    card.appendChild(tags);

    const actions = document.createElement("div");
    actions.className = "v-actions";

    const playBtn = document.createElement("button");
    playBtn.className = "ghost v-play";
    playBtn.textContent = "Preview";
    playBtn.onclick = () => togglePreview(v.id, playBtn);

    const defBtn = document.createElement("button");
    defBtn.className = "ghost v-default";
    defBtn.textContent = "Set default";
    defBtn.title = "Use this voice for documents opened from now on";
    defBtn.onclick = async () => {
      const r = await fetch("/api/voice", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ voice: v.id, default: true }),
      });
      if (!r.ok) return;
      state.defaultVoice = v.id;
      markGalleryState();
    };

    const useBtn = document.createElement("button");
    useBtn.className = "ghost v-use";
    useBtn.textContent = "Use here";
    useBtn.title = "Use this voice for the document open in the reader";
    useBtn.onclick = () => applyVoiceToDocument(v.id);

    actions.append(playBtn, defBtn, useBtn);
    card.appendChild(actions);
    grid.appendChild(card);
  });
  markGalleryState();
  applyVoiceFilter();
}

/* Two voices talking at once is the one thing a voice gallery must never do,
   so starting a preview pauses the document. It is not resumed afterwards:
   the user came here to compare voices, and playback restarting under them
   when a three-second clip ends is worse than pressing Play again. */
function togglePreview(voiceId, btn) {
  if (previewAudio.dataset.voice === voiceId && !previewAudio.paused) {
    previewAudio.pause();
    return;
  }
  if (state.playing) {
    activeEl().pause();
    waPause();
    state.playing = false;
    $("playBtn").textContent = "Play";
    heartbeat(true);
  }
  btn.textContent = "…";
  previewAudio.src = `/api/voice/preview?voice=${encodeURIComponent(voiceId)}`;
  previewAudio.dataset.voice = voiceId;
  previewAudio.play().catch(resetPreviewButtons);
}

$("voiceSearch").addEventListener("input", (e) => {
  voiceFilter.query = e.target.value;
  applyVoiceFilter();
});
document.querySelectorAll("#voiceChips .chip").forEach((chip) => {
  chip.onclick = () => {
    voiceFilter.chip = chip.dataset.filter;
    document.querySelectorAll("#voiceChips .chip").forEach((c) => c.classList.toggle("active", c === chip));
    applyVoiceFilter();
  };
});

/* Bound once (not per card-render) so repeated gallery refreshes don't pile
   up duplicate listeners on the shared preview <audio> element. */
function previewButton(voiceId) {
  return document.querySelector(`.voice-card[data-voice-id="${CSS.escape(voiceId || "")}"] .v-play`);
}
function resetPreviewButtons() {
  document.querySelectorAll(".voice-card .v-play").forEach((b) => { b.textContent = "Preview"; });
  document.querySelectorAll(".voice-card").forEach((c) => c.classList.remove("is-previewing"));
}
previewAudio.addEventListener("play", () => {
  resetPreviewButtons();
  const b = previewButton(previewAudio.dataset.voice);
  if (b) { b.textContent = "Pause"; b.closest(".voice-card").classList.add("is-previewing"); }
});
previewAudio.addEventListener("pause", resetPreviewButtons);
previewAudio.addEventListener("ended", resetPreviewButtons);
previewAudio.addEventListener("error", resetPreviewButtons);

/* -------------------------------------------------------------- popovers */

function closeAllPopovers() {
  ["voicePanel", "speedPanel", "sleepPanel", "themePanel", "chaptersPanel"].forEach(closePop);
}
function closePop(id) { $(id).hidden = true; }
function togglePop(id) {
  const willOpen = $(id).hidden;
  closeAllPopovers();
  $(id).hidden = !willOpen;
}
document.addEventListener("click", (e) => {
  if (!e.target.closest(".pop, #voiceBtn, #speedBtn, #sleepBtn, #themeBtn, #chaptersBtn")) closeAllPopovers();
});

/* ------------------------------------------------------------------ theme */

function applyTheme(theme) {
  document.body.dataset.theme = theme;
  localStorage.setItem("talkify.theme", theme);
  document.querySelectorAll("#themePanel [data-theme]").forEach((b) =>
    b.classList.toggle("active", b.dataset.theme === theme));
}
document.querySelectorAll("#themePanel [data-theme]").forEach((b) =>
  b.onclick = () => applyTheme(b.dataset.theme));
applyTheme(localStorage.getItem("talkify.theme") || "default");

function setNight(on) {
  document.body.dataset.night = on ? "on" : "off";
  localStorage.setItem("talkify.night", on ? "1" : "0");
  $("nightToggle").textContent = `Night filter: ${on ? "On" : "Off"}`;
}
$("nightToggle").onclick = () => setNight(document.body.dataset.night !== "on");
setNight(localStorage.getItem("talkify.night") === "1");

$("themeBtn").onclick = () => togglePop("themePanel");

/* ------------------------------------------------------------ sleep timer */

function sleepLabel() {
  if (state.sleep.mode === "off") return "Sleep: Off";
  if (state.sleep.mode === "doc") return "Sleep: End of doc";
  return `Sleep: ${state.sleep.mode} min`;
}

function setSleep(mode) {
  state.sleep.mode = mode;
  state.sleep.deadline = (mode !== "off" && mode !== "doc") ? Date.now() + (+mode) * 60000 : null;
  $("sleepBtn").textContent = sleepLabel();
  document.querySelectorAll("#sleepPanel [data-sleep]").forEach((b) =>
    b.classList.toggle("active", b.dataset.sleep === mode));
}
document.querySelectorAll("#sleepPanel [data-sleep]").forEach((b) =>
  b.onclick = () => { setSleep(b.dataset.sleep); closePop("sleepPanel"); });
$("sleepBtn").onclick = () => togglePop("sleepPanel");

function checkSleep() {
  if (state.sleep.deadline && Date.now() >= state.sleep.deadline) {
    stop();
    setSleep("off");
    status("sleep timer stopped playback");
  }
}
setInterval(checkSleep, 1000);

/* -------------------------------------------------------------- in-doc search */

/* Lowercased blocks, made once per document instead of once per keystroke.
   Searching a textbook used to allocate a lowercase copy of the whole book on
   every character typed. */
function lowerBlocks() {
  if (!state.lowerBlocks || state.lowerBlocks.length !== state.blocks.length) {
    state.lowerBlocks = state.blocks.map((t) => t.toLowerCase());
  }
  return state.lowerBlocks;
}

/* Where each word of a block starts, in characters. Only built for blocks
   that actually contain the query - which on a real search is a handful out
   of thousands. */
function wordStarts(text) {
  const starts = [];
  const re = /\S+/g;
  let m;
  while ((m = re.exec(text)) !== null) starts.push(m.index);
  return starts;
}

function findMatches(query) {
  const q = query.toLowerCase();
  const matches = [];
  if (!q) return matches;
  const lows = lowerBlocks();
  state.blocks.forEach((text, bi) => {
    const lower = lows[bi];
    let idx = lower.indexOf(q);
    if (idx === -1) return;
    /* The word a match sits in is the number of word-starts before it. Walking
       a cursor through those in step with the matches (both run left to right)
       replaces a slice-plus-split of the block per match - which on a common
       substring was quadratic in the block and froze the tab on every
       keystroke. */
    const starts = wordStarts(text);
    let wordIndex = 0;
    while (idx !== -1) {
      while (wordIndex < starts.length && starts[wordIndex] < idx) wordIndex++;
      matches.push({ bi, wordIndex });
      idx = lower.indexOf(q, idx + q.length);
    }
  });
  return matches;
}

let searchHitEl = null;

function clearSearchHit() {
  /* The element is remembered rather than found again: querySelectorAll over
     a document of a few hundred thousand spans, to clear one class, ran on
     every keystroke. */
  if (searchHitEl) { searchHitEl.classList.remove("hit"); searchHitEl = null; }
}

function showMatch(idx) {
  const m = state.search.matches[idx];
  if (!m) return;
  clearSearchHit();
  const p = $(`b-${m.bi}`);
  const el = p && p._words[m.wordIndex];
  if (el) { markProgrammaticScroll(700); el.classList.add("hit"); searchHitEl = el; el.scrollIntoView({ block: "center", behavior: "smooth" }); }
  $("searchCount").textContent = `${idx + 1} of ${state.search.matches.length}`;
}

function jumpToMatch(idx) {
  const m = state.search.matches[idx];
  if (!m) return;
  state.search.idx = idx;
  showMatch(idx);
  loadBlock(m.bi).then((ok) => {
    if (!ok) return;
    const w = state.words[m.bi][m.wordIndex] || { s: 0 };
    playBlock(m.bi, w.s, { explicit: true });
  });
}

function runSearch(query) {
  state.search.query = query;
  state.search.matches = findMatches(query);
  state.search.idx = state.search.matches.length ? 0 : -1;
  if (state.search.idx >= 0) showMatch(0);
  else { clearSearchHit(); $("searchCount").textContent = query ? "no matches" : ""; }
}

$("searchInput").addEventListener("input", (e) => runSearch(e.target.value.trim()));
$("searchInput").addEventListener("keydown", (e) => {
  e.stopPropagation();
  if (e.key === "Enter" && state.search.idx >= 0) jumpToMatch(state.search.idx);
  if (e.key === "Escape") closeSearch();
});
$("searchNext").onclick = () => {
  if (!state.search.matches.length) return;
  showMatch((state.search.idx + 1) % state.search.matches.length);
  state.search.idx = (state.search.idx + 1) % state.search.matches.length;
};
$("searchPrev").onclick = () => {
  if (!state.search.matches.length) return;
  state.search.idx = (state.search.idx - 1 + state.search.matches.length) % state.search.matches.length;
  showMatch(state.search.idx);
};
$("searchClose").onclick = () => closeSearch();

function openSearch() {
  $("readerSearch").hidden = false;
  $("searchInput").focus();
  $("searchInput").select();
}
function closeSearch() {
  $("readerSearch").hidden = true;
  clearSearchHit();
}
$("searchToggleBtn").onclick = () => ($("readerSearch").hidden ? openSearch() : closeSearch());

/* ------------------------------------------------------------ chapters */

/* Absent (not disabled) until chapters.py's background detection finds
   something for the open document - see talkify.py's _detect_chapters.
   That runs after the document is already open and readable, so this polls
   briefly for a result rather than expecting one immediately; most
   documents never get one, and the button just never appears. */
state.chapters = [];
let chaptersPollTimer = null;

function updateChaptersUI() {
  $("chaptersWrap").hidden = state.chapters.length === 0;
}

function stopChaptersPoll() {
  if (chaptersPollTimer) { clearTimeout(chaptersPollTimer); chaptersPollTimer = null; }
}

/* Backoff, and it stops on an answer rather than on a stopwatch. The server
   now says when detection has settled - "ready" with chapters, or "none"
   because this book prints no running heads and has no parsable contents
   page - so the old fixed two-minute cutoff (which gave up on a long scan
   that was still working, and kept polling forever on one that never would)
   is gone. */
function pollChaptersIfNeeded(docId) {
  stopChaptersPoll();
  if (state.chapters.length) return;
  let delay = 1500;
  const tick = async () => {
    if (docId !== state.docId) return;
    let j = null;
    try {
      const r = await fetch(`/api/chapters?doc=${docId}`);
      if (r.ok) j = await r.json();
    } catch { /* offline for a moment; just try again */ }
    if (docId !== state.docId) return;
    if (j && j.chapters && j.chapters.length) {
      state.chapters = j.chapters;
      updateChaptersUI();
      stopChaptersPoll();
      return;
    }
    if (j && j.chapter_detection && j.chapter_detection.state === "none") {
      stopChaptersPoll();      // nothing is coming; the button simply never appears
      return;
    }
    if (docId !== state.docId) return;
    delay = Math.min(delay * 1.6, 20000);
    chaptersPollTimer = setTimeout(tick, delay);
  };
  chaptersPollTimer = setTimeout(tick, delay);
}

function renderChaptersPanel() {
  const panel = $("chaptersPanel");
  panel.innerHTML = "";
  state.chapters.forEach((c) => {
    const b = document.createElement("button");
    b.className = "pop-item";
    b.textContent = c.title;
    b.onclick = () => {
      closePop("chaptersPanel");
      loadBlock(c.block).then((ok) => { if (ok) playBlock(c.block, 0, { explicit: true }); });
    };
    panel.appendChild(b);
  });
}
$("chaptersBtn").onclick = () => { renderChaptersPanel(); togglePop("chaptersPanel"); };


/* ------------------------------------------------------------- figures */

/* Optional enhancement: when the narration reaches "as shown in
   Figure 3.20", show Figure 3.20.

   Everything about this is built so that failure is invisible. The dock's DOM
   is not created until a valid reference is actually reached, so a document
   with no figures has no empty panel to look at. A manifest that says "none",
   a fetch that fails, an image that will not decode, a document that has no
   PDF behind it at all - each one leaves the reader exactly as it was, with
   no dialog and no interruption to playback. */

let figuresPollTimer = null;
let figureDock = null;
/* Goes up whenever the reader moves somewhere else. An image that finishes
   loading after that is stale and must not appear, however long it took. */
let figureGen = 0;

function stopFiguresPoll() {
  if (figuresPollTimer) { clearTimeout(figuresPollTimer); figuresPollTimer = null; }
}

function resetFigures() {
  stopFiguresPoll();
  state.figures = { state: "none", items: {}, refs: {} };
  state.figuresGone = {};
  figureGen++;
  hideFigure();
}

/* Same backoff-and-settle shape as the chapter poll: indexing a long book
   takes a while, and "none" is a final answer worth stopping on. */
function pollFiguresIfNeeded(docId) {
  stopFiguresPoll();
  let delay = 1200;
  const tick = async () => {
    if (docId !== state.docId) return;
    let j = null;
    try {
      const r = await fetch(`/api/figures?doc=${docId}`);
      if (r.ok) j = await r.json();
    } catch { /* try again */ }
    if (docId !== state.docId) return;
    if (j && j.state === "ready") {
      const items = {};
      (j.items || []).forEach((it) => { items[it.id] = it; });
      state.figures = { state: "ready", items, refs: j.refs || {} };
      stopFiguresPoll();
      updateFigureForBlock();
      return;
    }
    if (!j || j.state !== "pending") { stopFiguresPoll(); return; }
    if (docId !== state.docId) return;
    delay = Math.min(delay * 1.6, 20000);
    figuresPollTimer = setTimeout(tick, delay);
  };
  figuresPollTimer = setTimeout(tick, delay);
}

function refsForCurrentBlock() {
  if (!state.figures || state.figures.state !== "ready" || state.i < 0) return [];
  return state.figures.refs[String(state.i)] || [];
}

/* Called on every block change. A paragraph that mentions no figure closes
   whatever was showing - the dock follows the reading, it does not accumulate. */
function updateFigureForBlock() {
  figureGen++;
  if (!refsForCurrentBlock().length) hideFigure();
}

/* Called on every word transition. Reveal at the word the sentence names the
   figure at, not at the top of the paragraph: hearing "as shown in Figure
   3.20" and seeing it at that instant is the whole feature. */
function updateFigureForWord(wi) {
  const refs = refsForCurrentBlock();
  if (!refs.length) return;
  const reached = refs.filter((r) => wi >= r.start_word && !state.figuresGone[r.figure_id]);
  if (!reached.length) return;
  const hit = reached[reached.length - 1];
  if (state.figureShown === hit.figure_id) return;
  showFigure(hit);
}

function ensureFigureDock() {
  if (figureDock) return figureDock;
  const dock = document.createElement("div");
  dock.className = "figure-dock";
  dock.id = "figureDock";

  const bar = document.createElement("div");
  bar.className = "figure-dock-bar";
  const label = document.createElement("span");
  label.className = "figure-dock-label";
  bar.appendChild(label);
  const close = document.createElement("button");
  close.className = "ghost figure-dock-close";
  close.type = "button";
  close.textContent = "×";
  close.title = "Hide this figure";
  /* Dismiss is for this figure, for this session only. Narration is never
     touched by it. */
  close.onclick = (e) => {
    e.stopPropagation();
    if (state.figureShown) state.figuresGone[state.figureShown] = true;
    hideFigure();
  };
  bar.appendChild(close);

  const img = document.createElement("img");
  img.className = "figure-dock-img";
  img.alt = "";
  img.decoding = "async";
  img.onerror = () => {
    /* No retry, no placeholder, no badge. One failure means this figure is
       simply unavailable for the rest of the session. */
    if (state.figureShown) state.figuresGone[state.figureShown] = true;
    hideFigure();
  };
  img.onclick = () => {
    const page = dock.dataset.page ? +dock.dataset.page : null;
    if (page === null || !state.hasPages) return;
    setSplitView(true);
    /* They asked for the figure's page, not the page being read. Hold it
       there until they say otherwise. */
    state.pdfAutoFollow = false;
    loadPdfPage(page, { force: true });
  };

  const caption = document.createElement("div");
  caption.className = "figure-dock-caption";

  dock.append(bar, img, caption);
  document.body.appendChild(dock);
  figureDock = dock;
  return dock;
}

function hideFigure() {
  state.figureShown = null;
  if (!figureDock) return;
  figureDock.classList.remove("is-open");
  figureDock.querySelector(".figure-dock-img").removeAttribute("src");
}

function showFigure(ref) {
  const item = state.figures.items[ref.figure_id];
  if (!item) return;
  const gen = figureGen, doc = state.docId;
  const dock = ensureFigureDock();
  const img = dock.querySelector(".figure-dock-img");
  const probe = new Image();
  probe.src = `/api/figure?doc=${encodeURIComponent(doc)}&id=${encodeURIComponent(ref.figure_id)}`;
  const commit = () => {
    /* Decoded - but the reader may have skipped, switched document or closed
       this in the meantime, and a preview that arrives after the moment it
       belonged to is worse than none. */
    if (gen !== figureGen || doc !== state.docId) return;
    dock.dataset.page = String(item.page);
    dock.querySelector(".figure-dock-label").textContent = item.label || "Figure";
    dock.querySelector(".figure-dock-caption").textContent = item.caption || "";
    img.src = probe.src;
    dock.classList.add("is-open");
    state.figureShown = ref.figure_id;
  };
  const fail = () => {
    state.figuresGone[ref.figure_id] = true;
  };
  if (probe.decode) {
    probe.decode().then(commit, fail);
  } else {
    probe.onload = commit;
    probe.onerror = fail;
  }
}

/* ---------------------------------------------------------------- explain */

/* Absent, not disabled, when nothing can answer - checked once at boot.
   No button, no toolbar, no panel: same degrade-to-nothing rule as every
   other optional feature here. */
state.aiAvailable = false;
state.ai = { available: false, provider: null, model: null, local: false, label: null };
async function checkAiAvailable() {
  try {
    const r = await fetch("/api/ai/status");
    const j = await r.json();
    state.ai = j;
    state.aiAvailable = !!j.available;
  } catch { state.ai = { available: false }; state.aiAvailable = false; }
  renderProviderBadge();
}

/* Name the model in the panel, and say plainly when the passage is leaving
   the machine - a reader may well be reading something private, and should
   never have to guess where the text went. */
function renderProviderBadge() {
  const badge = $("explainProvider");
  const ai = state.ai || {};
  badge.classList.toggle("is-cloud", !!ai.available && !ai.local);
  badge.title = ai.local
    ? "Runs on this machine - nothing is sent anywhere"
    : "This passage is sent to the provider's API";
  badge.textContent = !ai.available ? ""
    : ai.local ? ai.label || "Local (Ollama)"
    : `${ai.label || ai.provider} — sent to the cloud`;
}

const explainPanel = $("explainPanel");
let explainBlockIndex = null;
let explainSelection = null;
let hoverHideTimer = null;

/* The panel is a column in the split wrap, so opening it has to tell the
   wrap that a third pane exists - the other two size themselves off that. */
function setExplainOpen(on) {
  explainPanel.hidden = !on;
  $("readerSplitWrap").classList.toggle("is-explaining", on);
}

/* position: fixed, so these coordinates are plain viewport coordinates from
   getBoundingClientRect() - no container-relative math, no scrollTop/Left
   to add back in. The earlier version computed an offset against #reader
   but the button is actually laid out in #reader's *parent* (#view-reader) -
   a real container mismatch that could push it past the edge and trigger
   the implicit overflow-x:auto that comes from #reader/.view only setting
   overflow-y (see the overflow-x:hidden added in style.css), shifting the
   whole centered column sideways. Viewport coordinates sidestep the whole
   class of bug instead of computing them correctly for a nested scroller. */
function positionHoverBtn(blockEl) {
  const btn = $("explainHoverBtn");
  const r = blockEl.getBoundingClientRect();
  btn.style.top = `${r.top + 4}px`;
  btn.style.left = `${Math.min(r.right - 74, window.innerWidth - 80)}px`;
  btn.dataset.block = blockEl.id.slice(2);
  btn.hidden = false;
}
reader.addEventListener("mouseover", (e) => {
  if (!state.aiAvailable) return;
  const blk = e.target.closest(".blk");
  if (!blk) return;
  if (hoverHideTimer) { clearTimeout(hoverHideTimer); hoverHideTimer = null; }
  positionHoverBtn(blk);
});
reader.addEventListener("mouseout", (e) => {
  const to = e.relatedTarget;
  if (to && to.closest && to.closest("#explainHoverBtn")) return;
  hoverHideTimer = setTimeout(() => { $("explainHoverBtn").hidden = true; }, 250);
});
$("explainHoverBtn").onmouseenter = () => { if (hoverHideTimer) { clearTimeout(hoverHideTimer); hoverHideTimer = null; } };
$("explainHoverBtn").onclick = () => {
  const bi = +$("explainHoverBtn").dataset.block;
  $("explainHoverBtn").hidden = true;
  openExplainPanel(bi, null);
};

reader.addEventListener("mouseup", () => {
  const bar = $("selectionToolbar");
  if (!state.aiAvailable) { bar.hidden = true; return; }
  const sel = window.getSelection();
  const text = sel.toString().trim();
  if (!text || sel.rangeCount === 0) { bar.hidden = true; return; }
  const anchorEl = sel.anchorNode.nodeType === 1 ? sel.anchorNode : sel.anchorNode.parentElement;
  const blk = anchorEl && anchorEl.closest(".blk");
  if (!blk) { bar.hidden = true; return; }
  const rect = sel.getRangeAt(0).getBoundingClientRect();
  bar.style.top = `${rect.top - 38}px`;
  bar.style.left = `${rect.left}px`;
  bar.dataset.block = blk.id.slice(2);
  bar.dataset.selection = text;
  bar.hidden = false;
});
document.addEventListener("mousedown", (e) => {
  if (!e.target.closest("#selectionToolbar")) $("selectionToolbar").hidden = true;
});
$("selectionExplainBtn").onclick = () => {
  const bar = $("selectionToolbar");
  const bi = +bar.dataset.block, selection = bar.dataset.selection;
  bar.hidden = true;
  openExplainPanel(bi, selection);
};

async function openExplainPanel(blockIndex, selection) {
  explainBlockIndex = blockIndex;
  // Held for the life of the panel, because a follow-up question has to be
  // grounded in the same passage the first answer was. Sending null here
  // silently swapped the selection for whichever block it started in.
  explainSelection = selection || null;
  setExplainOpen(true);
  $("explainPassage").textContent = explainSelection || state.blocks[blockIndex] || "";
  $("explainAskInput").value = "";
  await runExplain(blockIndex, explainSelection, null);
}

async function runExplain(blockIndex, selection, question) {
  $("explainLoading").hidden = false;
  $("explainErr").hidden = true;
  $("explainWarning").hidden = true;
  $("explainAnswer").textContent = "";
  try {
    const r = await fetch("/api/explain", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ doc: state.docId, block: blockIndex, selection, question }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(j.error || "explain failed");
    $("explainAnswer").textContent = j.answer;
    if (j.warning) { $("explainWarning").textContent = j.warning; $("explainWarning").hidden = false; }
  } catch (e) {
    $("explainErr").textContent = e.message;
    $("explainErr").hidden = false;
  } finally {
    $("explainLoading").hidden = true;
  }
}

$("explainCloseBtn").onclick = () => { setExplainOpen(false); };
$("explainAskForm").addEventListener("submit", (e) => {
  e.preventDefault();
  const q = $("explainAskInput").value.trim();
  if (!q || explainBlockIndex === null) return;
  runExplain(explainBlockIndex, explainSelection, q);
});

/* --------------------------------------------------------- back-to-position */

function updateResumeBtnVisibility() {
  const btn = $("resumeBtn");
  if (state.view !== "reader" || state.i < 0) { btn.hidden = true; return; }
  const p = $(`b-${state.i}`);
  if (!p) { btn.hidden = true; return; }
  const r = p.getBoundingClientRect();
  const top = reader.getBoundingClientRect().top;
  const h = reader.clientHeight;
  const y = r.top - top;
  const textDiverged = (y < -40 || y > h * 0.9);
  const activePage = (state.blockPages && state.i >= 0) ? state.blockPages[state.i] : null;
  /* Turning auto-follow off is itself a divergence. The pane stops tracking
     the reader the moment it is paged by hand (see the early return in the
     follow routine), so the page it is left showing can still happen to be
     the active block's - and comparing page numbers alone would hide the
     one control that puts the two panes back together. */
  const pdfDiverged = !!(state.splitView && (state.pdfAutoFollow === false
    || (activePage !== null && activePage !== undefined && state.currentPage !== activePage)));
  btn.hidden = !(textDiverged || pdfDiverged);
}
reader.addEventListener("scroll", () => {
  if (Date.now() > programmaticScrollUntil) state.autoFollow = false;
  requestAnimationFrame(updateResumeBtnVisibility);
});
$("resumeBtn").onclick = () => {
  const p = $(`b-${state.i}`);
  state.autoFollow = true;
  if (p) { markProgrammaticScroll(700); p.scrollIntoView({ block: "center", behavior: "smooth" }); }
  if (state.splitView && state.blockPages && state.i >= 0) {
    state.pdfAutoFollow = true;
    const activePage = state.blockPages[state.i];
    if (activePage !== undefined && activePage !== null) loadPdfPage(activePage);
  }
  requestAnimationFrame(updateResumeBtnVisibility);
};
setInterval(updateResumeBtnVisibility, 1000);

/* ------------------------------------------------------------- split-view */

const splitToggleBtn = $("splitToggleBtn");
const readerSplitWrap = $("readerSplitWrap");
const pdfPane = $("pdfPane");
const pdfPageImg = $("pdfPageImg");
const pdfPageLabel = $("pdfPageLabel");
const pdfPrevPage = $("pdfPrevPage");
const pdfNextPage = $("pdfNextPage");
const pdfCloseBtn = $("pdfCloseBtn");

let pdfPrefetchTimer = null;

/* The pages this document actually holds, as an inclusive [first, last].

   Opening "pages 20-100" of a 980-page book opens a book of eighty-one
   pages. The pane used to be scoped to the file instead: it opened on page 1
   - which the reader had not loaded and could not be reading - and Next
   would walk all 980, rendering multi-megabyte scans of pages that are not
   part of this document at all. The server sends the range (it is the one
   that knows the selection the document was opened with) and enforces it on
   /api/page as well, so a tab still running a cached copy of this file
   cannot reproduce the walk. A document opened whole sends null and browses
   the whole file, exactly as before. */
/* Every page this document contains, ascending. Null means "the whole file",
   which is left as arithmetic rather than a thousand-element array. */
function loadedPages() {
  const runs = state.pageRuns;
  if (!Array.isArray(runs) || !runs.length) return null;
  const last = Math.max(0, state.pageCount - 1);
  const out = [];
  for (const run of runs) {
    if (!Array.isArray(run) || run.length !== 2) continue;
    const lo = Math.max(0, Math.min(run[0], last));
    const hi = Math.max(0, Math.min(run[1], last));
    for (let p = lo; p <= hi; p++) out.push(p);
  }
  return out.length ? out : null;
}

function loadedPageRange() {
  const pages = loadedPages();
  if (!pages) return [0, Math.max(0, state.pageCount - 1)];
  return [pages[0], pages[pages.length - 1]];
}

/* The nearest page this document actually has, at or after `p` when stepping
   forward and at or before it when stepping back. Null when there is none -
   which is what disables Prev and Next at the ends. */
function nearestLoadedPage(p, direction) {
  const pages = loadedPages();
  if (!pages) {
    const last = Math.max(0, state.pageCount - 1);
    const clamped = Math.max(0, Math.min(p, last));
    return (p < 0 || p > last) && direction ? null : clamped;
  }
  if (direction >= 0) {
    for (const q of pages) if (q >= p) return q;
    return direction > 0 ? null : pages[pages.length - 1];
  }
  for (let i = pages.length - 1; i >= 0; i--) if (pages[i] <= p) return pages[i];
  return direction < 0 ? null : pages[0];
}

/* Where the pane opens when nothing is playing yet: the first page of what
   was loaded, not page one of the file. */
function currentPdfPage() {
  if (state.blockPages && state.i >= 0
      && state.blockPages[state.i] !== undefined) {
    return state.blockPages[state.i];
  }
  if (typeof state.currentPage === "number") return state.currentPage;
  return loadedPageRange()[0];
}

function updateSplitViewUI() {
  if (splitToggleBtn) splitToggleBtn.hidden = !state.hasPages;
  if (!state.hasPages) {
    setSplitView(false);
  } else if (state.splitView) {
    loadPdfPage(currentPdfPage(), { force: true });
  }
}

function setSplitView(on) {
  state.splitView = !!on && state.hasPages;
  if (splitToggleBtn) splitToggleBtn.classList.toggle("active", state.splitView);
  if (readerSplitWrap) readerSplitWrap.classList.toggle("is-split", state.splitView);
  if (pdfPane) pdfPane.hidden = !state.splitView;
  if (state.splitView) {
    state.pdfAutoFollow = true;
    loadPdfPage(currentPdfPage(), { force: true });
  } else if (pdfPageImg) {
    pdfPageImg.removeAttribute("src");
  }
}

function loadPdfPage(p, { force = false } = {}) {
  if (!state.hasPages || !state.docId || state.pageCount <= 0) return;
  const pages = loadedPages();
  const [first, last] = loadedPageRange();
  /* Snap onto a page this document actually has. Forward first, then back,
     so a block that names an unselected page still shows something sensible
     rather than nothing. */
  const target = nearestLoadedPage(Math.max(first, Math.min(p, last)), 0);
  if (target === null || target === undefined) return;
  p = target;
  if (!force && p === state.currentPage && pdfPageImg && pdfPageImg.getAttribute("src")) return;
  state.currentPage = p;
  if (pdfPageLabel) {
    /* The PDF's own page number, because that is what is printed on the page
       and what the reader would look for - but counted against what this
       document contains, not against the file it was cut from. */
    const total = pages ? pages.length : state.pageCount;
    const nth = pages ? pages.indexOf(p) + 1 : p + 1;
    pdfPageLabel.textContent = pages
      ? `Page ${p + 1} (${nth} of ${total})`
      : `Page ${p + 1} of ${total}`;
  }
  const previous = nearestLoadedPage(p - 1, -1);
  const next = nearestLoadedPage(p + 1, 1);
  if (pdfPrevPage) pdfPrevPage.disabled = previous === null || previous >= p;
  if (pdfNextPage) pdfNextPage.disabled = next === null || next <= p;
  if (pdfPageImg) {
    pdfPageImg.src = `/api/page?doc=${state.docId}&page=${p}&dpi=150`;
  }
  // Prefetch the neighbours *this document has* (debounced). A page outside
  // the selection is not part of this document, and on a scanned book each
  // one is a multi-megabyte render nobody asked for.
  clearTimeout(pdfPrefetchTimer);
  pdfPrefetchTimer = setTimeout(() => {
    if (!state.splitView || !state.hasPages) return;
    if (next !== null && next > p) {
      const nextImg = new Image();
      nextImg.src = `/api/page?doc=${state.docId}&page=${next}&dpi=150`;
    }
    if (previous !== null && previous < p) {
      const prevImg = new Image();
      prevImg.src = `/api/page?doc=${state.docId}&page=${previous}&dpi=150`;
    }
  }, 120);
}

function syncPdfToCurrentBlock() {
  if (!state.splitView || !state.pdfAutoFollow) return;
  if (!state.blockPages || state.i < 0) return;
  const p = state.blockPages[state.i];
  if (p !== undefined && p !== null && p !== state.currentPage) {
    loadPdfPage(p);
  }
}

if (splitToggleBtn) splitToggleBtn.onclick = () => setSplitView(!state.splitView);
if (pdfCloseBtn) pdfCloseBtn.onclick = () => setSplitView(false);
/* Prev/Next step to the next page the document has, which on a selection is
   not always the adjacent number. */
if (pdfPrevPage) {
  pdfPrevPage.onclick = () => {
    const p = nearestLoadedPage(state.currentPage - 1, -1);
    if (p !== null && p < state.currentPage) {
      state.pdfAutoFollow = false;
      loadPdfPage(p);
      requestAnimationFrame(updateResumeBtnVisibility);
    }
  };
}
if (pdfNextPage) {
  pdfNextPage.onclick = () => {
    const p = nearestLoadedPage(state.currentPage + 1, 1);
    if (p !== null && p > state.currentPage) {
      state.pdfAutoFollow = false;
      loadPdfPage(p);
      requestAnimationFrame(updateResumeBtnVisibility);
    }
  };
}

if (pdfPageImg) {
  pdfPageImg.onclick = () => {
    if (!state.blockPages || !state.blockPages.length) return;
    let bi = state.blockPages.indexOf(state.currentPage);
    if (bi < 0) {
      bi = state.blockPages.findIndex((p) => p >= state.currentPage);
    }
    if (bi >= 0) {
      state.pdfAutoFollow = true;
      playBlock(bi, 0, { explicit: true });
    }
  };
}

/* ------------------------------------------------------------------ views */

function setView(view) {
  state.view = view;
  document.body.dataset.view = view;
  document.querySelectorAll(".viewBtn").forEach((b) => b.classList.toggle("active", b.dataset.view === view));
  closeAllPopovers();
  if (view === "library") {
    const p = heartbeat(true);
    if (p && p.finally) p.finally(() => refreshLibraryView());
    else refreshLibraryView();
  }
  if (view === "voices") refreshVoiceGallery();
  if (view === "reader") {
    state.autoFollow = true;
    const curBlock = state.i >= 0 ? state.i : (state.resumeSeek ? state.resumeSeek.block : 0);
    const p = $(`b-${curBlock}`);
    if (p) {
      markProgrammaticScroll(700);
      p.scrollIntoView({ block: "center", behavior: "smooth" });
    }
    /* Returning to the reader is a return to the reading position, pane
       included - whatever page was left showing behind the library. */
    state.pdfAutoFollow = true;
    if (state.splitView) {
      const activePage = state.blockPages ? state.blockPages[curBlock] : undefined;
      if (activePage !== undefined && activePage !== null) loadPdfPage(activePage);
    }
    requestAnimationFrame(updateResumeBtnVisibility);
  }
}
document.querySelectorAll(".viewBtn").forEach((b) => b.onclick = () => setView(b.dataset.view));

/* ------------------------------------------------------------------ open */

const dlg = $("openDlg");
const dlgOpen = () => dlg.open;
let openPollTimer = null;

function resetOpenDlg() {
  if (openPollTimer) { clearInterval(openPollTimer); openPollTimer = null; }
  $("openErr").hidden = true;
  $("openProgress").hidden = true;
  $("doOpen").disabled = false;
  $("fileInput").disabled = false;
}

$("openBtn").onclick = () => {
  resetOpenDlg();
  dlg.showModal();
  $("textInput").focus();
};
dlg.addEventListener("close", resetOpenDlg);

/* A PDF may need OCR - minutes, for a real scanned book - so /api/open and
   /api/upload hand back a job for one instead of blocking the request; a
   pasted-text or txt/epub open is fast enough to come back with the
   document directly. Route whichever shape actually came back. */
async function handleOpenResult(r, j, err) {
  if (!r.ok) {
    err.textContent = j.error || "could not open that";
    err.hidden = false;
    status("idle");
    $("doOpen").disabled = false;
    $("fileInput").disabled = false;
    return;
  }
  if (j.job) {
    pollOpen(j.job, err);
    return;
  }
  dlg.close();
  afterOpen(j);
}

function pollOpen(jobId, err) {
  $("doOpen").disabled = true;
  $("fileInput").disabled = true;
  $("openProgress").hidden = false;
  if (openPollTimer) clearInterval(openPollTimer);
  openPollTimer = setInterval(async () => {
    let r, j;
    try {
      r = await fetch(`/api/open/status?job=${jobId}`);
      j = await r.json();
    } catch {
      clearInterval(openPollTimer);
      err.textContent = "lost the connection while reading that file";
      err.hidden = false;
      $("doOpen").disabled = false;
      $("fileInput").disabled = false;
      return;
    }
    if (!r.ok || j.status === "error") {
      clearInterval(openPollTimer);
      err.textContent = j.error || "could not open that";
      err.hidden = false;
      $("doOpen").disabled = false;
      $("fileInput").disabled = false;
      return;
    }
    const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
    $("openBar").style.width = `${pct}%`;
    $("openProgressText").textContent = j.total
      ? `reading page ${j.done} of ${j.total}…`
      : "reading…";
    if (j.status === "done") {
      clearInterval(openPollTimer);
      dlg.close();
      afterOpen(j.doc);
    }
  }, 700);
}

document.addEventListener("keydown", (e) => {
  // never swallow keys while the open dialog has focus - it eats paste
  if (dlgOpen()) return;
  if (e.ctrlKey && e.code === "KeyF" && state.view === "reader" && state.blocks.length) {
    e.preventDefault();
    $("readerSearch").hidden ? openSearch() : $("searchInput").focus();
    return;
  }
  if (e.target.closest("input, textarea, dialog")) return;
  if (e.code === "Escape") { closeSearch(); closeAllPopovers(); }
  if (e.code === "Space") { e.preventDefault(); $("playBtn").click(); }
  if (e.code === "ArrowRight") $("nextBtn").click();
  if (e.code === "ArrowLeft") $("prevBtn").click();
});

/* picking a file uploads its bytes, so no one has to type a path */
$("fileInput").onchange = async (e) => {
  const f = e.target.files[0];
  if (!f) return;
  const err = $("openErr");
  err.hidden = true;
  status(`reading ${f.name}…`);
  const pages = $("pagesInput").value.trim();
  let r, j;
  try {
    r = await fetch("/api/upload", {
      method: "POST",
      headers: { "X-Filename": encodeURIComponent(f.name),
                 "X-Pages": encodeURIComponent(pages),
                 "Content-Type": "application/octet-stream" },
      body: f,
    });
    j = await r.json();
  } catch {
    err.textContent = "lost the connection while uploading that file";
    err.hidden = false;
    status("idle");
    return;
  }
  handleOpenResult(r, j, err);
};

function afterOpen(j) {
  state.playbackEpoch++;
  activeEl().pause();
  /* Before anything is torn down: a scheduled node outlives the DOM it was
     started for, and render() below is about to renumber nothing but rebuild
     everything. */
  waStopAll();
  state.playing = false;
  $("playBtn").textContent = "Play";
  setDocTitle(j.title);
  render(j.id, j.blocks, j.practice);
  state.voice = state.serverVoice = j.voice || null;
  updateVoiceBtn();
  if (pdfPageImg) pdfPageImg.removeAttribute("src");
  state.hasPages = !!j.has_pages;
  state.pageCount = j.page_count || 0;
  state.pageRuns = j.page_runs || null;
  state.blockPages = j.block_pages || [];
  /* Not 0 - that is a page this document may not contain. The pane opens on
     the first page of what was actually loaded; see currentPdfPage(). */
  state.currentPage = null;
  updateSplitViewUI();
  resetFigures();
  startIngestion(j.id, j.ingestion, j.content_revision);
  pollChaptersIfNeeded(j.id);
  pollFiguresIfNeeded(j.id);
  setView("reader");
}

$("doOpen").onclick = async (e) => {
  e.preventDefault();
  const path = $("pathInput").value.trim();
  const text = $("textInput").value.trim();
  const title = $("pasteTitle").value.trim();
  const pages = $("pagesInput").value.trim();
  if (!path && !text) { return; }

  const err = $("openErr");
  err.hidden = true;
  status("reading…");

  let r, j;
  try {
    r = await fetch("/api/open", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(text ? { text, title }
                                 : { path, options: pages ? { pages } : {} }),
    });
    j = await r.json();
  } catch {
    err.textContent = "lost the connection while reading that";
    err.hidden = false;
    status("idle");
    return;
  }
  if (r.ok && !j.job) { $("pasteTitle").value = ""; $("textInput").value = ""; }
  handleOpenResult(r, j, err);
};

/* ---------------------------------------------------------------- library */

let renamingId = null;   // guards against the periodic poll wiping out an in-progress edit

function fmtProgress(d) {
  const pct = d.percent || 0;
  if (d.total_seconds) {
    const mins = (s) => Math.round(s / 60);
    return `${mins(d.elapsed_seconds)} min / ${mins(d.total_seconds)} min (${pct}%)`;
  }
  return `${pct}%`;
}

function fmtWhen(iso) {
  if (!iso) return "";
  const d = new Date(iso);
  const days = Math.floor((Date.now() - d.getTime()) / 86400000);
  if (days <= 0) return "today";
  if (days === 1) return "yesterday";
  if (days < 7) return `${days} days ago`;
  return d.toLocaleDateString();
}

/* Background exports surface in the library so a job left running behind the
   reader is never invisible. Defined here rather than in the export section
   because it is the library that asks; state.activeExports is the source of
   truth either way, and an empty one means the rows look exactly as they did
   before any of this existed. */
function exportRunning(docId) {
  const rec = state.activeExports[docId];
  return rec && rec.status === "running" ? rec : null;
}

function exportRunningTitle(rec) {
  const of = rec.total ? ` (paragraph ${rec.done} of ${rec.total})` : "";
  return `Exporting in the background${of} - click to watch it`;
}

/* One function decides what a library entry's export controls say, because the
   shelf card and the full row are two renderings of the same document and must
   never disagree about it. The three phases are exclusive, which is the whole
   point: "running" deliberately shows no download, since a re-export reuses the
   package folder and the file the old link points at is the one being replaced.

     idle     [Export]                   [Delete]
     running  [Exporting...]             [Delete]
     done     [Download MP3] [Re-export] [Delete]

   Only the running phase is read from session state. Done is the server's
   answer and nothing else: `exported` means a finished export of exactly this
   text is still on disk (talkify.py's _is_exported, which checks the files
   rather than the index), and a session record still saying "done" about an
   MP3 the reader has since deleted would leave the row offering a download
   that 404s. There is no race to lose by trusting it - an export writes its
   index entry before it reports done, so the /api/docs fetch that follows a
   finished job already sees it. */
function exportPhase(d) {
  if (exportRunning(d.id)) return "running";
  return d.exported ? "done" : "idle";
}

/* Both ids, always. The server serves the job while this process still
   remembers it and falls back to the package index by document otherwise -
   which is what keeps the link alive across a restart, a re-export that
   failed while the previous file is still on disk, or a job id issued by a
   process that has since gone away. Sending only the job id gave that
   fallback nothing to fall back to, and the link 404'd with the MP3 sitting
   right there. */
function exportDownloadHref(d, rec) {
  const job = rec && rec.job;
  return "/api/export/download?" +
         (job ? `job=${encodeURIComponent(job)}&` : "") +
         `doc=${encodeURIComponent(d.id)}`;
}

async function refreshLibraryView() {
  if (renamingId) return;
  const [recentRes, docsRes] = await Promise.all([fetch("/api/recent?limit=3"), fetch("/api/docs")]);
  const recent = (await recentRes.json()).docs;
  const docs = (await docsRes.json()).docs;

  $("libShelfWrap").hidden = recent.length === 0;
  const cards = $("recentCards");
  cards.innerHTML = "";
  recent.forEach((d) => {
    const card = document.createElement("div");
    card.className = "lib-card";
    card.innerHTML = `<div class="card-title"></div>
      <div class="card-meta"></div>
      <div class="card-bar"><span></span></div>`;
    card.querySelector(".card-title").textContent = d.title;
    const curPara = (d.position && typeof d.position.block === "number") ? Math.min(d.position.block + 1, d.blocks) : 1;
    const cardPhase = exportPhase(d);
    const running = cardPhase === "running" ? exportRunning(d.id) : null;
    const cardActiveExp = state.activeExports[d.id];
    const cardReady = running ? Math.max(d.ready, running.done || 0) : d.ready;
    const warmMeta = cardReady < d.blocks
      ? (running ? ` · ⚡ Exporting ${running.done}/${running.total || d.blocks}` : ` · ⚡ Warmed ${cardReady}/${d.blocks}`)
      : ` · ✓ Fully Warmed`;
    card.querySelector(".card-meta").textContent =
      `Para ${curPara}/${d.blocks} · ${fmtWhen(d.last_opened)} · ${fmtProgress(d)}${warmMeta}`;
    card.querySelector(".card-bar > span").style.width = `${d.percent || 0}%`;
    card.onclick = () => openDocForReading(d.id);
    if (running) {
      /* An export started here and left running behind the reader is
         otherwise invisible. The badge is the way back into its progress. */
      const badge = document.createElement("button");
      badge.className = "ghost card-exporting exporting";
      badge.textContent = "Exporting\u2026";
      badge.title = exportRunningTitle(running);
      badge.onclick = (ev) => { ev.stopPropagation(); openExportDialog(d); };
      card.appendChild(badge);
    }
    if (cardPhase === "done") {
      const dlBtn = document.createElement("a");
      dlBtn.className = "btn-green-download";
      dlBtn.textContent = "Download MP3";
      dlBtn.href = exportDownloadHref(d, cardActiveExp);
      dlBtn.download = "";
      dlBtn.onclick = (ev) => ev.stopPropagation();
      card.appendChild(dlBtn);
    }
    cards.appendChild(card);
  });

  const query = $("libSearch").value.trim().toLowerCase();
  const list = $("libFullList");
  list.innerHTML = "";
  $("libEmptyMsg").hidden = docs.length > 0;
  docs.forEach((d) => {
    const li = document.createElement("li");
    li.className = "lib-row" + (d.id === state.docId ? " active" : "");
    if (query && !d.title.toLowerCase().includes(query)) li.classList.add("hidden-by-search");

    const curPara = (d.position && typeof d.position.block === "number") ? Math.min(d.position.block + 1, d.blocks) : 1;

    const title = document.createElement("span");
    title.className = "row-title";
    title.textContent = d.title;
    title.title = "Click to rename";
    title.onclick = (ev) => { ev.stopPropagation(); startRename(title, d); };

    const bar = document.createElement("span");
    bar.className = "row-bar";
    bar.innerHTML = "<span></span>";
    bar.querySelector("span").style.width = `${d.percent || 0}%`;

    const pos = document.createElement("span");
    pos.className = "row-pos";
    pos.textContent = `Para ${curPara}/${d.blocks}`;
    pos.title = `Reading position: paragraph ${curPara} of ${d.blocks}`;

    const pct = document.createElement("span");
    pct.className = "row-pct";
    pct.textContent = `${d.percent || 0}%`;

    const phase = exportPhase(d);
    const exporting = phase === "running" ? exportRunning(d.id) : null;
    const activeExp = state.activeExports[d.id];
    const readyCount = exporting ? Math.max(d.ready, exporting.done || 0) : d.ready;

    const warm = document.createElement("span");
    warm.className = "row-warm";
    if (readyCount < d.blocks) {
      if (exporting) {
        warm.textContent = `⚡ Exporting ${exporting.done}/${exporting.total || d.blocks}`;
        warm.title = `Export warming progress: ${exporting.done} of ${exporting.total || d.blocks} paragraphs`;
      } else {
        warm.textContent = `⚡ ${readyCount}/${d.blocks}`;
        warm.title = `${readyCount} of ${d.blocks} paragraphs synthesized and cached`;
      }
    } else {
      warm.textContent = "✓ Fully Warmed";
      warm.title = "All paragraphs synthesized and cached";
    }

    const actions = document.createElement("span");
    actions.className = "row-actions";
    if (phase === "done") {
      const dlBtn = document.createElement("a");
      dlBtn.className = "btn-green-download";
      dlBtn.textContent = "Download MP3";
      dlBtn.href = exportDownloadHref(d, activeExp);
      dlBtn.download = "";
      dlBtn.onclick = (ev) => ev.stopPropagation();
      actions.appendChild(dlBtn);
    }
    /* Once the MP3 exists, the download beside this button is how you get it,
       so "Export" could only ever mean one thing here: the voice was wrong. It
       says so, and it opens the dialog fresh rather than re-attaching to the
       finished job - the re-attach is what made this button a dead end. */
    const expBtn = document.createElement("button");
    expBtn.className = exporting ? "ghost exporting" : "ghost";
    expBtn.textContent = exporting ? "Exporting\u2026"
                       : phase === "done" ? "Re-export" : "Export";
    if (exporting) expBtn.title = exportRunningTitle(exporting);
    else if (phase === "done") {
      expBtn.title = d.exported_voice
        ? `Exported in ${voiceLabel(d.exported_voice)} - export again in a different voice`
        : "Export again in a different voice";
    }
    expBtn.onclick = (ev) => {
      ev.stopPropagation();
      openExportDialog(d, { fresh: phase === "done" });
    };
    actions.appendChild(expBtn);
    const delBtn = document.createElement("button");
    delBtn.className = "ghost";
    delBtn.textContent = "Delete";
    delBtn.onclick = async (ev) => {
      ev.stopPropagation();
      if (!confirm(`Delete "${d.title}"? This can't be undone.`)) return;
      await fetch("/api/delete", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc: d.id }),
      });
      if (d.id === state.docId) { state.docId = null; render(null, []); setDocTitle("Nothing open"); }
      refreshLibraryView();
    };
    actions.appendChild(delBtn);

    li.append(title, bar, pos, pct, warm, actions);
    li.onclick = () => openDocForReading(d.id);
    list.appendChild(li);
  });
}

$("libSearch").addEventListener("input", () => {
  const query = $("libSearch").value.trim().toLowerCase();
  document.querySelectorAll("#libFullList .lib-row").forEach((li) => {
    const t = li.querySelector(".row-title").textContent.toLowerCase();
    li.classList.toggle("hidden-by-search", !!query && !t.includes(query));
  });
});

/* Swaps a library row's title for an inline input. Renaming saves to the
   document's file in library/ server-side (talkify.py's _save_doc), so it
   survives a restart just like the document itself does. */
function startRename(titleEl, d) {
  renamingId = d.id;
  const input = document.createElement("input");
  input.className = "lib-rename";
  input.value = d.title;
  titleEl.replaceWith(input);
  input.focus();
  input.select();

  let done = false;
  const finish = () => { done = true; renamingId = null; };
  const commit = async () => {
    if (done) return;
    const val = input.value.trim();
    finish();
    if (val && val !== d.title) {
      await fetch("/api/rename", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc: d.id, title: val }),
      });
      if (d.id === state.docId) setDocTitle(val);
    }
    refreshLibraryView();
  };
  input.addEventListener("blur", commit);
  input.addEventListener("keydown", (ev) => {
    ev.stopPropagation();
    if (ev.key === "Enter") input.blur();
    if (ev.key === "Escape") { finish(); refreshLibraryView(); }
  });
  input.addEventListener("click", (ev) => ev.stopPropagation());
}

/* Opens a document into the Reader view and queues (but does not force-start)
   playback at the position saved from last time - see /api/position and
   _progress() in talkify.py. Actually starting audio still waits for the
   user to press Play, same as any other document. */
async function openDocForReading(docId) {
  if (docId === state.docId) { setView("reader"); return; }
  const r = await fetch(`/api/doc?doc=${docId}`);
  if (!r.ok) return;
  const j = await r.json();
  state.playbackEpoch++;
  activeEl().pause();
  /* Before anything is torn down: a scheduled node outlives the DOM it was
     started for, and render() below is about to renumber nothing but rebuild
     everything. */
  waStopAll();
  state.playing = false;
  $("playBtn").textContent = "Play";
  setDocTitle(j.title);
  render(j.id, j.blocks, j.practice);
  state.voice = state.serverVoice = j.voice || null;
  updateVoiceBtn();
  if (pdfPageImg) pdfPageImg.removeAttribute("src");
  state.hasPages = !!j.has_pages;
  state.pageCount = j.page_count || 0;
  state.pageRuns = j.page_runs || null;
  state.blockPages = j.block_pages || [];
  /* Not 0 - that is a page this document may not contain. The pane opens on
     the first page of what was actually loaded; see currentPdfPage(). */
  state.currentPage = null;
  updateSplitViewUI();
  state.chapters = j.chapters || [];
  state.generation = typeof j.generation === "number" ? j.generation : 0;
  updateChaptersUI();
  resetFigures();
  startIngestion(j.id, j.ingestion, j.content_revision);
  pollChaptersIfNeeded(j.id);
  pollFiguresIfNeeded(j.id);

  const pos = j.position || { block: 0, word: 0, within: 0 };
  const totalBlocks = (j.blocks || []).length;
  if (pos.block > 0 || pos.within > 0) {
    state.resumeSeek = pos;
    const p = $(`b-${pos.block}`);
    if (p) p.classList.add("here");
    status(`${pos.block + 1} of ${totalBlocks}`);
  } else {
    status(totalBlocks > 0 ? `1 of ${totalBlocks}` : "idle");
  }
  state.autoFollow = true;
  setView("reader");
  requestAnimationFrame(() => {
    const p = $(`b-${pos.block}`);
    if (p) { markProgrammaticScroll(700); p.scrollIntoView({ block: "center" }); }
  });
}

/* -------------------------------------------------------------- mp3 export */

const exportDlg = $("exportDlg");

/* An export outlives the dialog that started it. Closing the modal to carry
   on reading used to abandon the poll: the server kept encoding, but the
   client forgot the job existed, so re-opening offered a fresh export of a
   book that was already three quarters of the way through one. So the job
   lives in state, keyed by document, and polling is a background thing the
   dialog attaches to rather than something the dialog owns.

     state.activeExports[docId] = {job, mode, voice, doc, status, done, total}

   The record only has to survive long enough to find the job again: what the
   re-opened dialog renders comes from asking the server, so a finished export
   comes straight back up with its download links. (state.activeExports is
   declared with the rest of state, up top.) */
const exportTimers = {};        // docId -> setInterval handle, while running
let exportDlgDoc = null;        // the library summary the dialog is showing
let exportIngestTimer = null;   // ingestion poll attached to the open dialog

function stopExportIngestionPolling() {
  if (exportIngestTimer) clearInterval(exportIngestTimer);
  exportIngestTimer = null;
}

function updateExportIngestUI(doc) {
  const ing = (doc.ingestion && doc.ingestion.state) || "complete";
  const note = $("exportIngesting");
  const scanning = ing === "running";
  if (note) {
    note.hidden = !scanning;
    if (scanning) {
      const done = doc.ingestion.pages_processed || 0;
      const total = doc.ingestion.pages_total || 0;
      note.textContent = total
        ? `Available when scanning finishes — ${done} of ${total} pages read.`
        : "Available when scanning finishes.";
    }
  }
  $("exportStartBtn").disabled = scanning;
  return scanning;
}

function startExportIngestionPolling(doc) {
  stopExportIngestionPolling();
  if (!doc || !updateExportIngestUI(doc)) return;
  const tick = async () => {
    if (!exportDlg.open || exportDlgDoc !== doc || exportDlg.dataset.doc !== doc.id) return;
    try {
      const r = await fetch(`/api/doc/blocks?doc=${encodeURIComponent(doc.id)}&from=0`);
      if (!r.ok) return;
      const reply = await r.json();
      if (!exportDlg.open || exportDlgDoc !== doc || exportDlg.dataset.doc !== doc.id) return;
      if (reply.ingestion) doc.ingestion = reply.ingestion;
      if (!updateExportIngestUI(doc)) stopExportIngestionPolling();
    } catch { /* retry on the next interval */ }
  };
  exportIngestTimer = setInterval(tick, 1000);
  tick();
}

function stopExportPolling(docId) {
  if (exportTimers[docId]) clearInterval(exportTimers[docId]);
  delete exportTimers[docId];
}

function forgetExport(docId) {
  stopExportPolling(docId);
  delete state.activeExports[docId];
}

/* null means the server has no such job - it finished long ago and was
   swept, or the server restarted under us. Either way the client should
   forget it rather than poll a ghost forever. */
async function fetchExportStatus(jobId) {
  try {
    const r = await fetch(`/api/export/status?job=${encodeURIComponent(jobId)}`);
    const j = await r.json();
    return r.ok ? j : null;
  } catch (e) {
    return null;
  }
}

/* `opts.fresh` is the library's "Re-export" button saying the user wants to
   make another one, not look at the last one. Without it, a finished job is
   re-attached to and the dialog comes back up showing the download the row
   was already offering - which is the dead end this whole change is about.

   It is honoured only for a job the server says is `done`. A *running* export
   still re-attaches to its progress no matter which button was pressed: the
   server runs one export per document, so there is no second one to start. */
async function openExportDialog(doc, opts) {
  const fresh = !!(opts && opts.fresh);
  $("exportDocTitle").textContent = doc.title;
  exportDlg.dataset.doc = doc.id;
  exportDlgDoc = doc;
  resetExportDialog(doc);
  exportDlg.showModal();

  const rec = state.activeExports[doc.id];
  if (rec) {
    /* Ask the server rather than trusting the last thing we saw: while the
       dialog was closed the job may have finished, failed, or vanished. */
    const live = await fetchExportStatus(rec.job);
    if (live && live.status === "running") {
      lockExportControls(doc.id);
      renderExportStatus(doc.id, live);
      startExportPolling(doc.id);
      return;
    }
    if (live && live.status === "done") {
      if (!fresh) {
        lockExportControls(doc.id);
        renderExportStatus(doc.id, live);
        return;
      }
      /* Deliberately kept, not forgotten: the record still names the job the
         row's download link points at, and starting another export replaces
         it anyway. */
    } else {
      forgetExport(doc.id);     // errored, or gone - offer a fresh export
      refreshLibraryView();
    }
  }
  /* After the picker, not before: voiceLabel() can only turn an id into a
     readable name once the catalogue is loaded, and fillExportVoices is what
     loads it on a cold start. */
  await fillExportVoices(doc);
  if (fresh) noteExportedVoice(doc);
}

/* Names the voice already on disk, so "I regret the voice" is something the
   reader can act on rather than guess at. The server's `exported_voice` is
   what survives a reload; the in-session record is the fresher answer right
   after a job finishes. Neither is required - with no answer the dialog is
   simply the first-export dialog, which is the honest thing to show. */
function noteExportedVoice(doc) {
  const rec = state.activeExports[doc.id];
  /* This session's finished job is the fresher answer and is taken whole,
     including when it is empty: "" from it means the export ran in the engine's
     default voice, which is an answer, not a reason to fall back to the older
     one in the summary and name a voice that is no longer on disk. An empty
     answer from either source says nothing at all. */
  const voice = (rec && rec.status === "done")
    ? (rec.voice || "") : (doc.exported_voice || "");
  const el = $("exportPrevVoice");
  if (!el) return;
  if (!voice) { el.hidden = true; return; }
  el.textContent = `Already exported in ${voiceLabel(voice)}. ` +
                   "Picking a different voice replaces it.";
  el.hidden = false;
  $("exportStartBtn").textContent = "Export again";
}

exportDlg.onclose = stopExportIngestionPolling;

function resetExportDialog(doc) {
  stopExportIngestionPolling();
  $("exportProgress").hidden = true;
  $("exportBar").style.width = "0%";
  $("exportProgressText").textContent = "";
  $("exportErr").hidden = true;
  $("exportDone").hidden = true;
  $("exportFiles").hidden = true;
  $("exportFiles").textContent = "";
  $("exportDownloadBtn").hidden = true;
  $("exportStartBtn").hidden = false;
  $("exportStartBtn").disabled = false;
  $("exportStartBtn").textContent = "Export as MP3";
  if ($("exportPrevVoice")) $("exportPrevVoice").hidden = true;
  if ($("exportRedoBtn")) $("exportRedoBtn").hidden = true;
  $("exportActiveVoice").hidden = true;
  $("exportActiveVoice").textContent = "";
  $("exportVoiceWrap").hidden = false;
  $("exportModeWrap").hidden = !(doc.chapters > 0);
  exportDlg.querySelector('input[name="exportMode"][value="full"]').checked = true;
  /* A book still coming off the scanner has no full version to export yet.
     The server refuses it (409) rather than writing a prefix under the whole
     book's name; saying so here means the button is never a trap. */
  if (updateExportIngestUI(doc)) startExportIngestionPolling(doc);
}

/* Once the job is running the picker is a lie: the voice and the mode are
   fixed server-side for the life of that job, and a dropdown that still
   moves invites the user to think it changed something. Replace both with
   one plain line saying what is actually being made. */
function lockExportControls(docId) {
  $("exportVoiceWrap").hidden = true;
  $("exportModeWrap").hidden = true;
  renderActiveVoiceLine(state.activeExports[docId] || {});
  /* Re-opening onto a running job must not offer to start a second one. */
  $("exportStartBtn").disabled = true;
  $("exportProgress").hidden = false;
}

/* Kept separate because the voice can be filled in a moment after the line
   first appears: choosing "the document's own voice" sends "" and only the
   first status reply says which voice that actually turned out to be. */
function renderActiveVoiceLine(rec) {
  const el = $("exportActiveVoice");
  el.textContent = "Voice: ";
  const strong = document.createElement("strong");
  strong.textContent = rec.voice ? voiceLabel(rec.voice) : "Default voice";
  el.append(strong, rec.mode === "chapters"
                    ? " (One MP3 per chapter)" : " (Full document)");
  el.hidden = false;
}

/* A dead job is the only thing that hands the controls back - picking another
   voice and trying again is the sensible next move. The picker is refilled
   rather than just unhidden: a dialog that re-attached to an already-running
   export never filled it, and an empty dropdown is worse than no dialog. */
async function restoreExportControls(doc) {
  $("exportActiveVoice").hidden = true;
  $("exportVoiceWrap").hidden = false;
  $("exportModeWrap").hidden = !(doc && doc.chapters > 0);
  $("exportStartBtn").hidden = false;
  $("exportStartBtn").disabled = false;
  if (doc) await fillExportVoices(doc);
}

/* The export voice picker.

   Trying a few voices mid-read leaves a book with its blocks cached under
   several of them, and the export has to pick one - every paragraph not
   already made in that voice gets synthesized during the export. So the
   counts go in the option text and the voices that already have audio sort
   first: choosing the top one is the difference between stitching cached
   audio and reading the whole book again.

   Everything here is best-effort. If the counts cannot be fetched the picker
   still lists every voice, and if the voice catalogue itself is empty the
   dialog exports in the document's own voice exactly as it did before this
   existed - the select simply has one option. */
async function fillExportVoices(doc) {
  const sel = $("exportVoiceSelect");
  const hint = $("exportVoiceHint");
  sel.textContent = "";
  if (!state.voices.length) {
    try { await loadVoices(); } catch (e) { /* catalogue stays empty */ }
  }
  const total = Array.isArray(doc.blocks) ? doc.blocks.length : (doc.blocks || 0);
  let current = doc.voice || "";
  let counts = doc.cached_voices || {};
  try {
    /* The library summaries, not /api/doc: asking for the document itself
       marks it opened and hands it the active warm tier, which is wrong for a
       book you are only exporting while reading a different one. */
    const r = await fetch("/api/docs");
    const fresh = ((await r.json()).docs || []).find((d) => d.id === doc.id);
    if (fresh) {
      current = fresh.voice || "";
      counts = fresh.cached_voices || {};
    }
  } catch (e) { /* keep whatever the card already knew */ }

  const opts = state.voices.map((v) => ({ id: v.id, label: v.label,
                                          count: counts[v.id] || 0 }));
  if (!opts.some((o) => o.id === current)) {
    /* The document's own voice always has to be selectable, even if it is the
       engine default (id "") or one the running engine no longer lists. */
    opts.unshift({ id: current, label: current ? voiceLabel(current) : "Default voice",
                   count: counts[current] || 0 });
  }
  const cached = opts.filter((o) => o.count > 0).sort((a, b) => b.count - a.count);
  const rest = opts.filter((o) => o.count === 0);
  [...cached, ...rest].forEach((o, i) => {
    const el = document.createElement("option");
    el.value = o.id;
    el.textContent = o.count
      ? `${o.label} - ${o.count} paragraph${o.count === 1 ? "" : "s"} ready` +
        (i === 0 ? " (fastest)" : "")
      : o.label;
    el.selected = o.id === current;
    sel.appendChild(el);
  });

  const describe = () => {
    const n = counts[sel.value] || 0;
    hint.textContent = n
      ? `${n} of ${total} paragraphs are already synthesized in this voice; the rest are made as the export runs.`
      : "Default: last used voice. Choosing a voice with cached paragraphs exports faster.";
  };
  sel.onchange = describe;
  describe();
}

$("exportStartBtn").onclick = async (e) => {
  e.preventDefault();
  const docId = exportDlg.dataset.doc;
  $("exportStartBtn").disabled = true;
  $("exportErr").hidden = true;
  const mode = exportDlg.querySelector('input[name="exportMode"]:checked').value;
  const voice = $("exportVoiceSelect").value;   // "" means the document's own
  const r = await fetch("/api/export", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ doc: docId, mode, voice }),
  });
  const j = await r.json();
  if (!r.ok) {
    $("exportErr").textContent = j.error || "export failed";
    $("exportErr").hidden = false;
    /* A book that is still being scanned has nothing to retry yet, so the
       button stays disabled and the line above it says when it will. Anything
       else is worth another go. */
    if (j.code === "ingestion_running") {
      const note = $("exportIngesting");
      if (note) {
        const ing = j.ingestion || {};
        note.hidden = false;
        note.textContent = ing.pages_total
          ? `Available when scanning finishes — ${ing.pages_processed || 0} of ${ing.pages_total} pages read.`
          : "Available when scanning finishes.";
      }
    } else {
      $("exportStartBtn").disabled = false;
    }
    return;
  }
  state.activeExports[docId] = { job: j.job, mode, voice, doc: exportDlgDoc,
                                 status: "running", done: 0, total: 0 };
  lockExportControls(docId);
  startExportPolling(docId);
  refreshLibraryView();          // the row reads "Exporting..." from now on
};

/* Hands a finished dialog back to the voice picker. Everything the done state
   put on screen goes away first: leaving "Saved to ..." above a fresh Export
   button reads as though the new export has already happened. */
if ($("exportRedoBtn")) {
  $("exportRedoBtn").onclick = async (e) => {
    e.preventDefault();
    const docId = exportDlg.dataset.doc;
    const doc = exportDlgDoc;
    $("exportRedoBtn").hidden = true;
    $("exportDone").hidden = true;
    $("exportFiles").hidden = true;
    $("exportFiles").textContent = "";
    $("exportDownloadBtn").hidden = true;
    $("exportProgress").hidden = true;
    $("exportBar").style.width = "0%";
    $("exportProgressText").textContent = "";
    await restoreExportControls(doc);
    if (doc && docId) noteExportedVoice(doc);
  };
}

function startExportPolling(docId) {
  if (exportTimers[docId]) return;
  exportTimers[docId] = setInterval(() => tickExport(docId), 700);
  tickExport(docId);             // don't make the first frame wait 700ms
}

async function tickExport(docId) {
  const rec = state.activeExports[docId];
  if (!rec) return stopExportPolling(docId);
  const j = await fetchExportStatus(rec.job);
  if (!j) {                      // the job is gone; nothing left to watch
    const showing = exportDlgShowing(docId);
    const doc = rec.doc;
    forgetExport(docId);
    if (showing) {
      $("exportErr").textContent = "this export job is no longer available";
      $("exportErr").hidden = false;
      await restoreExportControls(doc);
    }
    refreshLibraryView();
    return;
  }
  renderExportStatus(docId, j);
  if (j.status !== "running") {
    stopExportPolling(docId);
    refreshLibraryView();
  }
}

function exportDlgShowing(docId) {
  return exportDlg.open && exportDlg.dataset.doc === docId;
}

/* Renders one status payload into the dialog. Always updates the stored
   record - the library rows read it - and only touches the DOM when the
   dialog happens to be showing this document, so an export running behind
   the reader costs nothing but the poll. */
function renderExportStatus(docId, j) {
  const rec = state.activeExports[docId];
  if (rec) {
    rec.status = j.status;
    rec.done = j.done || 0;
    rec.total = j.total || 0;
    if (j.voice) rec.voice = j.voice;
    if (j.files) rec.mode = "chapters";
  }
  if (!exportDlgShowing(docId)) return;
  const jobId = rec ? rec.job : null;
  if (rec && !$("exportActiveVoice").hidden) renderActiveVoiceLine(rec);

  const pct = j.total ? Math.round((j.done / j.total) * 100) : 0;
  $("exportBar").style.width = `${pct}%`;
  $("exportProgressText").textContent =
    `Exporting paragraph ${j.done} of ${j.total} in ${voiceLabel(j.voice)}`;
  if (j.status === "done") {
    $("exportDone").hidden = false;
    if (j.files) {
      // One row per chapter, each with its own link - the folder path is
      // said once above them rather than repeated on every line.
      $("exportDone").textContent =
        `Saved ${j.files.length} chapter files to ${j.dir || "your export folder"}`;
      const list = $("exportFiles");
      list.textContent = "";
      j.files.forEach((f, i) => {
        const li = document.createElement("li");
        const name = document.createElement("span");
        name.className = "chap-title";
        name.textContent = `${String(i + 1).padStart(2, "0")}. ${f.title}`;
        const a = document.createElement("a");
        a.textContent = "Download";
        a.href = `/api/export/download?job=${jobId}&file=${i}`;
        li.append(name, a);
        list.append(li);
      });
      list.hidden = false;
    } else {
      $("exportDone").textContent = `Saved to ${j.path}`;
      $("exportDownloadBtn").textContent = "Download MP3";
      $("exportDownloadBtn").href = `/api/export/download?job=${jobId}`;
      $("exportDownloadBtn").hidden = false;
    }
    $("exportStartBtn").hidden = true;
    /* Finishing with the dialog open is the other door onto the same dead
       end: the download is here, and nothing offers to try another voice.
       This does. */
    if ($("exportRedoBtn")) $("exportRedoBtn").hidden = false;
  } else if (j.status === "error") {
    $("exportErr").textContent = j.error || "export failed";
    $("exportErr").hidden = false;
    restoreExportControls(rec && rec.doc);
  }
}

/* Export jobs continue in the background when the dialog closes; only the
   dialog's ingestion-status timer is stopped because it has no visible owner. */

let libTimer = null;
function ensureLibraryPolling() {
  if (libTimer) return;
  libTimer = setInterval(() => { if (state.view === "library") refreshLibraryView(); }, 3000);
}

/* ------------------------------------------------------------------ boot */

/* Library view is home. Nothing is
   auto-opened into the reader; server-side position tracking means one click
   from the library gets you back to exactly where you left off. */
loadVoices();
checkAiAvailable();
refreshLibraryView();
ensureLibraryPolling();
setView("library");
