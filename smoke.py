#!/usr/bin/env python3
"""
Everything about Talkify that can be checked without a browser.

    python smoke.py

Run it after any change to the server, engines or extraction. It starts a
throwaway server on a spare port, exercises the real endpoints, and asserts the
invariants that actually matter. It cannot see the page - the browser pass is
still yours.
"""
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

PORT = 8799
BASE = f"http://127.0.0.1:{PORT}"
SAMPLE = ("The Coast Miwok leader delivered a speech. Drake reciprocated by "
          "giving him a hat, and the ships sailed on.")

# A worker with no models in it. engines.WorkerClient is generic about what
# it spawns, which is exactly what makes the response dispatcher testable:
# this speaks the same JSON-line protocol and can be told, through the
# request text, to do the awkward things a real worker does only by
# accident - reply out of order, print a line that is not JSON, answer with
# an id nobody is waiting on, or die mid-conversation.
FAKE_WORKER = r"""import base64
import json
import sys

held = []


def reply(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def answer(req_id, text):
    words = text.split()
    heard = [{"w": w, "s": i * 0.2, "e": i * 0.2 + 0.15} for i, w in enumerate(words)]
    result = {"req_id": req_id, "ok": True,
              "audio_b64": base64.b64encode(b"WAVE" + text.encode()).decode("ascii"),
              "heard": heard}
    if text.startswith("P S I"):
        result["alignment_method"] = "kokoro-pytorch-duration-v1"
    return result


for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    req = json.loads(line)
    op = req.get("op", "synth")
    if op == "shutdown":
        break
    if op == "cancel":
        for i, (rid, text) in enumerate(held):
            if rid == req.get("req_id"):
                held.pop(i)
                reply({"req_id": rid, "ok": False,
                       "error": {"code": "cancelled", "stage": "queued",
                                 "message": "request superseded",
                                 "retryable": False, "fatal": False}})
                break
        continue
    rid, text = req["req_id"], req.get("text", "")
    if text == "die":
        sys.exit(1)
    if text == "hold":
        held.append((rid, text))
        continue
    if text == "release":
        # Answer this one first, then everything held - so replies arrive in
        # the opposite order to the requests.
        reply(answer(rid, text))
        while held:
            hid, htext = held.pop(0)
            reply(answer(hid, htext))
        continue
    if text == "stray":
        sys.stdout.write("torch: some warning nobody asked for\n")
        reply({"req_id": "nobody-is-waiting-on-this", "ok": True,
               "audio_b64": "", "heard": []})
        reply(answer(rid, text))
        continue
    if text == "fail":
        reply({"req_id": rid, "ok": False,
               "error": {"code": "alignment_failed", "stage": "asr",
                         "message": "ValueError: boom", "retryable": False,
                         "fatal": False}})
        continue
    reply(answer(rid, text))
"""

ok = fail = 0


def check(label, cond, detail=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  pass  {label}")
    else:
        fail += 1
        print(f"  FAIL  {label}{('  -> ' + detail) if detail else ''}")


def get(path):
    """Non-2xx is data here, not an exception - we assert on status codes."""
    st, body, ctype, _headers = get_full(path)
    return st, body, ctype


def get_full(path):
    """Same, plus the headers - some invariants (whether a preview was served
    from cache) are only visible there."""
    try:
        with urllib.request.urlopen(BASE + path, timeout=60) as r:
            return r.status, r.read(), r.headers.get("Content-Type", ""), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers.get("Content-Type", ""), e.headers


def post(path, obj):
    """Non-2xx is data here too, same as get() - callers assert on it."""
    req = urllib.request.Request(
        BASE + path, data=json.dumps(obj).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def start_server(env):
    srv = subprocess.Popen(
        [sys.executable, "talkify.py", "--engine", "stub",
         "--port", str(PORT), "--no-browser"],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, env=env)
    for _ in range(40):
        try:
            get("/")
            return srv
        except Exception:
            time.sleep(0.25)
    srv.terminate()
    raise RuntimeError("server never came up")


def main():
    # A throwaway library dir - this hits /api/open for real, and it must
    # never write test documents into the user's actual library.
    tmp_library = tempfile.mkdtemp(prefix="talkify_smoke_library_")
    tmp_config = os.path.join(tempfile.mkdtemp(prefix="talkify_smoke_config_"), "config.json")
    tmp_exports = tempfile.mkdtemp(prefix="talkify_smoke_exports_")
    env = {**os.environ, "TALKIFY_LIBRARY_DIR": tmp_library, "TALKIFY_CONFIG_PATH": tmp_config,
           "TALKIFY_EXPORT_DIR": tmp_exports}
    try:
        srv = start_server(env)
    except RuntimeError as e:
        print(e)
        shutil.rmtree(tmp_library, ignore_errors=True)
        return 1
    try:
        print("static")
        for p, kind in (("/", "text/html"), ("/web/app.js", "javascript"),
                        ("/web/style.css", "css")):
            st, body, ctype = get(p)
            check(f"{p} served", st == 200 and len(body) > 100, f"status {st}")
            check(f"{p} content-type", kind in ctype, ctype)

        print("open + extract")
        st, doc = post("/api/open", {"text": SAMPLE})
        check("text opens", st == 200 and doc["blocks"], str(doc)[:80])
        check("open returns a document id", st == 200 and doc.get("id"), str(doc)[:80])
        doc_id = doc["id"]
        blocks = doc["blocks"]
        check("blocks are short enough to keep latency down",
              all(len(b) <= 400 for b in blocks),
              f"longest {max(len(b) for b in blocks)}")

        print("opening a PDF (async job - OCR on a real scan can take minutes,"
              " so this must never block the request)")
        import pymupdf
        pdf_path = os.path.join(tmp_library, "smoke_test.pdf")
        pdoc_gen = pymupdf.open()
        page = pdoc_gen.new_page()
        page.insert_text((72, 72), "A tiny test document for smoke.py.")
        pdoc_gen.save(pdf_path)
        pdoc_gen.close()

        st, body = post("/api/open", {"path": pdf_path})
        check("opening a PDF returns a job, not a blocking document",
              st == 200 and body.get("job"), str(body)[:120])
        job_id = body.get("job")
        job = None
        for _ in range(40):
            st, jbody, _ = get(f"/api/open/status?job={job_id}")
            job = json.loads(jbody)
            if job["status"] != "running":
                break
            time.sleep(0.25)
        check("PDF open job finishes", job and job["status"] == "done", str(job)[:160])
        if job and job["status"] == "done":
            pdf_doc = job["doc"]
            check("the finished job carries the opened document",
                  pdf_doc.get("id") and pdf_doc.get("blocks"), str(pdf_doc)[:160])
            check("the PDF's actual text made it into a block",
                  any("tiny test document" in b for b in pdf_doc["blocks"]),
                  str(pdf_doc["blocks"])[:160])
        st, body, _ = get("/api/open/status?job=nonexistent")
        check("status for an unknown open job 404s cleanly", st == 404, str(body)[:80])

        print("extraction triage view (shows what's kept vs. dropped, and why)")
        st_triage, body_triage, _ = get("/triage")
        check("/triage serves triage.html cleanly", st_triage == 200 and b"Extraction triage" in body_triage, str(st_triage))
        import urllib.parse
        st, body, _ = get(f"/api/diagnose?path={urllib.parse.quote(pdf_path)}")
        diag = json.loads(body)
        check("diagnose returns one entry per requested page",
              st == 200 and len(diag.get("pages", [])) == 1, str(diag)[:160])
        if st == 200 and diag.get("pages"):
            dpage = diag["pages"][0]
            check("diagnose includes a rendered page image",
                  bool(dpage.get("image_b64")), str(dpage)[:120])
            check("the real text on the page is tagged as kept (body)",
                  any(l["category"] == "body" and "tiny test document" in l["text"]
                      for l in dpage["lines"]),
                  str(dpage["lines"])[:200])
        st, body, _ = get("/api/diagnose?path=nonexistent.pdf")
        check("diagnosing a missing file 400s cleanly", st == 400, str(body)[:80])

        print("...and a page range skips front matter instead of reading all of it")
        multi_path = os.path.join(tmp_library, "smoke_test_multipage.pdf")
        mdoc = pymupdf.open()
        mdoc.new_page().insert_text((72, 72), "Cover page nobody wants read aloud.")
        mdoc.new_page().insert_text((72, 72), "Real chapter one content starts here.")
        mdoc.new_page().insert_text((72, 72), "Trailing page three material.")
        mdoc.save(multi_path)
        mdoc.close()

        st, body = post("/api/open", {"path": multi_path, "options": {"pages": "2"}})
        job_id = body.get("job")
        job = None
        for _ in range(40):
            st, jbody, _ = get(f"/api/open/status?job={job_id}")
            job = json.loads(jbody)
            if job["status"] != "running":
                break
            time.sleep(0.25)
        check("page-range open finishes", job and job["status"] == "done", str(job)[:160])
        pdf_doc_id = None
        if job and job["status"] == "done":
            pdf_doc_id = job["doc"]["id"]
            check("newly opened PDF doc reports has_pages", job["doc"].get("has_pages") is True, str(job["doc"]))
            check("newly opened PDF doc reports correct page_count", job["doc"].get("page_count") == 3, str(job["doc"]))
            joined = " ".join(job["doc"]["blocks"])
            check("the requested page's text is present",
                  "Real chapter one" in joined, joined[:160])
            check("the page before the range is excluded",
                  "Cover page" not in joined, joined[:160])
            check("the page after the range is excluded",
                  "Trailing page three" not in joined, joined[:160])
            check("...and the document says which pages it actually is",
                  job["doc"].get("page_runs") == [[1, 1]],
                  str(job["doc"].get("page_runs")))
            st, png, ctype = get(f"/api/page?doc={pdf_doc_id}&page=1&dpi=72")
            check("the page view serves a page inside the selection",
                  st == 200 and ctype == "image/png", f"{st} {ctype}")
            st, _body, _c = get(f"/api/page?doc={pdf_doc_id}&page=0&dpi=72")
            check("...and refuses one before it, even though the file has it",
                  st == 404, str(st))
            st, _body, _c = get(f"/api/page?doc={pdf_doc_id}&page=2&dpi=72")
            check("...and one after it",
                  st == 404, str(st))
            st, dbody, _c = get(f"/api/doc?doc={pdf_doc_id}")
            check("...and /api/doc reports the same range, so a reopened tab"
                  " scopes its page view too",
                  json.loads(dbody).get("page_runs") == [[1, 1]], dbody[:160])

        print("library")
        st, body, _ = get("/api/docs")
        docs = json.loads(body)["docs"]
        check("newly opened document appears in the library",
              any(d["id"] == doc_id for d in docs), str(docs)[:120])
        st2, _ = post("/api/open", {"text": "A second, unrelated paragraph."})
        st3, body, _ = get(f"/api/doc?doc={doc_id}")
        check("opening a second document doesn't evict the first",
              st2 == 200 and st3 == 200, f"{st2}, {st3}")

        print("timings")
        st, body, _ = get(f"/api/block?doc={doc_id}&i=0")
        words = json.loads(body)["words"]
        src = blocks[0].split()
        check("every source word is timed", len(words) == len(src),
              f"{len(words)} timings vs {len(src)} words")
        check("timings are monotonic",
              all(words[i]["s"] <= words[i + 1]["s"] for i in range(len(words) - 1)))
        check("no zero-length words", all(w["e"] > w["s"] for w in words))
        check("first word starts at zero", words[0]["s"] < 0.05)

        print("audio")
        st, audio, ctype = get(f"/api/audio?doc={doc_id}&i=0")
        check("audio returned", st == 200 and len(audio) > 1000, f"{len(audio)} bytes")
        check("audio mime is sane", ctype.startswith("audio/"), ctype)
        if ctype == "audio/wav":
            import io, wave
            with wave.open(io.BytesIO(audio)) as w:
                dur = w.getnframes() / w.getframerate()
            check("audio covers the last word",
                  dur >= words[-1]["e"] - 0.25, f"audio {dur:.2f}s vs last {words[-1]['e']:.2f}s")

        print("background warming")
        for _ in range(40):
            st, body, _ = get("/api/docs")
            d = next(d for d in json.loads(body)["docs"] if d["id"] == doc_id)
            if d["ready"] >= d["blocks"]:
                break
            time.sleep(0.25)
        check("a document keeps synthesizing in the background unprompted",
              d["ready"] >= d["blocks"], f"{d['ready']}/{d['blocks']} after waiting")

        print("errors are handled, not crashes")
        st, body, _ = get(f"/api/block?doc={doc_id}&i=9999")
        check("out-of-range block 404s cleanly", st == 404, f"got {st}")
        check("...and says why in JSON", b"error" in body, body[:60].decode("utf-8", "replace"))
        st, body, _ = get("/api/block?doc=nonexistent&i=0")
        check("unknown document 404s cleanly too", st == 404, f"got {st}")

        print("naming and persistence")
        st, body = post("/api/rename", {"doc": doc_id, "title": "Renamed by smoke.py"})
        check("rename succeeds", st == 200 and body["title"] == "Renamed by smoke.py", str(body)[:80])
        st, body, _ = get("/api/docs")
        renamed = next(d for d in json.loads(body)["docs"] if d["id"] == doc_id)
        check("renamed title shows up in the library",
              renamed["title"] == "Renamed by smoke.py", renamed["title"])
        st, body = post("/api/rename", {"doc": doc_id, "title": "   "})
        check("renaming to blank is rejected, not silently accepted", st == 400, str(body)[:80])
        saved_path = os.path.join(tmp_library, f"{doc_id}.json")
        check("the document's text is actually written to the library on disk",
              os.path.exists(saved_path), saved_path)
        with open(saved_path, encoding="utf-8") as f:
            on_disk = json.load(f)
        check("the saved copy reflects the rename, not the original title",
              on_disk["title"] == "Renamed by smoke.py", on_disk["title"])

        # The JSON file above is the contract; library.db is the durable index
        # beside it. Read it from here, as a separate process, exactly the way
        # a later feature (search, recents) would.
        import sqlite3
        db_file = os.path.join(tmp_library, "library.db")
        check("the running server keeps a SQLite index next to the JSON files",
              os.path.exists(db_file), db_file)
        if os.path.exists(db_file):
            probe = sqlite3.connect(db_file)
            try:
                titles = probe.execute(
                    "SELECT title FROM documents WHERE id = ?", (doc_id,)).fetchall()
                nblocks = probe.execute(
                    "SELECT COUNT(*) FROM document_blocks WHERE document_id = ?",
                    (doc_id,)).fetchone()[0]
            finally:
                probe.close()
            check("the index carries the document, with the renamed title",
                  titles and titles[0][0] == "Renamed by smoke.py", str(titles))
            check("...and one indexed row per block of its text",
                  nblocks == len(blocks), f"{nblocks} rows vs {len(blocks)} blocks")

        # Position heartbeats are coalesced on the server (talkify.POSITION_
        # SAVE_INTERVAL): the drift beat that fires every few seconds while a
        # paragraph plays does not rewrite the whole document, but anything
        # that actually changes play state does, and the player marks those
        # with `settled`. What must never happen is losing someone's place, so
        # assert both halves of that on disk rather than through the API,
        # which answers from memory either way.
        post("/api/position", {"doc": doc_id, "block": 2, "word": 4,
                               "within": 1.25, "settled": True})
        with open(saved_path, encoding="utf-8") as f:
            on_disk = json.load(f)
        check("a settled position heartbeat is on disk immediately",
              on_disk.get("position", {}).get("block") == 2,
              str(on_disk.get("position")))
        post("/api/position", {"doc": doc_id, "block": 3, "word": 0, "within": 0.0,
                               "settled": True})
        post("/api/position", {"doc": doc_id, "block": 3, "word": 9, "within": 2.5})
        with open(saved_path, encoding="utf-8") as f:
            on_disk = json.load(f)
        check("...and a drift beat inside that same block costs no rewrite",
              on_disk.get("position", {}).get("block") == 3
              and on_disk.get("position", {}).get("within") == 0.0,
              str(on_disk.get("position")))
        st, body, _ = get(f"/api/doc?doc={doc_id}")
        check("...though the server still knows where playback really is",
              json.loads(body)["position"]["within"] == 2.5,
              str(json.loads(body)["position"]))

        print("surviving a restart (the actual point of the library)")
        srv.terminate()
        srv.wait(timeout=5)
        srv = start_server(env)
        st, body, _ = get("/api/docs")
        after_restart = json.loads(body)["docs"]
        reloaded = next((d for d in after_restart if d["id"] == doc_id), None)
        check("the document is still there after a restart", reloaded is not None,
              str(after_restart)[:120])
        if reloaded:
            check("...with the renamed title, not the original one",
                  reloaded["title"] == "Renamed by smoke.py", reloaded["title"])
            check("...and at the place playback had reached, not back at the top",
                  reloaded["position"]["block"] == 3, str(reloaded["position"]))
        st, body, _ = get(f"/api/block?doc={doc_id}&i=0")
        check("...and it resynthesizes on demand after the restart",
              st == 200, f"got {st}: {body[:100]}")

        print("alignment (the part that breaks quietly)")
        sys.path.insert(0, ".")
        from engines import (align, align_native, make_spoken_plan, spoken_form,
                             Word, words_of)
        s = words_of("The Miwok leader gave a well-made basket.")
        heard = [Word(w, i * 0.5, i * 0.5 + 0.4) for i, w in enumerate(
            "the me walk leader gave a well made basket".split())]
        out = align(s, heard)
        check("misheard and split words still map back", len(out) == len(s),
              f"{len(out)} vs {len(s)}")
        check("aligned timings stay monotonic",
              all(out[i].s <= out[i + 1].s for i in range(len(out) - 1)))
        repeated = align_native(
            make_spoken_plan("go go go"),
            [Word("go", 0.0, 0.2), Word("go", 0.6, 0.8)], [0, 2])
        check("native positional marks never swap repeated word occurrences",
              len(repeated) == 3 and repeated[0].s == 0.0
              and repeated[1].s == 0.2 and repeated[2].s == 0.6,
              str(repeated))

        print("two-column OCR splitting and reading order (breaks quietly, real-book bug)")
        import readaloud

        def W(text, x0, x1, y0=0, y1=12):
            return {"x0": x0, "x1": x1, "y0": y0, "y1": y1, "text": text}

        # A row where Tesseract's line grouping ran a left-column item and a
        # right-column item together, exactly as measured on a real
        # two-column numbered list (48pt gutter vs. ~3-6pt normal word gaps).
        row = [W("1.", 53, 61), W("List", 67, 80), W("of", 83, 93),
               W("4.", 141, 150), W("(a)", 156, 168)]
        split = readaloud._split_on_column_gap(row)
        check("a merged two-column OCR row splits at the real column gutter",
              len(split) == 2, str(split))
        if len(split) == 2:
            check("...left half keeps only the left column's words",
                  " ".join(w["text"] for w in split[0]) == "1. List of",
                  str(split))
            check("...right half keeps only the right column's words",
                  " ".join(w["text"] for w in split[1]) == "4. (a)", str(split))
        normal_row = [W("List", 67, 80), W("six", 83, 93), W("different", 96, 132)]
        check("ordinary word spacing on a single-column row is never split",
              len(readaloud._split_on_column_gap(normal_row)) == 1, str(normal_row))

        def L(text, x0, y0, x1, y1=None):
            return {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1 or y0 + 10}

        two_col_page = (
            [L("Intro sentence spanning the full page width here today", 10, 0, 300)] +
            [L(f"left {i}", 10, 20 + i * 12, 140) for i in range(3)] +
            [L(f"right {i}", 160, 20 + i * 12, 300) for i in range(2)]
        )
        ordered = readaloud.order_page_lines(two_col_page, 310)
        texts = [l["text"] for l in ordered]
        check("a short two-column section reads left-then-right, not interleaved by height",
              texts == ["Intro sentence spanning the full page width here today",
                        "left 0", "left 1", "left 2", "right 0", "right 1"],
              str(texts))

        print("OCR junk filtering (breaks quietly, real-book bug)")
        j = readaloud.is_ocr_junk
        check("a chapter-opening drop cap survives on its own",
              j("A") is False)
        check("a trailing-parenthetical fragment survives despite punctuation",
              j("(bottom).") is False)
        check("short stub words don't sink a line that also has real words",
              j("as it applies to materials.") is False)
        check("a lone word with trailing punctuation survives",
              j("capacity.") is False)
        check("a comma-separated list line survives",
              j("ent categories: mechanical, electrical, thermal, magnetic, "
                "optical, and deteriorative. For") is False)
        check("a verso running head with no long stub-free word still survives "
              "is_ocr_junk (furniture stripping handles it downstream)",
              j("4 • Chapter 1 / Introduction") is False)
        check("real confetti with no anchor word is still dropped",
              j("xk qz vb") is True)
        check("pure digits/symbols are still dropped",
              j("12 . , 34") is True)

        def GL(text, x0, y0, x1, y1, size=10.0, font="ocr", ypos=0.5):
            return {"text": text, "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                    "size": size, "font": font, "ypos": ypos}

        import pymupdf
        rect = pymupdf.Rect(100, 100, 200, 150)
        inside = {"x0": 110, "y0": 110, "x1": 190, "y1": 140}
        outside = {"x0": 300, "y0": 300, "x1": 350, "y1": 320}
        check("text mostly inside a figure's image rect is caught",
              readaloud._inside_figure(inside, [rect]) is True)
        check("text outside a figure's image rect is left alone",
              readaloud._inside_figure(outside, [rect]) is False)

        check("a verso running head (page + Chapter N / Title) is recognized on sight",
              readaloud.VERSO_HEAD.match("4 + Chapter1 / Introduction") is not None)
        check("...even with the bullet OCR'd as a real bullet character",
              readaloud.VERSO_HEAD.match("12 • Chapter 3 / Structures of Metals") is not None)
        check("...but an ordinary sentence starting with a number is left alone",
              readaloud.VERSO_HEAD.match("4 people were surveyed about Chapter 1 / results") is None)

        margin_lines = (
            [GL("4 + Chapter1 / Introduction", 40, 20, 220, 30, ypos=0.05)] +
            [GL(f"body line {i} of the actual paragraph text here", 40,
                60 + i * 12, 260, 70 + i * 12, ypos=0.3) for i in range(5)]
        )
        dropped = readaloud.drop_floating_labels(margin_lines, page_height=600)
        check("a running head at the top margin is not treated as an isolated "
              "floating label just because nothing sits above it on the page",
              any(l["text"].startswith("4 + Chapter1") for l in dropped),
              str([l["text"] for l in dropped]))

        print("paragraph reassembly: bullets and figure captions")

        bullet_line = [GL("° Explain the difference between", 50, 50, 300, 62)]
        paras = readaloud.group_paragraphs(bullet_line, set(), True)
        check("an OCR-misread bullet glyph is stripped from a new paragraph's start",
              paras == [["Explain the difference between"]], str(paras))

        # A footnote marker OCRs the same way a bullet does, but glues
        # directly to the word with no space - a real example from the test
        # book, where the stray "°" survived to verbalize() and turned into
        # "degrees This paradigm..." out loud.
        footnote_line = [GL("°This paradigm has recently been updated", 50, 50, 300, 62)]
        text = readaloud.verbalize(readaloud.join_lines(
            readaloud.group_paragraphs(footnote_line, set(), True)[0]))
        check("a footnote marker glued to the next word doesn't survive to "
              "be read as \"degrees\"",
              text == "This paradigm has recently been updated", text)

        # The actual 4-line Figure 1.1 caption from the real test book,
        # followed by the real body paragraph that follows it on the page.
        # A capital-letter start ends caption mode on purpose - that's what
        # correctly draws the line here, not a bug to fix. See the long
        # comment on this exact case in group_paragraphs.
        caption_then_body = [
            GL("Figure 1.1 Three thin disk specimens of aluminum oxide that",
               50, 100, 300, 112),
            GL("have been placed over a printed page in order to demonstrate",
               50, 114, 300, 126),
            GL("their differences in light-transmittance characteristics.",
               50, 128, 300, 140),
            GL("The disk on the left is transparent, the center one is",
               50, 142, 300, 154),
            GL("translucent, and the one on the right is opaque.",
               50, 156, 300, 168),
        ]
        tags = {}
        result_paras = readaloud.group_paragraphs(
            caption_then_body, set(), True,
            tag=lambda l, c: tags.setdefault(l["text"], c))
        check("the caption's own continuation lines are all tagged caption",
              tags.get("have been placed over a printed page in order to demonstrate")
              == "caption"
              and tags.get("their differences in light-transmittance characteristics.")
              == "caption", str(tags))
        check("the real body paragraph right after it is kept and read, not "
              "swallowed into the caption",
              result_paras == [["The disk on the left is transparent, the center one is",
                                 "translucent, and the one on the right is opaque."]],
              str(result_paras))

        # A citation wrapping mid-title ("Brittle Behavior of Engineering" /
        # "Structures, National Academy...") - real example from the test
        # book. The continuation starts uppercase, but the line before it
        # has no terminal punctuation, so it must still be a continuation
        # regardless of case.
        citation = [
            GL("Figure 1.3 The Liberty ship S.S. Schenectady, which, in 1943, failed",
               50, 100, 300, 112),
            GL("before leaving the shipyard.", 50, 114, 300, 126),
            GL("(Reprinted with permission of Earl R. Parker, Brittle Behavior of Engineering",
               50, 128, 300, 140),
            GL("Structures, National Academy of Sciences, National Research Council, John",
               50, 142, 300, 154),
            GL("Wiley & Sons, New York, 1957.)", 50, 156, 300, 168),
        ]
        tags = {}
        readaloud.group_paragraphs(citation, set(), True,
                                    tag=lambda l, c: tags.setdefault(l["text"], c))
        check("a citation wrapping mid-title stays caption across an "
              "uppercase continuation, because the line before it didn't "
              "end a sentence",
              tags.get("Structures, National Academy of Sciences, National "
                       "Research Council, John") == "caption", str(tags))

        # A running head OCR'd as two separate line objects on the same
        # row - a real example: "6 + Chapter1 /" and "Introduction" landed
        # as independent lines. Neither alone looks like a running head.
        split_head = [
            {"text": "6 + Chapter1 /", "x0": 400, "y0": 20, "x1": 460, "y1": 30},
            {"text": "Introduction", "x0": 465, "y0": 20, "x1": 520, "y1": 30},
        ]
        merged = readaloud._merge_edge_fragments(split_head, page_height=700)
        check("same-row running-head fragments in the edge band are "
              "recombined into one line",
              len(merged) == 1 and merged[0]["text"]
              == "6 + Chapter1 / Introduction", str(merged))
        check("...and the recombined line is now recognized as furniture",
              readaloud.VERSO_HEAD.match(merged[0]["text"]) is not None)

        # A short garbage line (photo-bleed noise that slips past
        # is_ocr_junk because it contains one coincidental 4-letter
        # "word") must not corrupt the hanging-indent baseline for
        # everything that follows - a real example turned "In addition to
        # structure..." into "addition to structure...", silently
        # dropping "In", because a 2-word noise fragment's x0 became the
        # trusted baseline.
        noisy_then_real = [
            GL("Deteriorative characteristics relate to the chemical reactivity of materials",
               50, 100, 400, 112),
            GL("anid ax", 250, 200, 280, 212),          # new para: big gap, odd x0
            GL("ok", 340, 214, 360, 226),               # its "continuation" - poisons baseline
            GL("In addition to structure and properties, two other important",
               50, 300, 400, 312),                      # new para: big gap back to normal x0
            GL("components are involved in the science and engineering of materials.",
               50, 314, 400, 326),
        ]
        paras = readaloud.group_paragraphs(noisy_then_real, set(), True)
        check("a short noise line doesn't poison the baseline used to spot "
              "hanging bullet markers - 'In' survives",
              any(p and p[0].startswith("In addition") for p in paras),
              str(paras))

        # verbalize()'s trailing-marker strip: a footnote mark glued to the
        # very end of a sentence ("...to each failure:°") is a different
        # position than BULLET_LEAD (start of paragraph) and would
        # otherwise survive into the global "°" -> " degrees " rule.
        check("a footnote marker glued to the end of a sentence doesn't "
              "get read as \"degrees\"",
              readaloud.verbalize("contributed to each failure:°")
              == "contributed to each failure:")
        check("...but a genuine trailing degree measurement is untouched",
              readaloud.verbalize("heated to 100°") == "heated to 100 degrees")

        # Footnote suppression: a real footnote sits below a much bigger
        # gap than any paragraph break, low on the page - the test book's
        # own footnotes measured at ~2.1-2.5x gap_break, ~0.79-0.87 down
        # the page. Mirrors caption handling: dropped entirely, not read.
        body_then_footnote = [
            GL("This is the last real sentence of the chapter section here",
               50, 500, 400, 512, ypos=0.65),
            GL("continuing for one more line to end the paragraph cleanly.",
               50, 514, 400, 526, ypos=0.66),
            GL("This is a footnote explaining something in more detail here",
               50, 560, 400, 572, ypos=0.75),
            GL("and it continues here with more citation detail text.",
               50, 574, 400, 586, ypos=0.76),
        ]
        tags = {}
        fn_paras = readaloud.group_paragraphs(
            body_then_footnote, set(), True,
            tag=lambda l, c: tags.setdefault(l["text"], c))
        check("a footnote below a big gap, low on the page, is tagged footnote",
              tags.get("This is a footnote explaining something in more detail here")
              == "footnote"
              and tags.get("and it continues here with more citation detail text.")
              == "footnote",
              str(tags))
        check("...and it's dropped from the read-aloud paragraphs, not "
              "the real body text before it",
              fn_paras == [["This is the last real sentence of the chapter section here",
                             "continuing for one more line to end the paragraph cleanly."]],
              str(fn_paras))

        # A caption below a photo has the same big-gap/low-page shape (the
        # photo itself is the gap) - must not be misclassified as a
        # footnote just because of that.
        photo_caption = [
            GL("This is the last real sentence before a photo appears here",
               50, 100, 400, 112, ypos=0.2),
            GL("Figure 1.9 a photo caption sitting far below on the page",
               50, 560, 400, 572, ypos=0.75),
            GL("continuing the caption text for one more line here now.",
               50, 574, 400, 586, ypos=0.76),
        ]
        tags = {}
        readaloud.group_paragraphs(photo_caption, set(), True,
                                    tag=lambda l, c: tags.setdefault(l["text"], c))
        check("a caption below a photo (same big-gap/low-page shape as a "
              "footnote) is still recognized as a caption, not a footnote",
              tags.get("Figure 1.9 a photo caption sitting far below on the page")
              == "caption", str(tags))

        print("seeking (a jump re-aims the warmer instead of racing it)")
        seek_text = "\n\n".join(f"Paragraph number {i} of a seek test document."
                                for i in range(12))
        st, sdoc = post("/api/open", {"text": seek_text, "title": "Seek test"})
        seek_id = sdoc["id"]
        last = len(sdoc["blocks"]) - 1
        check("the seek test document opens with blocks to jump between",
              st == 200 and len(sdoc["blocks"]) >= 6, str(len(sdoc["blocks"])))
        st, body = post("/api/position",
                        {"doc": seek_id, "block": last, "word": 0, "within": 0.0})
        check("a jump to the far end of a document is accepted", st == 200 and body.get("ok"),
              str(body)[:80])
        st, body, _ = get(f"/api/block?doc={seek_id}&i={last}")
        check("...and the block jumped to still synthesizes on demand",
              st == 200 and json.loads(body)["words"], body[:100].decode("utf-8", "replace"))
        st, body, _ = get("/api/docs")
        seeked = next(d for d in json.loads(body)["docs"] if d["id"] == seek_id)
        check("...and the new position is what got saved",
              seeked["position"]["block"] == last, str(seeked["position"]))

        print("voices")
        st, body, _ = get("/api/voices")
        vjson = json.loads(body)
        check("voice catalogue is non-empty for the running engine",
              st == 200 and len(vjson["voices"]) > 0, str(vjson)[:120])
        check("stub engine exposes exactly its one voice",
              vjson["engine"] == "stub" and [v["id"] for v in vjson["voices"]] == ["stub"],
              str(vjson))

        print("voice switching mid-read")
        st, doc3 = post("/api/open", {"text": SAMPLE + " " + SAMPLE + " " + SAMPLE})
        doc3_id = doc3["id"]
        # force every block cached, then advance "position" past block 0 and switch voice
        for i in range(len(doc3["blocks"])):
            get(f"/api/block?doc={doc3_id}&i={i}")
        st, body = post("/api/position", {"doc": doc3_id, "block": 1, "word": 0, "within": 0.5})
        check("position heartbeat accepted", st == 200 and body.get("ok"), str(body)[:80])
        st, body, _ = get("/api/docs")
        before = next(d for d in json.loads(body)["docs"] if d["id"] == doc3_id)
        check("ready count reflects full cache before any voice switch",
              before["ready"] == len(doc3["blocks"]), str(before))
        st, body = post("/api/voice", {"doc": doc3_id, "voice": "stub"})
        check("switching to a valid voice succeeds", st == 200 and body.get("ok"), str(body)[:80])
        st, body, _ = get("/api/docs")
        after = next(d for d in json.loads(body)["docs"] if d["id"] == doc3_id)
        check("voice switch invalidates cache for blocks after the current one, "
              "keeps the current/past ones", after["ready"] <= before["ready"] and after["ready"] >= 1,
              f"before={before['ready']} after={after['ready']}")
        st, body = post("/api/voice", {"doc": doc3_id, "voice": "not-a-real-voice"})
        check("switching to an unknown voice is rejected", st == 400, str(body)[:80])

        print("default voice")
        st, body = post("/api/voice", {"voice": "stub", "default": True})
        check("setting a default voice succeeds", st == 200, str(body)[:80])
        st, body, _ = get("/api/voices")
        check("the new default is reflected back", json.loads(body)["default"] == "stub", body[:120])

        print("voices & voice gallery")
        # Pre-warmed sample clips. The gallery is only pleasant if a card
        # plays the instant it is clicked, and the only way to tell warming
        # from a fast engine is to ask the server which it did.
        ready = []
        for _ in range(40):
            st, body, _ = get("/api/voices")
            ready = json.loads(body).get("previews_ready") or []
            if "stub" in ready:
                break
            time.sleep(0.25)
        check("voice previews are pre-warmed in the background, not on demand",
              "stub" in ready, str(ready))

        st, pbody, pctype, pheaders = get_full("/api/voice/preview?voice=stub")
        check("a voice preview returns audio bytes", st == 200 and len(pbody) > 100,
              f"status {st} len {len(pbody)}")
        check("...with an audio content-type", pctype.startswith("audio/"), pctype)
        check("...served from the preview cache, not synthesized per request",
              pheaders.get("X-Preview-Cache") == "hit", str(pheaders.get("X-Preview-Cache")))

        st2, pbody2, _, pheaders2 = get_full("/api/voice/preview?voice=stub")
        check("a repeated preview is byte-identical and still a cache hit",
              st2 == 200 and pbody2 == pbody and pheaders2.get("X-Preview-Cache") == "hit",
              f"status {st2} same={pbody2 == pbody} {pheaders2.get('X-Preview-Cache')}")
        st, _, _ = get("/api/voice/preview?voice=not-a-real-voice")
        check("an unknown voice previews as 404, not a 500 from the engine", st == 404, f"status {st}")

        # Mid-read switching. What happens to the block that is sounding right
        # now is the whole question: re-making it would stall the boundary the
        # switch is trying to make seamless (see _voice in talkify.py).
        st, doc4 = post("/api/open", {"text": " ".join([SAMPLE] * 4)})
        doc4_id = doc4["id"]
        for i in range(len(doc4["blocks"])):
            get(f"/api/block?doc={doc4_id}&i={i}")
        post("/api/position", {"doc": doc4_id, "block": 1, "word": 0, "within": 0.5})

        st, body = post("/api/voice", {"doc": doc4_id, "voice": "stub", "playing": True})
        check("switching mid-play keeps the block being spoken and starts the "
              "new voice at the next one",
              st == 200 and body.get("kept_through") == 1 and body.get("warm_from") == 2,
              str(body)[:120])

        st, body = post("/api/voice", {"doc": doc4_id, "voice": "stub", "playing": False})
        check("switching while paused re-makes the current block instead, so "
              "Play speaks the new voice at once",
              st == 200 and body.get("kept_through") == 0 and body.get("warm_from") == 1,
              str(body)[:120])

        st, body = post("/api/voice", {"doc": doc4_id, "voice": "stub"})
        check("a client that says nothing about playback gets the safe "
              "(paused) answer",
              st == 200 and body.get("warm_from") == 1, str(body)[:120])

        # Which block to keep is the reader's fact, not the server's: the
        # position heartbeat is throttled and async, so a switch made just
        # after a boundary can beat it here. A block in the payload wins.
        st, doc5 = post("/api/open", {"text": "\n\n".join([SAMPLE] * 5)})
        doc5_id = doc5["id"]
        check("the explicit-block fixture has blocks to tell apart",
              len(doc5["blocks"]) >= 4, str(len(doc5["blocks"])))
        post("/api/position", {"doc": doc5_id, "block": 0, "word": 0, "within": 0.0})
        st, body = post("/api/voice", {"doc": doc5_id, "voice": "stub",
                                       "playing": True, "block": 3})
        check("an explicit block in the payload beats a stale position heartbeat",
              st == 200 and body.get("kept_through") == 3 and body.get("warm_from") == 4,
              str(body)[:120])
        st, body = post("/api/voice", {"doc": doc5_id, "voice": "stub",
                                       "playing": False, "block": 2})
        check("...and a paused switch on that block re-makes it, not the one "
              "the server last heard about",
              st == 200 and body.get("kept_through") == 1 and body.get("warm_from") == 2,
              str(body)[:120])
        st, body = post("/api/voice", {"doc": doc5_id, "voice": "stub",
                                       "playing": True, "block": 999})
        check("a block past the end is clamped, not a 500",
              st == 200 and body.get("kept_through") == len(doc5["blocks"]) - 1,
              str(body)[:120])
        st, body = post("/api/voice", {"doc": doc5_id, "voice": "stub",
                                       "playing": True, "block": -5})
        check("a negative block is clamped to the first one",
              st == 200 and body.get("kept_through") == 0, str(body)[:120])
        st, body = post("/api/voice", {"doc": doc5_id, "voice": "stub", "block": "two"})
        check("a block that isn't a number is rejected, not guessed at",
              st == 400, f"status {st} {str(body)[:80]}")
        st, body = post("/api/voice", {"doc": doc5_id, "voice": "stub", "playing": True})
        check("a client that sends no block still falls back to the heartbeat",
              st == 200 and "kept_through" in body, str(body)[:120])

        # The warmer is asked to catch up from warm_from; heard blocks are
        # never re-synthesized, so a full document ends up cached again.
        for _ in range(40):
            st, body, _ = get("/api/docs")
            after = next(d for d in json.loads(body)["docs"] if d["id"] == doc4_id)
            if after["ready"] == len(doc4["blocks"]):
                break
            time.sleep(0.25)
        check("the warmer refills every block after a voice switch",
              after["ready"] == len(doc4["blocks"]),
              f"{after['ready']} of {len(doc4['blocks'])}")
        check("...and the document reports the new active voice",
              after["voice"] == "stub", str(after)[:120])

        # Warm order, tested where it lives rather than through timing: the
        # plan a mid-play switch builds must reach cur + 1 before anything
        # else, and must not put cur first.
        import coordinator as coordinator_module
        plan = coordinator_module._WarmPlan("wo", 10, 4, "v", 0)
        check("a mid-play switch warms forward from the next block first",
              plan.order[:3] == [4, 5, 6], str(plan.order))
        check("...and never re-synthesizes the current block ahead of it",
              plan.order.index(3) > plan.order.index(5), str(plan.order))

        # Set-as-default from a gallery card has to outlive the process, so it
        # is on disk, not just in memory.
        st, body = post("/api/voice", {"voice": "stub", "default": True})
        check("setting a default from the gallery succeeds", st == 200, str(body)[:80])
        with open(tmp_config, encoding="utf-8") as fh:
            saved_settings = json.load(fh)
        check("...and the default voice is written to config.json, not just held in memory",
              saved_settings.get("default_voice") == "stub", str(saved_settings)[:120])
        st, body, _ = get("/api/voices")
        check("...and /api/voices reports it back to the gallery",
              json.loads(body)["default"] == "stub", body[:120])

        print("recent / progress")
        st, body, _ = get("/api/recent?limit=3")
        recent = json.loads(body)["docs"]
        check("recent returns at most the requested count", 0 < len(recent) <= 3, str(recent)[:120])
        check("recent entries carry runtime progress fields",
              all("percent" in d and "elapsed_seconds" in d for d in recent), str(recent)[:160])
        check("percent is a sane 0-100", all(0 <= d["percent"] <= 100 for d in recent), str(recent)[:160])

        print("split-view & page rendering")
        st, body, _ = get(f"/api/doc?doc={doc_id}")
        doc_meta = json.loads(body)
        check("doc metadata reports has_pages flag", "has_pages" in doc_meta, str(doc_meta)[:80])
        check("doc metadata reports page_count integer", isinstance(doc_meta.get("page_count"), int), str(doc_meta)[:80])
        st, body, _ = get("/api/page")
        check("page request without doc param 400s cleanly", st == 400, str(body)[:80])
        st, body, _ = get(f"/api/page?doc={doc_id}&page=0")
        check("page request on a non-pdf document 404s cleanly", st == 404, str(body)[:80])
        if pdf_doc_id:
            st, body, _ = get(f"/api/page?doc={pdf_doc_id}&page=abc")
            check("page request with non-integer page 400s cleanly", st == 400, str(body)[:80])
            st, body, _ = get(f"/api/page?doc={pdf_doc_id}&page=0&dpi=bad")
            check("page request with non-integer dpi 400s cleanly", st == 400, str(body)[:80])
            # Page 1, not page 0: this document was opened as page 2 of the
            # file, so page 0 is a page it does not contain and the route is
            # right to refuse it (asserted below).
            st, body, ctype = get(f"/api/page?doc={pdf_doc_id}&page=1&dpi=150")
            check("page request on a valid pdf returns 200", st == 200, str(st))
            check("page returns image/png content type", ctype == "image/png", str(ctype))
            check("page image contains rendered bytes", len(body) > 100, f"{len(body)} bytes")
            st, body, _ = get(f"/api/page?doc={pdf_doc_id}&page=9999")
            check("out-of-bounds page request 404s cleanly", st == 404, str(body)[:80])

        print("ai provider resolution (pure logic - no request leaves this machine)")
        sys.path.insert(0, ".")
        import ai as ai_module

        tmp_ai_dir = tempfile.mkdtemp(prefix="talkify_smoke_ai_")
        AI_ENV = ("TALKIFY_CONFIG_PATH", "TALKIFY_AI_PROVIDER", "OLLAMA_HOST",
                  "GEMINI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY")

        def with_ai_config(cfg, env_over=None):
            """provider_info() under a given config.json, with the ambient
            environment neutralised - otherwise a machine that happens to
            export OPENAI_API_KEY would change what these assert. The Ollama
            host is pointed at a dead port for the same reason: whether these
            pass must not depend on whether Ollama is running.

            None of this reaches the network. Cloud availability is decided by
            the presence of a key, deliberately - probing a metered endpoint
            just to grey out a button would cost the user money."""
            path = os.path.join(tmp_ai_dir, "ai_config.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(cfg, f)
            saved = {k: os.environ.pop(k, None) for k in AI_ENV}
            os.environ["TALKIFY_CONFIG_PATH"] = path
            os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"
            os.environ.update(env_over or {})
            try:
                return ai_module.provider_info()
            finally:
                for k in AI_ENV:
                    if saved[k] is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = saved[k]

        info = with_ai_config({})
        check("no key and no Ollama: the feature reports itself absent, not broken",
              info == {"available": False, "provider": None, "model": None,
                       "local": False, "label": None}, str(info))

        info = with_ai_config({"gemini_api_key": "test-key"})
        check("a Gemini key in config.json selects Gemini",
              info["available"] and info["provider"] == "gemini"
              and info["model"] == "gemini-2.5-flash", str(info))
        check("...and it is reported as a cloud provider, so the UI can say so",
              info["local"] is False and info["label"] == "Gemini 2.5 Flash", str(info))

        info = with_ai_config({}, {"ANTHROPIC_API_KEY": "test-key"})
        check("a key in the environment works the same as one in config.json",
              info["provider"] == "anthropic"
              and info["model"] == "claude-3-5-haiku-20241022", str(info))

        info = with_ai_config({"openai_api_key": "test-key"})
        check("an OpenAI key selects OpenAI",
              info["provider"] == "openai" and info["model"] == "gpt-4o-mini", str(info))

        info = with_ai_config({"gemini_api_key": "k", "openai_api_key": "k"})
        check("auto-detection follows the documented preference order",
              info["provider"] == "gemini", str(info))

        info = with_ai_config({"ai_provider": "openai",
                               "gemini_api_key": "k", "openai_api_key": "k"})
        check("an explicit ai_provider outranks auto-detection",
              info["provider"] == "openai", str(info))

        # The one that matters for privacy: a reader who picked local did so
        # on purpose, and a missing key must never quietly promote a cloud
        # provider in its place.
        info = with_ai_config({"ai_provider": "anthropic", "gemini_api_key": "k"})
        check("an explicit provider with no key is unavailable, never silently "
              "swapped for a different one", info["available"] is False, str(info))

        info = with_ai_config({"ai_provider": "nonesuch", "gemini_api_key": "k"})
        check("an unknown provider name is unavailable rather than a crash",
              info["available"] is False, str(info))

        info = with_ai_config({"openai_api_key": "k", "openai_model": "gpt-4o"})
        check("a model override is used and named in the badge",
              info["model"] == "gpt-4o" and info["label"] == "OpenAI (gpt-4o)", str(info))

        check("an error message never carries the key that produced it",
              "sekret" not in ai_module._redact(
                  "HTTP 403: bad key=sekret", "sekret"), "key leaked into an error")

        # The silent-truncation tripwire: the only way a dropped
        # system prompt ever becomes visible.
        check("a prompt the model barely read is flagged as degraded",
              ai_module._truncation_warning(20, 6000) is not None)
        check("a prompt the model read in full is not flagged",
              ai_module._truncation_warning(1400, 6000) is None)
        check("a provider that reports no token count is not flagged either",
              ai_module._truncation_warning(None, 6000) is None)

        print("ai explain endpoint (skips the live-answer checks if no backend is reachable)")
        st, body, _ = get("/api/ai/status")
        status = json.loads(body)
        check("ai status reports a boolean availability", isinstance(status.get("available"), bool),
              str(status))
        check("ai status names the provider and whether it is local",
              set(status) >= {"available", "provider", "model", "local", "label"}
              and isinstance(status["local"], bool), str(status))
        check("nothing is named when nothing is available",
              status["available"] or (status["provider"] is None and status["model"] is None),
              str(status))
        st, body = post("/api/explain", {"doc": doc_id, "block": 9999})
        check("explain on an out-of-range block 400s cleanly", st == 400, str(body)[:80])
        st, body = post("/api/explain", {"doc": "no-such-doc", "block": 0})
        check("explain on an unknown document 400s cleanly", st == 400, str(body)[:80])

        # A backend over a network is the most likely thing here to fail, so
        # every failure has to come back as JSON the panel can print.
        raw = urllib.request.Request(BASE + "/api/explain", data=b"{not json",
                                      headers={"Content-Type": "application/json"},
                                      method="POST")
        try:
            with urllib.request.urlopen(raw, timeout=30) as r:
                st, body = r.status, r.read()
        except urllib.error.HTTPError as e:
            st, body = e.code, e.read()
        check("a malformed explain body is a clean JSON 400, not an uncaught 500",
              st == 400 and json.loads(body).get("error"), f"status {st}: {body[:80]}")

        if ai_module.available():
            st, body = post("/api/explain", {"doc": doc_id, "block": 0})
            check("explain answers when a backend is available",
                  st == 200 and body.get("answer"), str(body)[:160])
            check("...and says which provider answered",
                  body.get("provider"), str(body)[:160])

            # A selection is meant to ground the answer in exactly what was
            # selected, not just the paragraph it started in (a selection
            # can span several blocks) - this only exercises that code path
            # if /api/explain is actually using the selection text as
            # context rather than silently falling back to doc["blocks"][i].
            long_selection = "This sentence about kelp forests repeats. " * 150
            st, body = post("/api/explain",
                             {"doc": doc_id, "block": 0, "selection": long_selection})
            check("an overlong selection is truncated with an explicit warning, "
                  "not silently dropped", st == 200 and body.get("warning") and "too long" in body["warning"],
                  str(body)[:160])
        else:
            st, body = post("/api/explain", {"doc": doc_id, "block": 0})
            check("with no backend, explain refuses in JSON rather than hanging or 500ing",
                  st == 503 and body.get("error"), f"status {st}: {str(body)[:120]}")
            print("  skip  (no reachable backend in this environment)")

        print("practice problem detection (never invents, only locates)")
        import practice as practice_module
        practice_text = ("Some ordinary paragraph about drainage basins.\n\n"
                          "Exercises\n\n"
                          "1. Compute the runoff coefficient for the site. Ans: 0.45\n\n"
                          "2. Explain why the coefficient changes with land use.\n\n"
                          "A closing paragraph with no numbers at all.")
        st, pdoc = post("/api/open", {"text": practice_text, "title": "Practice test"})
        check("practice sets come back on open", st == 200 and pdoc.get("practice"), str(pdoc)[:160])
        if pdoc.get("practice"):
            pset = pdoc["practice"][0]
            check("heading block is the 'Exercises' paragraph",
                  "Exercises" in pdoc["blocks"][pset["heading_block"]], str(pset))
            check("both numbered items are found", len(pset["items"]) == 2, str(pset))
            with_answer, thought = pset["items"][0], pset["items"][1]
            check("an item that prints its own answer is marked as having one",
                  with_answer["has_answer"] and with_answer["answer"] == "0.45", str(with_answer))
            check("an open-ended item is marked as a thought question, with no answer",
                  thought["has_answer"] is False and thought["answer"] is None, str(thought))
        st, plain = post("/api/open", {"text": SAMPLE})
        check("a document with no problem set reports an empty list, not an error",
              st == 200 and plain.get("practice") == [], str(plain)[:160])

        # The answers a book actually prints, and the near-misses that would
        # show the reader a wrong answer key if they were matched. A wrong
        # answer in a study tool is worse than no answer, so
        # every ambiguous case must come back as a thought question.
        check("'Ans:' is read as an answer",
              practice_module.answer_in("3. Find the reaction at A. Ans: 4.2 kN") == "4.2 kN")
        check("a parenthesised '(Ans. ...)' is read as an answer",
              practice_module.answer_in("3. Find the reaction at A. (Ans. 4.2 kN)") == "4.2 kN")
        check("a trailing bracketed result is read as an answer",
              practice_module.answer_in("3. Find the reaction at A. [4.2 kN]") == "4.2 kN")
        check("'Solution:' is read as an answer",
              practice_module.answer_in("7. What is the coefficient? Solution: 0.45.") == "0.45")
        check("multi-part '(a) ... (b) ...' is a question, not an answer key",
              practice_module.answer_in("4. Compute the moment. (a) at A (b) at B") is None)
        check("an item that opens with the word 'Answer' is still a question",
              practice_module.answer_in("1. Answer the following in one sentence.") is None)
        check("a bracketed cross-reference is not mistaken for an answer",
              practice_module.answer_in("5. Discuss the poem [see Chapter 3].") is None)
        check("an ordinary open question has no answer",
              practice_module.answer_in("2. Explain why the coefficient changes.") is None)
        check("an imperative sentence with 'Answer' mid-text is not mistaken for an answer",
              practice_module.answer_in("1. Do you agree? Answer in one sentence.") is None)
        check("a bracketed figure citation is not mistaken for an answer",
              practice_module.answer_in("6. Compare results with [Fig. 2 on page 14].") is None)


        print("mp3 export")
        single_export_path = None       # picked up again by the sidecar checks
        st, doc4 = post("/api/open", {"text": SAMPLE + "\n\n" + SAMPLE + "\n\n" + SAMPLE})
        doc4_id = doc4["id"]
        st, body = post("/api/export", {"doc": doc4_id})
        check("export job starts", st == 200 and body.get("job"), str(body)[:80])
        job_id = body["job"]
        job = None
        for _ in range(120):
            st, jbody, _ = get(f"/api/export/status?job={job_id}")
            job = json.loads(jbody)
            if job["status"] != "running":
                break
            time.sleep(0.25)
        check("export job finishes", job["status"] == "done", str(job)[:160])
        if job["status"] == "done":
            single_export_path = job["path"]
            check("exported mp3 exists on disk and is non-trivial size",
                  os.path.exists(job["path"]) and os.path.getsize(job["path"]) > 500, str(job)[:160])
            with open(job["path"].rsplit(".", 1)[0] + ".json", encoding="utf-8") as f:
                exported_words = json.load(f)
            check("exported json has one timing per source word across all blocks",
                  len(exported_words) == sum(len(b.split()) for b in doc4["blocks"]),
                  f"{len(exported_words)} timings")
            check("exported timings are monotonic across the whole document",
                  all(exported_words[i]["s"] <= exported_words[i + 1]["s"]
                      for i in range(len(exported_words) - 1)))
            check("second and third block's words don't start at zero "
                  "(actually offset onto the combined timeline)",
                  exported_words[-1]["s"] > 1.0, str(exported_words[-1]))
            st, mp3_bytes, ctype = get(f"/api/export/download?job={job_id}")
            check("exported mp3 downloads", st == 200 and len(mp3_bytes) > 500, f"{len(mp3_bytes)} bytes")
            check("download content-type is audio/mpeg", ctype == "audio/mpeg", ctype)
            # The library's download link outlives the job that made it - a
            # restart, or a re-export that failed while the previous file is
            # still on disk, leaves a job id this process cannot serve. The
            # link carries the document id alongside it so the package index
            # can answer instead; sending only the job id 404'd with the MP3
            # sitting right there.
            st, fb_bytes, _ = get(f"/api/export/download?job=nonexistent&doc={doc4_id}")
            check("a download naming a job this process never issued still "
                  "serves the file, by document",
                  st == 200 and fb_bytes[:1] == b"\xff",
                  f"{st}, {len(fb_bytes)} bytes")
            check("the download really is MP3 - frame sync, not a truncated "
                  "part file renamed hopefully",
                  mp3_bytes[:1] == b"\xff" and (mp3_bytes[1] & 0xE0) == 0xE0,
                  mp3_bytes[:4].hex())
            check("a finished export leaves no .part files in the export folder",
                  not [f for f in os.listdir(tmp_exports) if f.endswith(".part")],
                  str(os.listdir(tmp_exports))[:160])
            # The library row turns into [Download MP3] [Re-export] [Delete] off
            # these two fields, and "Re-export" offers to replace a voice it has
            # to be able to name. state.activeExports is session memory, so
            # after a reload the summary is the only thing that knows either.
            st, dbody, _ = get("/api/docs")
            summary = next((d for d in json.loads(dbody)["docs"]
                            if d["id"] == doc4_id), None)
            check("the library reports a finished export for that document",
                  bool(summary) and summary.get("exported") is True,
                  str(summary and summary.get("exported")))
            check("...and names the voice it was exported in, so the row can "
                  "offer to replace it",
                  bool(summary) and summary.get("exported_voice") == (job.get("voice") or ""),
                  f"summary={summary and summary.get('exported_voice')!r} "
                  f"job={job.get('voice')!r}")
        st, body, _ = get("/api/export/status?job=nonexistent")
        check("status for an unknown export job 404s cleanly", st == 404, str(body)[:80])

        print("export voice selection (which voice the file is read in)")
        BLANK = "\n\n"
        # A document's blocks are cached under whichever voice made them, so
        # trying voices mid-read splits a book across several. /api/doc reports
        # that split; the export dialog uses it to say which voice is already
        # mostly made, and sends the chosen one back on /api/export.
        st, vdoc = post("/api/open", {"text": SAMPLE + BLANK + SAMPLE,
                                      "title": "Export voice test"})
        vdoc_id = vdoc["id"]
        get(f"/api/audio?doc={vdoc_id}&i=0")             # cache one block as-is
        st, body, _ = get(f"/api/doc?doc={vdoc_id}")
        vdoc_full = json.loads(body)
        doc_voice = vdoc_full.get("voice") or ""
        check("a document reports which voices it already has audio in",
              isinstance(vdoc_full.get("cached_voices"), dict)
              and vdoc_full["cached_voices"].get(doc_voice, 0) >= 1,
              str(vdoc_full.get("cached_voices")))
        check("...and the library summary carries the same counts, so the "
              "export dialog can offer them without reopening the document",
              any(d["id"] == vdoc_id and isinstance(d.get("cached_voices"), dict)
                  and d["cached_voices"].get(doc_voice, 0) >= 1
                  for d in json.loads(get("/api/docs")[1])["docs"]),
              str(json.loads(get("/api/docs")[1])["docs"][:1])[:200])

        # Export it in a voice that is *not* the one it has been read in. The
        # completed job names the chosen voice; its one-shot synthesis is not
        # retained, because export is deliberately a scan-resistant cache
        # consumer rather than an LRU-filling second warmer.
        alt_voice = "smoke_alt_voice"
        st, body = post("/api/export", {"doc": vdoc_id, "voice": alt_voice})
        check("export accepts an explicit voice", st == 200 and body.get("job"), str(body)[:120])
        vjob_id = body.get("job")
        vjob = None
        for _ in range(120):
            st, jbody, _ = get(f"/api/export/status?job={vjob_id}")
            vjob = json.loads(jbody)
            if vjob["status"] != "running":
                break
            time.sleep(0.25)
        check("an export in a chosen voice finishes",
              vjob and vjob["status"] == "done", str(vjob)[:200])
        check("...and the job says which voice it is being read in, so the "
              "progress line can name it",
              vjob and vjob.get("voice") == alt_voice, str(vjob)[:200])
        if vjob and vjob["status"] == "done":
            check("...producing a real mp3 like any other export",
                  os.path.exists(vjob["path"]) and os.path.getsize(vjob["path"]) > 500,
                  str(vjob)[:200])
            # The library summary, not /api/doc: asking for the document
            # re-aims the warmer at its own voice, which would race the very
            # counts being checked here.
            after = next(d for d in json.loads(get("/api/docs")[1])["docs"]
                         if d["id"] == vdoc_id)
            check("...without replacing the reader cache with one-shot export audio",
                  after["cached_voices"].get(alt_voice, 0) == 0,
                  str(after.get("cached_voices")))
            check("...and exporting never changes what the reader is set to",
                  after.get("voice") == vdoc_full.get("voice"),
                  f"{vdoc_full.get('voice')} -> {after.get('voice')}")

        st, body = post("/api/export", {"doc": vdoc_id, "voice": ""})
        check("an empty voice still means the document's own - what every "
              "client sent before the picker existed", st == 200 and body.get("job"),
              str(body)[:120])
        blank_job = body.get("job")
        for _ in range(120):
            st, jbody, _ = get(f"/api/export/status?job={blank_job}")
            job2 = json.loads(jbody)
            if job2["status"] != "running":
                break
            time.sleep(0.25)
        check("...and that export reports the document's voice",
              job2.get("voice") == vdoc_full.get("voice"), str(job2)[:160])

        print("chapter export (one file pair per chapter, each on its own timeline)")
        # Real chapters come from chapters.detect(), which needs a PDF and a
        # language model - neither of which this run has. The chapter list is
        # written into the library record instead and the server restarted, so
        # the document comes back exactly as a really-detected one would.
        chap_text = "\n\n".join(
            f"Paragraph number {i} of the chaptered smoke test document, long "
            "enough to be worth synthesizing on its own." for i in range(6))
        st, cdoc = post("/api/open", {"text": chap_text, "title": "Chaptered"})
        cdoc_id, cblocks = cdoc["id"], cdoc["blocks"]
        check("the chaptered fixture split into enough blocks to divide",
              len(cblocks) >= 4, f"{len(cblocks)} blocks")
        mid = len(cblocks) // 2
        rec_path = os.path.join(tmp_library, f"{cdoc_id}.json")
        with open(rec_path, encoding="utf-8") as f:
            rec = json.load(f)
        rec["chapters"] = [{"title": "The Opening", "block": 0},
                           {"title": "The Second Half", "block": mid}]
        with open(rec_path, "w", encoding="utf-8") as f:
            json.dump(rec, f)
        srv.terminate()
        srv.wait(timeout=5)
        srv = start_server(env)
        st, body, _ = get(f"/api/doc?doc={cdoc_id}")
        check("the chaptered document comes back with its chapters",
              st == 200 and len(json.loads(body).get("chapters", [])) == 2, body[:160])

        st, body = post("/api/export", {"doc": cdoc_id, "mode": "chapters"})
        check("chapter export job starts", st == 200 and body.get("job"), str(body)[:120])
        cjob_id = body.get("job")
        cjob = None
        for _ in range(240):
            st, jbody, _ = get(f"/api/export/status?job={cjob_id}")
            cjob = json.loads(jbody)
            if cjob["status"] != "running":
                break
            time.sleep(0.25)
        check("chapter export job finishes", cjob and cjob["status"] == "done", str(cjob)[:200])
        if cjob and cjob["status"] == "done":
            check("one file pair per detected chapter, not one file for the book",
                  len(cjob.get("files") or []) == 2, str(cjob.get("files"))[:200])
            check("the chapter export goes into its own package folder - the "
                  "unit that gets copied to a phone, not files loose in exports/",
                  os.path.dirname(cjob.get("dir") or "") == tmp_exports
                  and re.fullmatch(r"Chaptered--[0-9a-f]{8}",
                                   os.path.basename(cjob.get("dir") or "")),
                  str(cjob.get("dir")))
        if cjob and len(cjob.get("files") or []) == 2:
            files = cjob["files"]
            pkg_dir = cjob["dir"]
            check("chapter files lead with a zero-padded ordinal, which is what "
                  "a filename-only media player sorts by",
                  os.path.basename(files[0]["path"]) == "001_The Opening.mp3"
                  and os.path.basename(files[1]["path"]) == "002_The Second Half.mp3",
                  str([os.path.basename(f["path"]) for f in files]))
            for i, f in enumerate(files):
                check(f"chapter {i + 1} mp3 exists on disk and is non-trivial size",
                      os.path.exists(f["path"]) and os.path.getsize(f["path"]) > 500,
                      str(f)[:160])
                check(f"chapter {i + 1} json sits next to it",
                      os.path.exists(f["json_path"]), str(f)[:160])
            # In a package the sidecar is an object that remembers which source
            # block every word came from; the flat array is the single-file
            # form. Flattened back here so the timeline assertions below are
            # the same ones they always were.
            chap_docs, chap_words = [], []
            for f in files:
                with open(f["json_path"], encoding="utf-8") as fh:
                    doc_json = json.load(fh)
                chap_docs.append(doc_json)
                chap_words.append([w for b in doc_json["blocks"] for w in b["words"]])
            check("a chapter's timings name the source block each word came "
                  "from - a flat array cannot, and a phone resuming needs it",
                  all(d["schema"] == "talkify.chapter-timings/v1"
                      and all(isinstance(b["block_index"], int)
                              and b["word_start_index"] == 0 for b in d["blocks"])
                      for d in chap_docs), str(chap_docs[0])[:200])
            check("chapter 2's blocks carry their real document indices, not a "
                  "count restarted at zero for the file",
                  [b["block_index"] for b in chap_docs[1]["blocks"]]
                  == list(range(mid, len(cblocks))),
                  str([b["block_index"] for b in chap_docs[1]["blocks"]]))
            check("chapter 1's timings start at 0.0s",
                  bool(chap_words[0]) and chap_words[0][0]["s"] == 0.0, str(chap_words[0][:1]))
            check("chapter 2's timings ALSO start at 0.0s - each chapter is its "
                  "own audio file, so it needs its own timeline",
                  bool(chap_words[1]) and chap_words[1][0]["s"] == 0.0, str(chap_words[1][:1]))
            check("chapter 2 starts at the chapter's first word, not the book's",
                  chap_words[1][0]["w"].strip(".,").lower()
                  == cblocks[mid].split()[0].strip(".,").lower(),
                  f"{chap_words[1][0]['w']!r} vs {cblocks[mid].split()[0]!r}")
            check("every chapter's timings are monotonic",
                  all(all(w[i]["s"] <= w[i + 1]["s"] for i in range(len(w) - 1))
                      for w in chap_words))
            check("the two chapters together cover every word of the document, "
                  "none dropped at the split",
                  sum(len(w) for w in chap_words)
                  == sum(len(b.split()) for b in cblocks),
                  f"{sum(len(w) for w in chap_words)} timings")
            st, chap_bytes, ctype = get(f"/api/export/download?job={cjob_id}&file=1")
            check("an individual chapter downloads by index",
                  st == 200 and len(chap_bytes) > 500, f"{st}, {len(chap_bytes)} bytes")
            check("...as audio/mpeg", ctype == "audio/mpeg", ctype)
            check("...and it is the second chapter's file, not the first",
                  len(chap_bytes) == os.path.getsize(files[1]["path"]),
                  f"{len(chap_bytes)} vs {os.path.getsize(files[1]['path'])}")
            st, body, _ = get(f"/api/export/download?job={cjob_id}&file=99")
            check("asking for a chapter that isn't there 404s cleanly", st == 404, str(body)[:80])
        leftover_parts = [os.path.join(root, f)
                          for root, _dirs, names in os.walk(tmp_exports)
                          for f in names if f.endswith(".part")]
        check("a finished chapter export leaves no .part files behind either - "
              "inside the package folder as well as beside it",
              not leftover_parts, str(leftover_parts)[:200])

        print("commute package manifest (what a phone reads to find its way "
              "around the folder)")
        # See commute.py. The manifest is the only thing tying
        # per-chapter audio to one document-wide timeline, and every field
        # below is one a reader would silently get wrong if it drifted.
        manifest = None
        if cjob and cjob.get("dir"):
            manifest_path = os.path.join(cjob["dir"], "Chaptered.package.json")
            check("the manifest is named for the document and sits at package root",
                  os.path.exists(manifest_path), manifest_path)
            if os.path.exists(manifest_path):
                with open(manifest_path, encoding="utf-8") as fh:
                    manifest = json.load(fh)
        if manifest:
            hex64 = re.compile(r"^[0-9a-f]{64}$")
            check("the manifest declares the package schema and version",
                  manifest["schema"] == "talkify.commute-package/v1"
                  and manifest["package_version"] == 1, str(manifest)[:160])
            check("package_id is literally 'talkify:' plus the rendition id",
                  manifest["package_id"] == "talkify:" + manifest["rendition_id"]
                  and bool(hex64.match(manifest["rendition_id"])),
                  manifest["package_id"][:90])
            check("the content hash identifies the text, not the filename",
                  bool(hex64.match(manifest["document_content_sha256"])),
                  manifest["document_content_sha256"][:80])
            chs = manifest["chapters"]
            check("every detected chapter is in the manifest, in order, "
                  "one-based ordinals against zero-based indices",
                  [c["chapter_index"] for c in chs] == [0, 1]
                  and [c["ordinal"] for c in chs] == [1, 2], str(chs)[:200])
            check("block ranges are contiguous, non-empty and cover every "
                  "exported block exactly once",
                  chs[0]["block_start_index"] == 0
                  and all(c["block_start_index"] < c["block_end_index_exclusive"]
                          for c in chs)
                  and all(chs[i]["block_end_index_exclusive"]
                          == chs[i + 1]["block_start_index"]
                          for i in range(len(chs) - 1))
                  and chs[-1]["block_end_index_exclusive"] == len(cblocks),
                  str([(c["block_start_index"], c["block_end_index_exclusive"])
                       for c in chs]))
            check("a chapter's global start is the sum of the ones before it, "
                  "and the last one ends exactly at the package duration",
                  all(c["global_start_ms"]
                      == sum(p["duration_ms"] for p in chs[:i])
                      for i, c in enumerate(chs))
                  and (chs[-1]["global_start_ms"] + chs[-1]["duration_ms"]
                       == manifest["total_duration_ms"]),
                  str([(c["global_start_ms"], c["duration_ms"]) for c in chs]))
            check("no chapter is zero-length - the spec calls that invalid, and "
                  "it is what an empty chapter range would produce",
                  all(c["duration_ms"] > 0 for c in chs),
                  str([c["duration_ms"] for c in chs]))
            file_hashes_ok = True
            for c in chs:
                for key, name in (("audio_sha256", c["audio_file"]),
                                  ("timings_sha256", c["timings_file"])):
                    with open(os.path.join(cjob["dir"], name), "rb") as fh:
                        if hashlib.sha256(fh.read()).hexdigest() != c[key]:
                            file_hashes_ok = False
            check("every hash in the manifest matches the file that was "
                  "actually written - a truncated asset is detectable",
                  file_hashes_ok)
            check("the manifest names files that are really there, under the "
                  "ordinal-prefixed names",
                  all(os.path.exists(os.path.join(cjob["dir"], c["audio_file"]))
                      and os.path.exists(os.path.join(cjob["dir"], c["timings_file"]))
                      for c in chs)
                  and [c["audio_file"] for c in chs]
                  == ["001_The Opening.mp3", "002_The Second Half.mp3"],
                  str([c["audio_file"] for c in chs]))
            # Recomputed here from the spec's own words rather than by calling
            # commute.py, so a change to how the checksum is built has to be a
            # deliberate change to both.
            recomputed = hashlib.sha256(json.dumps(
                {"schema": manifest["schema"],
                 "package_version": manifest["package_version"],
                 "package_id": manifest["package_id"],
                 "rendition_id": manifest["rendition_id"],
                 "document_content_sha256": manifest["document_content_sha256"],
                 "total_duration_ms": manifest["total_duration_ms"],
                 "chapters": manifest["chapters"]},
                sort_keys=True, separators=(",", ":"),
                ensure_ascii=False).encode("utf-8")).hexdigest()
            check("the package checksum is over exactly the immutable members, "
                  "in canonical JSON",
                  manifest["package_checksum"] == {"algorithm": "sha256",
                                                   "value": recomputed},
                  f"{manifest['package_checksum'].get('value', '')[:16]} vs {recomputed[:16]}")
            check("the display title is NOT in the checksum, so renaming a book "
                  "cannot invalidate the sidecar sitting next to it",
                  "document_title" in manifest, str(manifest)[:80])
            check("each chapter's own timings start at zero and end inside the "
                  "duration the manifest declares for it",
                  all(w[0]["s"] == 0.0
                      and w[-1]["e"] <= c["duration_ms"] / 1000 + 0.001
                      for w, c in zip(chap_words, chs)),
                  str([(w[-1]["e"], c["duration_ms"]) for w, c in zip(chap_words, chs)]))

            def read_id3(path):
                """Parse the tag back off the front of an MP3.

                Deliberately not by calling commute.id3v2_3(): the whole point
                is that a *reader* can find these, and the tag size is
                syncsafe (seven bits a byte) while each frame size in v2.3 is
                a plain integer. Writing both the same way produces a tag one
                player shows and the next treats as noise."""
                with open(path, "rb") as fh:
                    head = fh.read(10)
                    if head[:3] != b"ID3":
                        return None, 0
                    size = ((head[6] << 21) | (head[7] << 14)
                            | (head[8] << 7) | head[9])
                    body, first = fh.read(size), fh.read(2)
                frames, at = {}, 0
                while at + 10 <= len(body) and body[at:at + 4].strip(b"\x00"):
                    fid = body[at:at + 4].decode("ascii")
                    length = int.from_bytes(body[at + 4:at + 8], "big")
                    payload = body[at + 10:at + 10 + length]
                    # Encoding byte 1 is UTF-16 with a BOM; strip the trailing
                    # double-null terminator before decoding.
                    frames[fid] = (payload[1:].decode("utf-16")
                                   if payload[:1] == b"\x01"
                                   else payload[1:].decode("latin-1")).rstrip("\x00")
                    at += 10 + length
                return frames, first

            tags, after_tag = read_id3(files[0]["path"])
            check("a chapter mp3 carries ID3 tags, so a player that reads them "
                  "shows a chapter title rather than a filename",
                  tags is not None and tags.get("TIT2") == "The Opening"
                  and tags.get("TALB") == "Chaptered", str(tags)[:200])
            check("...numbered within the document, one-based, out of the real "
                  "chapter count",
                  tags is not None and tags.get("TRCK") == "1/2"
                  and tags.get("TPOS") == "1/1"
                  and tags.get("TSOT") == "001_The Opening", str(tags)[:200])
            check("...and the audio still begins with a real MP3 frame right "
                  "after the tag - a mis-sized tag eats the first frame",
                  after_tag[:1] == b"\xff" and (after_tag[1] & 0xE0) == 0xE0,
                  after_tag.hex())
            check("the single-file export is untagged, byte-for-byte the file "
                  "it always was",
                  single_export_path is None
                  or open(single_export_path, "rb").read(3) != b"ID3")

        print("position sidecar (the one file in a package that changes)")
        POSITION_FIELDS = {"schema", "package_id", "rendition_id", "elapsed_ms",
                           "block_index", "word_index", "updated_at",
                           "total_duration_ms", "checksum", "update_id",
                           "writer_id", "chapter_index", "chapter_elapsed_ms"}
        UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-"
                             r"[0-9a-f]{4}-[0-9a-f]{12}$")
        RFC3339_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{3}Z$")

        def read_position(path):
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)

        pkg_position_path = (os.path.join(cjob["dir"], "Chaptered.position.json")
                             if cjob and cjob.get("dir") else None)
        check("a chapter export writes a position sidecar at package root",
              pkg_position_path and os.path.exists(pkg_position_path),
              str(pkg_position_path))
        if pkg_position_path and os.path.exists(pkg_position_path) and manifest:
            pos = read_position(pkg_position_path)
            check("the sidecar carries every field the schema requires",
                  {"schema", "package_id", "rendition_id", "elapsed_ms",
                   "block_index", "word_index", "updated_at", "total_duration_ms",
                   "checksum", "update_id"} <= set(pos), str(sorted(pos)))
            check("...and nothing the schema does not allow "
                  "(additionalProperties is false)",
                  set(pos) <= POSITION_FIELDS,
                  str(sorted(set(pos) - POSITION_FIELDS)))
            check("the sidecar names the package it belongs to",
                  pos["schema"] == "talkify.position/v1"
                  and pos["package_id"] == manifest["package_id"]
                  and pos["rendition_id"] == manifest["rendition_id"],
                  str(pos)[:160])
            check("its checksum is the manifest's package checksum - a "
                  "reference to immutable contents, not a hash of itself",
                  pos["checksum"] == {"algorithm": "sha256",
                                      "scope": "talkify.commute-package/v1",
                                      "value": manifest["package_checksum"]["value"]},
                  str(pos["checksum"])[:160])
            check("it agrees with the manifest about how long the package is",
                  pos["total_duration_ms"] == manifest["total_duration_ms"],
                  f"{pos['total_duration_ms']} vs {manifest['total_duration_ms']}")
            check("update_id is a uuid and updated_at is RFC 3339 UTC with "
                  "milliseconds - the two fields a conflict is resolved on",
                  bool(UUID_RE.match(pos["update_id"]))
                  and bool(RFC3339_RE.match(pos["updated_at"])),
                  f"{pos['update_id']} {pos['updated_at']}")
            check("in a chapter package the chapter coordinates are present "
                  "as a pair, never one without the other",
                  ("chapter_index" in pos) == ("chapter_elapsed_ms" in pos)
                  and "chapter_index" in pos, str(sorted(pos)))
            check("elapsed_ms is the chapter's global start plus how far into "
                  "it - the identity a phone resumes on",
                  pos["elapsed_ms"] == (chs[pos["chapter_index"]]["global_start_ms"]
                                        + pos["chapter_elapsed_ms"]),
                  str(pos)[:200])
            check("a never-played document's sidecar starts at the beginning, "
                  "not at some invented offset",
                  pos["elapsed_ms"] == 0 and pos["block_index"] == 0
                  and pos["chapter_index"] == 0, str(pos)[:200])

            # The heartbeat is the only thing that moves a sidecar afterwards.
            before = pos
            st, _ = post("/api/position", {"doc": cdoc_id, "block": mid,
                                           "word": 1, "within": 0.5,
                                           "settled": True})
            after = read_position(pkg_position_path)
            check("playing on updates the exported sidecar, with a fresh "
                  "update_id so a conflict can be resolved",
                  st == 200 and after["update_id"] != before["update_id"]
                  and after["elapsed_ms"] > before["elapsed_ms"], str(after)[:200])
            check("...and it lands in the chapter that block belongs to",
                  after["chapter_index"] == 1 and after["block_index"] == mid,
                  str(after)[:200])
            check("...still satisfying global start plus chapter elapsed",
                  after["elapsed_ms"] == (chs[after["chapter_index"]]["global_start_ms"]
                                          + after["chapter_elapsed_ms"])
                  and 0 <= after["chapter_elapsed_ms"]
                  <= chs[after["chapter_index"]]["duration_ms"], str(after)[:200])
            check("...and the immutable half of the sidecar did not move",
                  (after["package_id"], after["rendition_id"], after["checksum"],
                   after["total_duration_ms"])
                  == (before["package_id"], before["rendition_id"],
                      before["checksum"], before["total_duration_ms"]),
                  str(after)[:200])
            check("the sidecar is the only package file the heartbeat touches",
                  all(hashlib.sha256(open(os.path.join(cjob["dir"], c["audio_file"]),
                                          "rb").read()).hexdigest() == c["audio_sha256"]
                      for c in chs))
            check("a position heartbeat leaves no .part file behind",
                  not [f for f in os.listdir(cjob["dir"]) if f.endswith(".part")],
                  str(os.listdir(cjob["dir"]))[:200])

        single_position = (single_export_path.rsplit(".", 1)[0] + ".position.json"
                           if single_export_path else "")
        check("a single-file export gets a sidecar too, beside its mp3 and "
              "timings rather than in a folder",
              bool(single_position) and os.path.exists(single_position),
              str(sorted(os.listdir(tmp_exports)))[:240])
        if single_position and os.path.exists(single_position):
            spos = read_position(single_position)
            check("the single-file sidecar is the same schema and just as "
                  "complete - it has no manifest to lean on",
                  spos["schema"] == "talkify.position/v1"
                  and {"package_id", "rendition_id", "checksum", "update_id",
                       "elapsed_ms", "total_duration_ms"} <= set(spos)
                  and spos["total_duration_ms"] > 0, str(spos)[:200])
            check("...and carries no chapter coordinates, because a single "
                  "file has no chapters to index",
                  "chapter_index" not in spos and "chapter_elapsed_ms" not in spos,
                  str(sorted(spos)))
            check("a document exported in two different renditions does not "
                  "share one identity",
                  spos["rendition_id"] != (manifest or {}).get("rendition_id"),
                  spos["rendition_id"][:16])

        print("export dialog wiring (an id typo here is invisible without a browser)")
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "web", "app.js"), encoding="utf-8") as fh:
            app_js = fh.read()
        with open(os.path.join(here, "web", "index.html"), encoding="utf-8") as fh:
            index_html = fh.read()
        with open(os.path.join(here, "web", "style.css"), encoding="utf-8") as fh:
            style_css = fh.read()
        html_ids = set(re.findall(r'id="([^"]+)"', index_html))
        wanted = sorted(set(re.findall(r'\$\("(export[A-Za-z0-9_]*)"\)', app_js)))
        missing = [i for i in wanted if i not in html_ids]
        check("every #export... element app.js reaches for exists in index.html",
              wanted and not missing, f"missing {missing}")
        for el in ("exportVoiceWrap", "exportModeWrap", "exportActiveVoice"):
            check(f"#{el} exists - the picker is swapped for a static line once a "
                  f"job starts", el in html_ids)
        # Closing the dialog goes back to reading; it does not cancel the export.
        # The job is keyed by document so re-opening can re-attach to it.
        check("an export job is tracked per document, so closing the dialog cannot "
              "lose it", "state.activeExports[docId]" in app_js
              and "activeExports: {}" in app_js)
        check("...and nothing stops polling when the dialog closes",
              'exportDlg.addEventListener("close"' not in app_js)
        check("re-opening the dialog asks the server what the job is doing now, "
              "rather than trusting what it last saw",
              "await fetchExportStatus(rec.job)" in app_js)

        print("an export that fails mid-book leaves nothing behind")
        import export as export_module
        from engines import StubEngine

        stub_engine = StubEngine()
        good_audio, good_mime, good_timings = stub_engine.synth("One block that encodes fine.")
        good_block = (good_audio, [{"w": w.w, "s": w.s, "e": w.e} for w in good_timings])

        def dies_halfway():
            yield good_block
            raise RuntimeError("synthesis died halfway through the book")

        part_dir = tempfile.mkdtemp(prefix="talkify_smoke_part_")
        mp3_dest = os.path.join(part_dir, "Half a book.mp3")
        json_dest = os.path.join(part_dir, "Half a book.json")
        raised = False
        try:
            export_module.export_stream(dies_halfway(), mp3_dest, json_dest, good_mime)
        except RuntimeError:
            raised = True
        check("a failure mid-export raises rather than quietly finishing", raised)
        check("...leaving no .part files in the export folder",
              os.listdir(part_dir) == [], str(os.listdir(part_dir)))
        check("...and no half-written mp3 or json where the finished ones would be",
              not os.path.exists(mp3_dest) and not os.path.exists(json_dest))
        export_module.export_stream(iter([good_block]), mp3_dest, json_dest, good_mime)
        check("...and the same destination still exports cleanly afterwards",
              os.path.exists(mp3_dest) and os.path.exists(json_dest)
              and sorted(os.listdir(part_dir)) == ["Half a book.json", "Half a book.mp3"],
              str(sorted(os.listdir(part_dir))))
        shutil.rmtree(part_dir, ignore_errors=True)

        print("delete")
        st, body = post("/api/open", {"text": "Throwaway document for delete test."})
        del_id = body["id"]
        st, body = post("/api/delete", {"doc": del_id})
        check("delete succeeds", st == 200 and body.get("ok"), str(body)[:80])
        st, body, _ = get("/api/docs")
        check("deleted document is gone from the library",
              del_id not in [d["id"] for d in json.loads(body)["docs"]], body[:160])
        check("deleted document's file is removed from disk",
              not os.path.exists(os.path.join(tmp_library, f"{del_id}.json")))
        st, body = post("/api/delete", {"doc": "nonexistent"})
        check("deleting an unknown document 404s cleanly", st == 404, str(body)[:80])

        # ------------------------------------------------------------------
        # Phase 0 release gates. These exercise the seams directly rather than
        # through HTTP: what they assert - a byte budget, a bounded queue, a
        # write that is never half-done - is invisible from the outside right
        # up until the day it costs someone their library or their RAM.
        # ------------------------------------------------------------------
        print("audio cache stays inside its byte budget")
        import threading as threading_module
        import coordinator as coordinator_module
        import storage as storage_module

        def blob(n):
            return b"\x00" * n

        cache = coordinator_module.AudioCache(max_bytes=10_000)
        for i in range(20):
            cache.put("cachedoc", i, "v", blob(1_000), "audio/wav", [])
        stats = cache.stats()
        check("inserting past the budget evicts instead of growing",
              stats["bytes"] <= stats["max_bytes"], str(stats))
        check("...and it actually evicted something", stats["evictions"] > 0, str(stats))
        check("...leaving the most recent blocks, not the oldest",
              cache.get("cachedoc", 19) is not None and cache.get("cachedoc", 0) is None,
              str(cache.cached_indices("cachedoc")))

        lru = coordinator_module.AudioCache(max_bytes=3_000)
        for i in range(3):
            lru.put("lru", i, "v", blob(1_000), "audio/wav", [])
        lru.get("lru", 0)                       # block 0 is now the most recently used
        lru.put("lru", 3, "v", blob(1_000), "audio/wav", [])
        check("eviction is least-recently-*used*, not least-recently-added",
              lru.get("lru", 0) is not None and lru.get("lru", 1) is None,
              str(lru.cached_indices("lru")))

        huge = coordinator_module.AudioCache(max_bytes=1_000)
        huge.put("huge", 0, "v", blob(5_000), "audio/wav", [])
        check("a single block bigger than the whole budget is refused, not "
              "stored over the cap", huge.stats()["bytes"] == 0, str(huge.stats()))

        switched = coordinator_module.AudioCache(max_bytes=1 << 20)
        for i in range(6):
            switched.put("sw", i, "old", blob(100), "audio/wav", [])
        switched.drop_after("sw", 2)
        check("a voice switch drops the unread blocks and keeps the heard ones",
              switched.cached_indices("sw") == [0, 1, 2], str(switched.cached_indices("sw")))

        # Which voices a document is cached in, and how much of it in each.
        # This is what the export dialog reads to say which voice exports
        # fastest, so it has to follow a mid-read voice switch exactly the way
        # the cache does - one voice per block, the new one replacing the old.
        counted = coordinator_module.AudioCache(max_bytes=1 << 20)
        check("a document nothing has been synthesized for has no voice counts",
              counted.voice_counts("vc") == {}, str(counted.voice_counts("vc")))
        for i in range(5):
            counted.put("vc", i, "af_heart", blob(100), "audio/wav", [])
        check("voice counts report every block cached in a voice",
              counted.voice_counts("vc") == {"af_heart": 5}, str(counted.voice_counts("vc")))
        for i in range(3, 8):
            counted.put("vc", i, "af_bella", blob(100), "audio/wav", [])
        check("...and split across two voices after a mid-document switch",
              counted.voice_counts("vc") == {"af_heart": 3, "af_bella": 5},
              str(counted.voice_counts("vc")))
        counted.put("other", 0, "af_sky", blob(100), "audio/wav", [])
        check("...counting only the document asked about",
              counted.voice_counts("vc") == {"af_heart": 3, "af_bella": 5}
              and counted.voice_counts("other") == {"af_sky": 1},
              str(counted.voice_counts("other")))

        print("synthesis coordinator: bounded, prioritised, generation-guarded")

        def fake_engine(delay=0.0):
            calls = []
            call_lock = threading_module.Lock()

            def synth(text, voice):
                with call_lock:
                    calls.append((text, voice))
                if delay:
                    time.sleep(delay)
                return f"{voice}".encode() * 64, "audio/wav", []
            return synth, calls

        # Export is a sequential scan, while AudioCache is a byte-bounded
        # LRU. If misses are admitted, blocks 0..N evict warmed blocks N+1..
        # just before the scan reaches them. Keep a cache exactly full of the
        # later half and prove export misses leave those future hits intact.
        synth_fn, calls = fake_engine()
        scan_cache = coordinator_module.AudioCache(max_bytes=4 * 64)
        for i in range(4, 8):
            scan_cache.put("scan", i, "v", b"v" * 64, "audio/wav", [])
        scan_safe = coordinator_module.SynthCoordinator(
            synth_fn, cache=scan_cache, warm_window=2, max_pending=8)
        scan_safe.set_text_provider(lambda d, i: f"block {i}")
        for i in range(8):
            scan_safe.synth("scan", i, f"block {i}", "v",
                            priority=coordinator_module.PRIORITY_EXPORT,
                            cache_result=False, timeout=20)
        check("export misses do not evict warmed blocks still ahead in the book",
              scan_cache.cached_indices("scan") == [4, 5, 6, 7],
              str(scan_cache.cached_indices("scan")))
        check("...so the warmed later half stays all cache hits",
              len(calls) == 4, f"{len(calls)} synth calls for 8 blocks")
        scan_safe.shutdown()

        # A live reader attaching to an export-prefetched job still needs the
        # result retained. Its retry/rewind latency is more important than
        # the export scan's admission policy.
        synth_fn, calls = fake_engine(delay=0.05)
        shared_cache = coordinator_module.AudioCache(max_bytes=1 << 20)
        shared_policy = coordinator_module.SynthCoordinator(
            synth_fn, cache=shared_cache, warm_window=1, max_pending=8)
        shared_policy.set_text_provider(lambda d, i: f"block {i}")
        shared_policy.warm("shared-policy", 1, voice="v",
                           priority=coordinator_module.PRIORITY_EXPORT,
                           cache_result=False)
        shared_policy.synth("shared-policy", 0, "block 0", "v",
                            priority=coordinator_module.PRIORITY_INTERACTIVE,
                            timeout=20)
        check("live playback forces a shared export-prefetch result into cache",
              shared_cache.get("shared-policy", 0, voice="v") is not None)
        shared_policy.shutdown()

        # The shape that matters, and the one that broke: a book half read and
        # then exported. Warmed blocks in the middle are hits, so the walk
        # arrives at the first block after them with no job of its own to
        # attach to - and the request it makes ties, at export priority, with
        # prefetch already queued behind it. A dispatcher that yields to a
        # waiter sorting *after* it hands the same request back to itself, and
        # this walk used to spin there for good with the progress bar frozen.
        # It also has to make each uncached block exactly once: prefetch that
        # finished early used to be dropped and then synthesized again, which
        # on real models is a second pass over the whole book.
        synth_fn, calls = fake_engine(delay=0.002)
        half_cache = coordinator_module.AudioCache(max_bytes=1 << 20)
        for i in range(10, 30):
            half_cache.put("halfread", i, "v", b"v" * 64, "audio/wav", [])
        walk = coordinator_module.SynthCoordinator(
            synth_fn, cache=half_cache, warm_window=8, max_pending=64)
        walk.set_text_provider(lambda d, i: f"block {i}")
        walk.warm("halfread", 40, start=0, voice="v",
                  priority=coordinator_module.PRIORITY_EXPORT,
                  cache_result=False, wrap=False)
        walked, stalled = 0, ""
        try:
            for i in range(40):
                walk.synth("halfread", i, f"block {i}", "v",
                           priority=coordinator_module.PRIORITY_EXPORT,
                           cache_result=False, timeout=20)
                walked += 1
        except Exception as e:
            stalled = f"{type(e).__name__}: {e}"
        check("an export walk over a half-warmed book runs to the end",
              walked == 40, stalled or f"{walked}/40 blocks")
        check("...synthesizing each uncached block once, and no block twice",
              len(calls) == 20, f"{len(calls)} synth calls for 20 uncached blocks")
        check("...with no dispatcher spinning on work it keeps handing back",
              walk.stats["yielded"] < 100, f"{walk.stats['yielded']} yields")
        walk.shutdown()

        # Prefetch is bounded by the warm window even when the consumer stops
        # collecting - an export pausing to encode must not let the producer
        # run away with a book's worth of WAV in memory - and every block it
        # made while stalled is handed over rather than made again.
        synth_fn, calls = fake_engine(delay=0.002)
        ahead = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 20),
            warm_window=4, max_pending=64)
        ahead.set_text_provider(lambda d, i: f"block {i}")
        ahead.warm("ahead", 20, start=0, voice="v",
                   priority=coordinator_module.PRIORITY_EXPORT,
                   cache_result=False, wrap=False)
        deadline = time.time() + 10
        while ahead.prefetched("ahead") < 4 and time.time() < deadline:
            time.sleep(0.01)
        time.sleep(0.15)                    # long enough to overrun if it could
        check("a stalled export walk prefetches its window and then stops",
              ahead.prefetched("ahead") == 4 and len(calls) == 4,
              f"{ahead.prefetched('ahead')} held, {len(calls)} of 20 blocks made")
        for i in range(20):
            ahead.synth("ahead", i, f"block {i}", "v",
                        priority=coordinator_module.PRIORITY_EXPORT,
                        cache_result=False, timeout=20)
        check("...and hands what it made to the walk instead of remaking it",
              len(calls) == 20, f"{len(calls)} synth calls for 20 blocks")
        check("...holding nothing once the walk has passed it",
              ahead.prefetched("ahead") == 0, str(ahead.prefetched("ahead")))
        ahead.shutdown()

        # An export starting at a chapter reads forward from it and never
        # looks behind. Warming what it has already passed would be engine
        # time nobody collects, and it would sit in the window in front of the
        # blocks the walk is actually waiting for.
        synth_fn, calls = fake_engine()
        forward = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 20),
            warm_window=4, max_pending=64)
        forward.set_text_provider(lambda d, i: f"block {i}")
        forward.warm("forward", 10, start=6, voice="v", wrap=False)
        deadline = time.time() + 10
        while (forward.outstanding("forward") or forward.pending()) and time.time() < deadline:
            time.sleep(0.01)
        check("a walk that does not wrap warms forward from its start only",
              sorted(t for t, _ in calls) == ["block 6", "block 7", "block 8", "block 9"],
              str(sorted(t for t, _ in calls)))
        forward.shutdown()

        # 1. Pending work is bounded. A 500-block book must still get fully
        #    warmed, but never with 500 requests queued to do it.
        synth_fn, calls = fake_engine()
        bounded = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=4, max_pending=16)
        bounded.set_text_provider(lambda d, i: f"block {i}")
        peak_outstanding = peak_pending = 0
        bounded.warm("big", 500, start=0, voice="v")
        for _ in range(600):
            peak_outstanding = max(peak_outstanding, bounded.outstanding("big"))
            peak_pending = max(peak_pending, bounded.pending())
            if bounded.cache.ready("big") >= 500:
                break
            time.sleep(0.01)
        check("a 500-block document warms all the way through",
              bounded.cache.ready("big") == 500, f"{bounded.cache.ready('big')}/500")
        check("...without ever exceeding the per-document warm window",
              peak_outstanding <= bounded.warm_window,
              f"peak {peak_outstanding} vs window {bounded.warm_window}")
        check("...or the queue's overall pending limit",
              peak_pending <= bounded.max_pending,
              f"peak {peak_pending} vs max {bounded.max_pending}")
        bounded.shutdown()

        # 2. A superseded generation is dropped without being synthesized.
        synth_fn, calls = fake_engine(delay=0.05)
        staled = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=3, max_pending=16)
        staled.set_text_provider(lambda d, i: f"block {i}")
        staled.warm("d", 40, start=0, voice="v1")
        staled.bump("d")                         # voice switch, immediately
        time.sleep(0.4)
        check("work queued under an old generation is discarded, not run",
              staled.stats["discarded"] >= 1, str(staled.stats))
        check("...and the discarded work never reached the engine",
              len(calls) <= 3, f"{len(calls)} synth calls for a 40-block plan")
        staled.shutdown()

        # 3. A voice switch mid-synthesis cannot land old audio in the cache.
        #    The fake engine tags its bytes with the voice, so "stale audio
        #    published" is directly observable here in a way it never is with
        #    the real one.
        synth_fn, calls = fake_engine(delay=0.03)
        switch = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=2, max_pending=16)
        switch.set_text_provider(lambda d, i: f"block {i}")
        switch.warm("v", 24, start=0, voice="old")
        time.sleep(0.02)                         # let one block get in flight
        switch.bump("v")
        switch.cache.drop_after("v", 0)
        switch.warm("v", 24, start=0, voice="new")
        for _ in range(200):
            if switch.cache.ready("v") >= 24:
                break
            time.sleep(0.01)
        held = [switch.cache.get("v", i) for i in range(1, 24)]
        check("after a voice switch no unread block holds the old voice's audio",
              all(e is not None and e.voice == "new" for e in held),
              str([e.voice if e else None for e in held]))
        switch.shutdown()

        # 4. An interactive request jumps a long warm backlog.
        synth_fn, calls = fake_engine(delay=0.02)
        urgent = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=8, max_pending=64)
        urgent.set_text_provider(lambda d, i: f"block {i}")
        urgent.warm("u", 300, start=0, voice="v")
        began = time.time()
        urgent.synth("u", 280, "block 280", "v",
                     priority=coordinator_module.PRIORITY_INTERACTIVE, timeout=20)
        waited = time.time() - began
        check("a block someone is waiting to hear beats the whole warm backlog",
              waited < 1.0, f"waited {waited:.2f}s (fair order would be ~5s)")
        urgent.shutdown()

        # 5. Two callers wanting the same block get one synthesis, not two -
        #    what the old per-block synth_locks dict was for.
        synth_fn, calls = fake_engine(delay=0.15)
        shared = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=1, max_pending=8)
        shared.set_text_provider(lambda d, i: f"block {i}")
        got = []
        def ask():
            got.append(shared.synth("s", 7, "block 7", "v", timeout=20))
        askers = [threading_module.Thread(target=ask) for _ in range(4)]
        for t in askers:
            t.start()
        for t in askers:
            t.join(20)
        check("four simultaneous requests for one block run TTS once",
              len(calls) == 1, f"{len(calls)} synth calls")
        check("...and all four get the same audio back",
              len(got) == 4 and all(g[0] == got[0][0] for g in got), f"{len(got)} results")
        shared.shutdown()

        # 6. Tiered warm priority. load_library() warms every document at
        #    startup, so the reader's book and a textbook nobody has open used
        #    to arrive at the same priority and alternate in FIFO order - the
        #    background book took half the worker. The open document's warm
        #    window now outranks the rest of the library.
        #
        #    B is queued first and deliberately given a head start, so a FIFO
        #    warmer would be most of the way through it before A got a look in.
        synth_fn, calls = fake_engine(delay=0.02)
        tiered = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=4, max_pending=64)
        tiered.set_text_provider(lambda d, i: f"{d} block {i}")
        check("no document is active until one is named",
              tiered.active_document() is None, str(tiered.active_document()))
        tiered.warm("B", 60, start=0, voice="v")     # background textbook, first
        time.sleep(0.05)                             # ...and already running
        tiered.set_active_document("A")
        check("the coordinator remembers which document is open",
              tiered.active_document() == "A", str(tiered.active_document()))
        tiered.warm("A", 6, start=0, voice="v")      # the book being read, second
        for _ in range(400):
            if tiered.cache.ready("A") >= 6:
                break
            time.sleep(0.01)
        b_done = tiered.cache.ready("B")
        check("the open document warms fully even though a 60-block one queued first",
              tiered.cache.ready("A") == 6, f"{tiered.cache.ready('A')}/6")
        # Fair FIFO alternation would have B at ~6 by now, and a pure
        # head-start reading of it at ~60; the tier means B only gets the
        # blocks that were already in flight or queued when A arrived.
        check("...and the background document does not siphon the worker while it does",
              b_done <= tiered.warm_window + 2, f"B finished {b_done} blocks alongside A's 6")
        # Once the reader's document is done there is nothing to outrank, so
        # the background book still finishes - lower tier, not starved.
        for _ in range(600):
            if tiered.cache.ready("B") >= 60:
                break
            time.sleep(0.01)
        check("...and the background document still finishes afterwards",
              tiered.cache.ready("B") == 60, f"{tiered.cache.ready('B')}/60")
        tiered.shutdown()

        # 7. Deleting the open document leaves nothing marked active, so the
        #    next refill does not hand its tier to a document that is gone.
        synth_fn, calls = fake_engine(delay=0.01)
        cleared = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=2, max_pending=16)
        cleared.set_text_provider(lambda d, i: f"block {i}")
        cleared.warm("gone", 8, start=0, voice="v")
        cleared.set_active_document("gone")
        cleared.forget("gone")
        check("deleting the open document clears the active tier",
              cleared.active_document() is None, str(cleared.active_document()))
        cleared.shutdown()

        # 8. An export is not starved by the warmer. This is a regression test
        #    for a real hang: export used to queue below every warm tier, and
        #    because _refill() tops the warm window back up after *every*
        #    finished block, an export block sat behind an endless supply of
        #    warm requests and the progress bar stopped for good. The export
        #    now outranks warming, while still yielding to live playback.
        check("an export outranks every kind of warming...",
              coordinator_module.PRIORITY_EXPORT
              < min(coordinator_module.PRIORITY_ACTIVE_WARM,
                    coordinator_module.PRIORITY_WARM,
                    coordinator_module.PRIORITY_BACKGROUND_WARM),
              f"export {coordinator_module.PRIORITY_EXPORT}")
        check("...but never the paragraph being listened to",
              coordinator_module.PRIORITY_INTERACTIVE
              < coordinator_module.PRIORITY_EXPORT,
              f"interactive {coordinator_module.PRIORITY_INTERACTIVE}")

        for tier, label in ((None, "a background library document"),
                            ("bg", "the open document")):
            synth_fn, calls = fake_engine(delay=0.02)
            exporting = coordinator_module.SynthCoordinator(
                synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
                warm_window=8, max_pending=64)
            exporting.set_text_provider(lambda d, i: f"block {i}")
            if tier:
                exporting.set_active_document(tier)
            exporting.warm("bg", 100, start=0, voice="v")
            time.sleep(0.05)                     # let the warm backlog get going
            began = time.time()
            exporting.synth("bg", 99, "block 99", "v",
                            priority=coordinator_module.PRIORITY_EXPORT, timeout=20)
            waited = time.time() - began
            warmed = exporting.cache.ready("bg")
            check(f"an export block jumps the warm backlog of {label}",
                  waited < 1.0, f"waited {waited:.2f}s (behind the backlog would be ~2s)")
            check(f"...arriving long before {label} finishes warming",
                  warmed < 90, f"{warmed}/100 blocks warmed by the time it landed")
            exporting.shutdown()

        # ...and live playback still preempts the export, so an export running
        # in the background cannot make the paragraph you are hearing late.
        synth_fn, calls = fake_engine(delay=0.02)
        mixed = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=8, max_pending=64)
        mixed.set_text_provider(lambda d, i: f"block {i}")
        exported = []

        def export_walk():
            for i in range(60):
                try:
                    mixed.synth("mx", i, f"block {i}", "v",
                                priority=coordinator_module.PRIORITY_EXPORT, timeout=20)
                    exported.append(i)
                except Exception:
                    return

        walker = threading_module.Thread(target=export_walk)
        walker.start()
        time.sleep(0.1)
        began = time.time()
        mixed.synth("mx", 59, "block 59", "v",
                    priority=coordinator_module.PRIORITY_INTERACTIVE, timeout=20)
        waited = time.time() - began
        check("a block being listened to still cuts ahead of a running export",
              waited < 0.5, f"waited {waited:.2f}s while an export was walking the book")
        walker.join(30)
        check("...and the export still runs to the end of the document",
              len(exported) == 60, f"{len(exported)}/60 blocks exported")
        mixed.shutdown()

        # 9. The player's own lookahead is not the paragraph being listened
        #    to. /api/block and /api/audio served every caller at
        #    PRIORITY_INTERACTIVE, so the two blocks the browser reads ahead
        #    sorted level with the block a click had just asked for - and
        #    ahead of it, because they were queued first. Clicking a word
        #    further down the page then waited out speculation nobody was
        #    listening to yet.
        check("the lookahead tier sits below live playback...",
              coordinator_module.PRIORITY_INTERACTIVE
              < coordinator_module.PRIORITY_PREFETCH,
              f"prefetch {coordinator_module.PRIORITY_PREFETCH}")
        check("...and above every kind of warming, and the export",
              coordinator_module.PRIORITY_PREFETCH
              < min(coordinator_module.PRIORITY_EXPORT,
                    coordinator_module.PRIORITY_ACTIVE_WARM,
                    coordinator_module.PRIORITY_WARM,
                    coordinator_module.PRIORITY_BACKGROUND_WARM),
              f"prefetch {coordinator_module.PRIORITY_PREFETCH}")

        synth_fn, calls = fake_engine(delay=0.05)
        seeking = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=8, max_pending=64)
        seeking.set_text_provider(lambda d, i: f"block {i}")
        read_ahead = []

        def look_ahead(i):
            try:
                seeking.synth("pf", i, f"block {i}", "v",
                              priority=coordinator_module.PRIORITY_PREFETCH,
                              timeout=20)
                read_ahead.append(i)
            except Exception:
                pass

        lookers = [threading_module.Thread(target=look_ahead, args=(i,))
                   for i in range(8)]
        for t in lookers:
            t.start()
        time.sleep(0.08)                     # all eight queued, one in flight
        began = time.time()
        seeking.synth("pf", 99, "block 99", "v",
                      priority=coordinator_module.PRIORITY_INTERACTIVE, timeout=20)
        waited = time.time() - began
        check("a seek cuts ahead of the lookahead the player already asked for",
              waited < 0.20,
              f"waited {waited:.2f}s (behind eight prefetches would be ~0.4s)")
        for t in lookers:
            t.join(20)
        check("...and the lookahead is not starved by having been jumped",
              len(read_ahead) == 8, f"{len(read_ahead)}/8 read-ahead blocks made")
        seeking.shutdown()

        # 10. Closing one book and opening another. set_active_document()
        #     re-aims the *next* window, but a heap cannot be reordered, so the
        #     book just closed kept a full window of PRIORITY_ACTIVE_WARM
        #     requests that tied with the new book's and won on arrival order.
        #     On real models that is eight blocks - about a minute - of the
        #     worker spent on text nobody is reading. They are re-tiered as
        #     each is claimed instead.
        synth_fn, calls = fake_engine(delay=0.02)
        switched = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=8, max_pending=64)
        switched.set_text_provider(lambda d, i: f"{d} block {i}")
        switched.set_active_document("A")
        switched.warm("A", 60, start=0, voice="v")
        time.sleep(0.05)                     # A's window is queued at the active tier
        switched.set_active_document("B")    # ...and now A is not the open book
        switched.warm("B", 6, start=0, voice="v")
        for _ in range(400):
            if switched.cache.ready("B") >= 6:
                break
            time.sleep(0.01)
        a_done = switched.cache.ready("A")
        check("the book just closed stops outranking the one just opened",
              switched.cache.ready("B") == 6, f"{switched.cache.ready('B')}/6")
        check("...because its queued window is re-tiered, not run at the old tier",
              a_done <= 5, f"A finished {a_done} blocks (a whole stale window is 8)")
        check("...which the coordinator counts rather than doing silently",
              switched.stats["retiered"] > 0, str(switched.stats["retiered"]))
        switched.shutdown()

        # 11. Tiering orders the queue; it does not decide who gets into it.
        #     load_library() warms every document at startup, so a shelf of
        #     books fills max_pending before the reader has opened anything -
        #     and then each completed background block refilled its own window
        #     straight back into the slot it had just freed. The open
        #     document's window was refused, over and over.
        synth_fn, calls = fake_engine(delay=1.0)     # slow enough that nothing drains
        crowded = coordinator_module.SynthCoordinator(
            synth_fn, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
            warm_window=4, max_pending=20)
        crowded.set_text_provider(lambda d, i: f"{d} block {i}")
        for n in range(20):                          # 80 blocks want 20 slots
            crowded.warm(f"bg{n}", 50, start=0, voice="v")
        crowded.set_active_document("open")
        crowded.warm("open", 50, start=0, voice="v")
        check("a shelf of background books cannot lock the open one out of the queue",
              crowded.outstanding("open") >= 3,
              f"{crowded.outstanding('open')} of the open book's window queued")
        check("...and the reservation is bounded, so the library still warms",
              crowded.outstanding() > crowded.outstanding("open"),
              f"{crowded.outstanding()} outstanding overall")
        crowded.shutdown()

        print("client lookahead (duplicate fetches, and what a voice switch lets go of)")
        prefetch_param_line = ('const prefetchParam = (speculative) =>'
                               ' (speculative ? "&prefetch=1" : "");')
        abort_status_line = 'if (!isAbort(e)) status("synthesis failed");'
        check("the timings half of a block coalesces duplicate fetches too",
              "blockInflight" in app_js
              and "joinInflight(state.blockInflight" in app_js)
        check("...which is what stops every paragraph boundary asking twice",
              "loadBlock(next, SPECULATIVE)" in app_js
              and "loadBlock(k, SPECULATIVE)" in app_js)
        check("the player's read-ahead tells the server it is speculative",
              prefetch_param_line in app_js
              and "${prefetchParam(speculative)}" in app_js)
        check("a request somebody started waiting on is pinned, never abandoned",
              "if (!speculative) prior.pinned = true;" in app_js
              and "if (!record || record.pinned) continue;" in app_js)
        check("a voice switch lets go of the lookahead before clearing the tables",
              app_js.index("abortStale(afterIndex);")
              < app_js.index("URL.revokeObjectURL(state.audioUrls[k]);"))
        check("...and an abandoned fetch is not reported as a synthesis failure",
              "if (isAbort(e)) return null;" in app_js
              and abort_status_line in app_js)

        print("append-only content (a book can grow while somebody is reading it)")

        import session as session_module
        import extract as extract_module
        import readaloud as readaloud_module

        check("OCR pool workers are told to die with the application that started them",
              "_ocr_worker_init" in readaloud_module.__dict__
              and "initializer=_ocr_worker_init" in open(
                  os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "readaloud.py"), encoding="utf-8").read())
        check("the session layer's block-length limit matches extraction's",
              session_module.MAX_BLOCK_CHARS == extract_module.MAX_BLOCK,
              f"{session_module.MAX_BLOCK_CHARS} vs {extract_module.MAX_BLOCK}")

        def growing_doc(blocks=("page zero.", "still page zero."), pages=(0, 0),
                        task="ocr-1"):
            mgr = session_module.SessionManager()
            doc = mgr.new_document("g1", "Growing", list(blocks),
                                   block_pages=list(pages),
                                   ingestion={"task_id": task, "state": "running",
                                              "pages_total": 4, "pages_processed": 1,
                                              "next_ordinal": 1, "failed_pages": []})
            mgr.add(doc)
            return mgr, doc

        mgr, doc = growing_doc()
        first = doc.content_snapshot()
        after = doc.append_content(first.revision, ["page one text."], [1],
                                   ingestion={**first.ingestion, "pages_processed": 2,
                                              "next_ordinal": 2},
                                   task_id="ocr-1")
        check("an append extends the document and moves nothing that was already there",
              after.blocks[:len(first.blocks)] == first.blocks
              and after.blocks[-1] == "page one text."
              and after.block_pages == (0, 0, 1), str(after.blocks))
        check("...and it is a new revision, so anything mid-flight can tell",
              after.revision == first.revision + 1, f"{first.revision} -> {after.revision}")
        check("...while the snapshot taken before it is untouched",
              first.blocks == ("page zero.", "still page zero.")
              and len(first.blocks) == 2, str(first.blocks))

        # A page that produces no text still has to move the progress count,
        # or a book full of full-page figures looks like a stalled scan.
        settled = doc.settle_ingestion({**after.ingestion, "pages_processed": 3,
                                        "next_ordinal": 3}, task_id="ocr-1")
        check("a page that yields no text advances progress without a content revision",
              settled.revision == after.revision
              and settled.ingestion_revision > after.ingestion_revision
              and settled.ingestion["pages_processed"] == 3,
              f"rev {settled.revision}, ingestion rev {settled.ingestion_revision}")

        # Every one of these means an appender has lost track of what it is
        # appending to, and the whole point is that a block already on screen
        # must never change.
        refusals = []
        for label, fn in (
                ("a stale revision",
                 lambda: doc.append_content(1, ["late"], [3], task_id="ocr-1")),
                ("another task's append",
                 lambda: doc.append_content(settled.revision, ["other"], [3],
                                            task_id="ocr-2")),
                ("a page before the frontier",
                 lambda: doc.append_content(settled.revision, ["backwards"], [0],
                                            task_id="ocr-1")),
                ("a block longer than the limit",
                 lambda: doc.append_content(settled.revision,
                                            ["x" * (session_module.MAX_BLOCK_CHARS + 1)],
                                            [3], task_id="ocr-1")),
                ("an empty block",
                 lambda: doc.append_content(settled.revision, ["   "], [3],
                                            task_id="ocr-1")),
                ("an append with nothing in it",
                 lambda: doc.append_content(settled.revision, [], [], task_id="ocr-1")),
                ("page numbers that do not match the blocks",
                 lambda: doc.append_content(settled.revision, ["a", "b"], [3],
                                            task_id="ocr-1"))):
            try:
                fn()
                refusals.append(label)
            except session_module.ContentError:
                pass
        check("every way of breaking append-only is refused, not guessed at",
              not refusals, f"accepted: {refusals}")
        check("...and none of them changed the document",
              doc.content_snapshot().blocks == after.blocks, str(doc.content_snapshot().blocks))

        # The reason ContentSnapshot exists: two properties read separately can
        # land on either side of an append and disagree about the length.
        mgr, doc = growing_doc()
        torn = []
        stop_growing = threading_module.Event()

        def grow():
            page = 1
            while not stop_growing.is_set() and page < 400:
                snap = doc.content_snapshot()
                try:
                    doc.append_content(snap.revision, [f"page {page} text."], [page],
                                       task_id="ocr-1")
                except session_module.ContentError:
                    pass
                page += 1

        def watch():
            while not stop_growing.is_set():
                snap = doc.content_snapshot()
                if len(snap.blocks) != len(snap.block_pages):
                    torn.append((len(snap.blocks), len(snap.block_pages)))
                # text_at() is what the coordinator's warm requests come
                # through; it must see a whole block or no block, never half
                # an append.
                text = doc.text_at(len(snap.blocks) - 1)
                if not text:
                    torn.append(("empty text at", len(snap.blocks) - 1))

        growers = [threading_module.Thread(target=grow) for _ in range(2)]
        watchers = [threading_module.Thread(target=watch) for _ in range(2)]
        for t in growers + watchers:
            t.start()
        for t in growers:
            t.join(30)
        stop_growing.set()
        for t in watchers:
            t.join(10)
        check("a document being appended to is never observed half-appended",
              not torn, str(torn[:3]))

        # UTF-8 all the way through: OCR'd book text is full of curly quotes
        # and em dashes, and this is the layer that hands them to the library.
        mgr, doc = growing_doc(blocks=("“Bearing stress,” he said—firmly.",),
                               pages=(0,))
        snap = doc.content_snapshot()
        doc.append_content(snap.revision, ["Δσ ≈ 3.2 × 10⁻⁴ … and so on."], [1],
                           task_id="ocr-1")
        rec = doc.record()
        check("appended text keeps its curly quotes, dashes and Greek",
              rec["blocks"][1].startswith("Δσ") and "—" in rec["blocks"][0],
              repr(rec["blocks"]))
        check("...and the record carries the ingestion state that describes it",
              rec["content_revision"] == 2 and rec["ingestion"]["task_id"] == "ocr-1",
              str(rec.get("ingestion")))
        reloaded = mgr.new_document(
            "g2", rec["title"], rec["blocks"], block_pages=rec["block_pages"],
            content_revision=rec["content_revision"], ingestion=rec["ingestion"])
        check("...so a reload comes back at the same revision, ready to continue",
              reloaded.content_snapshot().revision == 2
              and reloaded.ingesting() and reloaded.content_snapshot().blocks
              == doc.content_snapshot().blocks, str(reloaded.content_snapshot().revision))

        # A finished document says nothing about ingestion at all - an ordinary
        # library of text files keeps exactly the record shape it always had.
        plain = mgr.new_document("p1", "Plain", ["just text"])
        check("a document that was never progressively read stays complete and silent",
              not plain.ingesting() and "ingestion" not in plain.record()
              and plain.content_snapshot().ingestion == {"state": "complete"},
              str(plain.record().keys()))


        print("progressive ingestion (a scanned book is readable from page one)")
        # Driven through the real open/publish/settle/resume machinery with a
        # fake page source. smoke.py must not need Tesseract, and what fails
        # quietly here is the ordering and identity checking, not the OCR.
        ingest_library = tempfile.mkdtemp(prefix="talkify_smoke_ingest_")
        os.environ["TALKIFY_LIBRARY_DIR"] = ingest_library
        os.environ["TALKIFY_CONFIG_PATH"] = os.path.join(ingest_library, "config.json")
        import engines as engines_for_ingest
        import talkify as talkify_module
        # Warming is part of what a publish has to do, so give it an engine
        # that can actually speak. The stub needs no models and is what the
        # rest of this file runs against anyway.
        talkify_module.ENGINE = engines_for_ingest.StubEngine()

        # The route half of the lookahead tier. The flag has to be opt-in and
        # fail towards interactive: a browser running yesterday's cached
        # app.js sends no flag at all, and must keep exactly the priority it
        # has always had rather than quietly dropping a tier.
        check("a flagged request is queued as the player's lookahead",
              talkify_module._requested_priority({"prefetch": ["1"]})
              == coordinator_module.PRIORITY_PREFETCH)
        check("...and anything else is the paragraph somebody is waiting for",
              all(talkify_module._requested_priority(q)
                  == coordinator_module.PRIORITY_INTERACTIVE
                  for q in ({}, {"prefetch": [""]}, {"prefetch": ["0"]},
                            {"prefetch": ["maybe"]})))

        def fake_line(text, ypos=0.5):
            return {"text": text, "ypos": ypos, "x0": 0.0, "x1": 1.0,
                    "size": 10.0, "ocr": True}

        def page_source(pages, gate=None, errors=(), empty=()):
            """A page iterator with a page that can be held, a page that fails,
            and a page that yields nothing - the three cases a real scan
            produces and a happy-path test never sees."""
            def source(path, indices, cancelled=None):
                for i in indices:
                    if cancelled is not None and cancelled():
                        return
                    if gate is not None and i == gate[0]:
                        gate[1].wait(30)
                    if i in errors:
                        yield i, None, "Tesseract fell over"
                        continue
                    if i in empty:
                        yield i, [], None
                        continue
                    yield i, [fake_line(pages[i])], None
            return source

        def plan_for(path, count):
            # A real file with a real fingerprint: ingestion re-checks that the
            # source has not changed underneath it before every batch, and a
            # fixture that fails that check would pass this section for the
            # wrong reason.
            with open(path, "wb") as fh:
                fh.write(b"%PDF-1.4 not really, but it is the file we started on")
            return readaloud_module.PdfIngestionPlan(
                path=path, selected_pages=tuple(range(count)), progressive=True,
                source_fingerprint=readaloud_module.source_fingerprint(path),
                math="skip", drop_captions=True, page_count=count)

        pages = {i: f"Page {i} says something worth hearing about beams. "
                    f"It runs on for a sentence or two so the block is real."
                 for i in range(6)}

        # 1. The document exists and is readable before the third page has
        #    even been attempted. This is the whole feature.
        held = threading_module.Event()
        plan = plan_for(os.path.join(ingest_library, "scan.pdf"), 6)
        job = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
        doc = talkify_module._open_progressive(
            job, plan.path, "Scanned book", {}, plan,
            source=page_source(pages, gate=(2, held)))
        check("a scanned book opens from its first pages, not its last",
              doc is not None and len(doc.content_snapshot().blocks) > 0,
              "no prefix was readable")
        opened = doc.content_snapshot()
        check("...and says it is still being read",
              opened.ingestion["state"] == "running", str(opened.ingestion.get("state")))
        check("...with equal-length blocks and page numbers, all from the prefix",
              len(opened.blocks) == len(opened.block_pages)
              and max(opened.block_pages) <= 1
              and all(0 < len(b) <= session_module.MAX_BLOCK_CHARS
                      for b in opened.blocks),
              f"{len(opened.blocks)} blocks, pages {opened.block_pages}")
        prefix_text = opened.blocks
        prefix_pages = opened.block_pages

        # 2. Release the rest. The suffix appears; the prefix does not move.
        held.set()
        for _ in range(600):
            if not doc.ingesting():
                break
            time.sleep(0.02)
        grown = doc.content_snapshot()
        check("the rest of the book arrives behind the reader",
              len(grown.blocks) > len(prefix_text),
              f"{len(prefix_text)} -> {len(grown.blocks)} blocks")
        check("...and every block that was already on screen is byte-for-byte where it was",
              grown.blocks[:len(prefix_text)] == prefix_text
              and grown.block_pages[:len(prefix_pages)] == prefix_pages,
              "the prefix moved")
        check("...and ingestion settles complete",
              grown.ingestion["state"] == "complete"
              and grown.ingestion["pages_processed"] == 6, str(grown.ingestion))
        for _ in range(400):
            if talkify_module.COORD.cache.ready(doc.id) >= len(grown.blocks):
                break
            time.sleep(0.02)
        check("...with the blocks that appeared after opening warmed too",
              talkify_module.COORD.cache.ready(doc.id) == len(grown.blocks),
              f"{talkify_module.COORD.cache.ready(doc.id)}/{len(grown.blocks)} warmed")

        # 3. A page that fails is one page, and a page with no text still
        #    moves the count. Both are ordinary on a real scan.
        held2 = threading_module.Event()
        held2.set()
        plan2 = plan_for(os.path.join(ingest_library, "patchy.pdf"), 6)
        job2 = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
        doc2 = talkify_module._open_progressive(
            job2, plan2.path, "Patchy scan", {}, plan2,
            source=page_source(pages, errors={3}, empty={4}))
        for _ in range(600):
            if not doc2.ingesting():
                break
            time.sleep(0.02)
        final2 = doc2.content_snapshot()
        check("one unreadable page does not hide the pages after it",
              any("Page 5" in b for b in final2.blocks),
              str([b[:14] for b in final2.blocks]))
        check("...and the book settles partial, naming the page that failed",
              final2.ingestion["state"] == "partial"
              and final2.ingestion["failed_pages"] == [3], str(final2.ingestion))
        check("...while a page with no text on it still counts as read",
              final2.ingestion["pages_processed"] == 6
              and final2.ingestion["pages_succeeded"] == 5, str(final2.ingestion))
        # Both indexers start when ingestion settles and run in the analysis
        # pool, so "has it settled yet" is a question with a real answer and
        # asking it the instant ingestion ends is a race - one this test lost
        # on a machine that happened to be busy. The assertion is unchanged;
        # it just waits for the state it is asserting about to exist.
        for _ in range(600):
            if (doc2.chapter_detection.get("state") != "pending"
                    and doc2.figure_index.get("state") != "pending"):
                break
            time.sleep(0.02)
        check("a partial book leaves its chapter and figure indexes quietly absent",
              doc2.chapter_detection.get("state") == "none"
              and doc2.figure_index.get("state") == "none",
              f"{doc2.chapter_detection} / {doc2.figure_index}")

        # 4. Deleting a document while its scan is blocked must stop every
        #    later append, save and warm for that task.
        blocked = threading_module.Event()
        plan3 = plan_for(os.path.join(ingest_library, "doomed.pdf"), 6)
        job3 = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
        doc3 = talkify_module._open_progressive(
            job3, plan3.path, "Doomed", {}, plan3,
            source=page_source(pages, gate=(2, blocked)))
        at_delete = len(doc3.content_snapshot().blocks)
        talkify_module.SESSIONS.remove(doc3.id)
        talkify_module.INGESTION.cancel(doc3.id)
        blocked.set()
        time.sleep(0.4)
        check("a document deleted mid-scan takes no further text",
              len(doc3.content_snapshot().blocks) == at_delete,
              f"{at_delete} -> {len(doc3.content_snapshot().blocks)} blocks after deletion")

        # 5. A task whose document has been replaced under the same id is
        #    rejected by object identity, not by hoping the ids differ.
        replaced = threading_module.Event()
        plan4 = plan_for(os.path.join(ingest_library, "reopened.pdf"), 6)
        job4 = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
        doc4 = talkify_module._open_progressive(
            job4, plan4.path, "Reopened", {}, plan4,
            source=page_source(pages, gate=(2, replaced)))
        stand_in = talkify_module.SESSIONS.new_document(
            doc4.id, "Reopened again", ["different text entirely"])
        talkify_module.SESSIONS.add(stand_in)
        before_swap = len(doc4.content_snapshot().blocks)
        replaced.set()
        time.sleep(0.4)
        check("a scan cannot publish into a document that was reopened under its id",
              len(doc4.content_snapshot().blocks) == before_swap
              and len(stand_in.content_snapshot().blocks) == 1,
              f"{before_swap} -> {len(doc4.content_snapshot().blocks)}, "
              f"stand-in {len(stand_in.content_snapshot().blocks)}")
        talkify_module.SESSIONS.remove(doc4.id)

        # 6. Resume. A record left "running" by an unclean stop picks up at
        #    next_ordinal; a source that has gone away settles partial and
        #    keeps every block it already had.
        record = doc.record()
        record["ingestion"] = {**record.get("ingestion", {}), "state": "running",
                               "next_ordinal": 3, "task_id": "ocr-resume",
                               "source_fingerprint": "gone"}
        record["source"] = {"path": os.path.join(ingest_library, "vanished.pdf"),
                            "options": {}}
        revived = talkify_module.SESSIONS.new_document(
            "revive1", record["title"], record["blocks"],
            block_pages=record.get("block_pages"),
            content_revision=record.get("content_revision", 1),
            ingestion=record["ingestion"], source=record["source"])
        talkify_module.SESSIONS.add(revived)
        kept = revived.content_snapshot().blocks
        talkify_module._resume_ingestion(revived)
        check("a scan whose source file has gone settles partial and keeps its text",
              revived.content_snapshot().ingestion["state"] == "partial"
              and revived.content_snapshot().blocks == kept,
              str(revived.content_snapshot().ingestion.get("state")))
        talkify_module.SESSIONS.remove("revive1")

        # 8. The routes a growing document is read through, driven against the
        #    real handler. A second server in this process, so the documents
        #    it serves are the ones the fake scan above built - the throwaway
        #    server the rest of this file uses has no way to hold a document
        #    that is still being read.
        from http.server import ThreadingHTTPServer as _TServer
        live_srv = _TServer(("127.0.0.1", PORT + 1), talkify_module.Handler)
        threading_module.Thread(target=live_srv.serve_forever, daemon=True).start()
        live_base = f"http://127.0.0.1:{PORT + 1}"

        def live_get(p):
            try:
                with urllib.request.urlopen(live_base + p, timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        def live_post(p, obj):
            req = urllib.request.Request(
                live_base + p, data=json.dumps(obj).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    return r.status, json.loads(r.read())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read())

        try:
            st, body = live_get(f"/api/doc/blocks?doc={doc.id}&from=0")
            check("the suffix route returns the whole document from block zero",
                  st == 200 and body["blocks"] == list(grown.blocks)
                  and body["revision"] == grown.revision,
                  f"status {st}, {len(body.get('blocks', []))} blocks")
            cut = len(prefix_text)
            st, tail = live_get(f"/api/doc/blocks?doc={doc.id}&from={cut}")
            check("...and exactly the suffix from anywhere else",
                  st == 200 and tail["blocks"] == list(grown.blocks[cut:])
                  and tail["block_pages"] == list(grown.block_pages[cut:]),
                  f"status {st}, {len(tail.get('blocks', []))} blocks after {cut}")
            st2, again = live_get(f"/api/doc/blocks?doc={doc.id}&from={cut}")
            check("...and asking twice gives the same answer, so no poll can lose an update",
                  (st2, again["blocks"]) == (st, tail["blocks"]), "the second poll differed")
            st, done = live_get(f"/api/doc/blocks?doc={doc.id}&from={len(grown.blocks)}")
            check("...and asking from the end is empty, not an error",
                  st == 200 and done["blocks"] == []
                  and done["ingestion"]["state"] == "complete", f"status {st}, {done}")

            for label, query, want, code in (
                    ("an unknown document", "doc=nosuchthing&from=0", 404, "not_found"),
                    ("a missing cursor", f"doc={doc.id}", 400, "invalid_cursor"),
                    ("a cursor that is not a number", f"doc={doc.id}&from=x", 400,
                     "invalid_cursor"),
                    ("a negative cursor", f"doc={doc.id}&from=-1", 400, "invalid_cursor"),
                    ("a cursor past the end", f"doc={doc.id}&from=9999", 409,
                     "content_reset")):
                st, body = live_get(f"/api/doc/blocks?{query}")
                check(f"{label} is refused in JSON, with a code the client can act on",
                      st == want and body.get("code") == code, f"status {st}, {body}")

            # Export while a scan is running would write however much has been
            # read so far under the whole book's name, and nothing in the file
            # would say so.
            running_plan = plan_for(os.path.join(ingest_library, "midscan.pdf"), 6)
            gate5 = threading_module.Event()
            job5 = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
            mid = talkify_module._open_progressive(
                job5, running_plan.path, "Mid-scan", {}, running_plan,
                source=page_source(pages, gate=(2, gate5)))
            st, body = live_post("/api/export", {"doc": mid.id})
            check("exporting a book that is still being scanned is refused, not truncated",
                  st == 409 and body.get("code") == "ingestion_running"
                  and body["ingestion"]["state"] == "running", f"status {st}, {body}")
            gate5.set()
            for _ in range(600):
                if not mid.ingesting():
                    break
                time.sleep(0.02)
            st, body = live_post("/api/export", {"doc": mid.id})
            check("...and allowed the moment the scan finishes",
                  st == 200 and body.get("job"), f"status {st}, {body}")

            st, opened_doc = live_get(f"/api/doc?doc={doc.id}")
            check("/api/doc carries the content revision and ingestion state",
                  st == 200 and opened_doc["content_revision"] == grown.revision
                  and opened_doc["ingestion"]["state"] == "complete",
                  f"status {st}, revision {opened_doc.get('content_revision')}")
        finally:
            live_srv.shutdown()
            live_srv.server_close()

        # 6b. A paragraph held back to join the next page is the easiest thing
        #     in this design to lose: it lives only in the task's memory. It
        #     must survive the page after it failing, and it must survive the
        #     process stopping.
        unfinished = dict(pages)
        unfinished[2] = "Page two ends mid-sentence and carries on"   # no full stop
        plan5 = plan_for(os.path.join(ingest_library, "carry.pdf"), 6)
        job5 = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
        doc5 = talkify_module._open_progressive(
            job5, plan5.path, "Carried", {}, plan5,
            source=page_source(unfinished, errors={3}))
        for _ in range(600):
            if not doc5.ingesting():
                break
            time.sleep(0.02)
        carried = doc5.content_snapshot()
        check("a paragraph carried across a page boundary survives the next page failing",
              any("ends mid-sentence" in b for b in carried.blocks),
              str([b[:26] for b in carried.blocks]))

        record5 = {"task_id": "ocr-resume", "state": "running", "next_ordinal": 3,
                   "pages_total": 6, "pages_processed": 3, "pages_succeeded": 3,
                   "failed_pages": [],
                   "carry": ["A paragraph the last run was still holding", 2],
                   "source_fingerprint": plan5.source_fingerprint}
        held_doc = talkify_module.SESSIONS.new_document(
            "resume-carry", "Resumed", ["page zero."], block_pages=[0],
            ingestion=record5,
            source={"path": plan5.path, "options": {}})
        talkify_module.SESSIONS.add(held_doc)
        resumed = talkify_module.IngestionTask(
            "ocr-resume", held_doc, plan5, ordinal=3,
            carry=(record5["carry"][0], record5["carry"][1]),
            source=page_source(unfinished))
        talkify_module.INGESTION.start(resumed)
        for _ in range(600):
            if not held_doc.ingesting():
                break
            time.sleep(0.02)
        after = held_doc.content_snapshot()
        check("...and a scan resumed after an unclean stop still has it",
              any("still holding" in b for b in after.blocks),
              str([b[:30] for b in after.blocks]))
        check("...published in reading order, before the pages that follow it",
              next(i for i, b in enumerate(after.blocks) if "still holding" in b)
              < next(i for i, b in enumerate(after.blocks) if "Page 4" in b),
              str([b[:22] for b in after.blocks]))
        talkify_module.SESSIONS.remove("resume-carry")

        # 6c. Cancelling takes the task's claim on the document away rather
        #     than only asking it to stop - a publish already past its own
        #     check would otherwise append and then decline to warm what it
        #     appended, leaving blocks with no audio.
        gate6 = threading_module.Event()
        plan6 = plan_for(os.path.join(ingest_library, "revoked.pdf"), 6)
        job6 = talkify_module.OPEN_JOBS.create(status="running", done=0, total=0)
        doc6 = talkify_module._open_progressive(
            job6, plan6.path, "Revoked", {}, plan6,
            source=page_source(pages, gate=(2, gate6)))
        talkify_module.INGESTION.cancel(doc6.id)
        check("cancelling a scan revokes its claim on the document",
              doc6.ingestion.get("task_id") is None, str(doc6.ingestion.get("task_id")))
        at_cancel = len(doc6.content_snapshot().blocks)
        gate6.set()
        time.sleep(0.4)
        check("...so nothing it was mid-way through can still land",
              len(doc6.content_snapshot().blocks) == at_cancel,
              f"{at_cancel} -> {len(doc6.content_snapshot().blocks)}")
        talkify_module.SESSIONS.remove(doc6.id)

        for bad in ("nan", float("inf"), "-inf"):
            pos = session_module.Position.from_dict({"block": 1, "within": bad})
            check(f"a position of {bad!r} is read as no position, not carried into arithmetic",
                  pos.within == 0.0, str(pos.within))

        # 7. What the browser is told. The full ingestion record carries task
        #    ids and fingerprints; none of that is a reader's business.
        public = talkify_module._public_ingestion(grown)
        check("the browser is told the state and the counts, not the bookkeeping",
              public["state"] == "complete" and "pages_total" in public
              and "task_id" not in public and "source_fingerprint" not in public,
              str(sorted(public)))


        print("pipelined coordinator: two blocks in flight, and a seek that stops one")

        class _Handle:
            """A pipelined engine's handle, faked. `cancel()` records whether
            the coordinator's lock was held when it was called - cancelling
            Kokoro means writing down a subprocess pipe, and doing that under
            the coordinator lock would put the session lock, the coordinator
            lock and a pipe write in one critical section."""

            def __init__(self, engine, text, voice):
                self.engine = engine
                self.text = text
                self.voice = voice
                self.cancelled = False
                self.done = threading_module.Event()
                self.value = None
                self.error = None

            def run(self):
                try:
                    self.value = self.engine.work(self)
                except Exception as e:
                    self.error = e
                self.done.set()

            def result(self, timeout=None):
                if not self.done.wait(timeout if timeout is not None else 30):
                    raise TimeoutError("fake engine stalled")
                if self.error is not None:
                    raise self.error
                return self.value

            def cancel(self):
                self.cancelled = True
                self.engine.cancels.append(self.text)
                if self.engine.lock is not None and self.engine.lock._is_owned():
                    self.engine.cancelled_under_lock.append(self.text)

        class _PipelinedEngine:
            """Two stages, one thread each, exactly like the Kokoro worker: a
            block moves TTS -> ASR, and the second block's TTS may overlap the
            first block's ASR."""

            def __init__(self, tts_delay=0.02, asr_delay=0.02, fail=(),
                         fatal_at=None):
                self.calls = []
                self.cancels = []
                self.cancelled_under_lock = []
                self.overlapped = False
                self.lock = None
                self.tts_delay = tts_delay
                self.asr_delay = asr_delay
                self.fail = set(fail)
                self.fatal_at = fatal_at
                self._stage_lock = threading_module.Lock()
                self._in_tts = self._in_asr = 0
                self._guard = threading_module.Lock()

            def submit(self, text, voice, meta):
                handle = _Handle(self, text, voice)
                threading_module.Thread(target=handle.run, daemon=True).start()
                return handle

            def synth(self, text, voice):
                return self.submit(text, voice, None).result(30)

            def work(self, handle):
                with self._guard:
                    self.calls.append((handle.text, handle.voice))
                if handle.text in self.fail:
                    raise RuntimeError(f"no audio for {handle.text}")
                if self.fatal_at is not None and handle.text == self.fatal_at:
                    boom = RuntimeError("the engine itself is broken")
                    boom.fatal = True
                    raise boom
                self._stage("_in_tts", self.tts_delay)
                if handle.cancelled:
                    stale = RuntimeError("cancelled")
                    stale.code = "cancelled"
                    raise stale
                self._stage("_in_asr", self.asr_delay)
                return f"{handle.voice}:{handle.text}".encode(), "audio/wav", []

            def _stage(self, name, delay):
                with self._stage_lock:
                    setattr(self, name, getattr(self, name) + 1)
                    if self._in_tts and self._in_asr:
                        self.overlapped = True
                time.sleep(delay)
                with self._stage_lock:
                    setattr(self, name, getattr(self, name) - 1)

        def pipelined(engine, **kw):
            coord = coordinator_module.SynthCoordinator(
                engine.synth, cache=coordinator_module.AudioCache(max_bytes=1 << 24),
                submit_fn=engine.submit, max_inflight=2, **kw)
            engine.lock = coord._lock
            return coord

        # 1. Two dispatchers really do put two blocks in the engine at once,
        #    and every block still arrives exactly once.
        engine = _PipelinedEngine()
        two = pipelined(engine, warm_window=4, max_pending=32)
        two.set_text_provider(lambda d, i: f"block {i}")
        two.warm("p", 30, start=0, voice="v")
        for _ in range(600):
            if two.cache.ready("p") >= 30:
                break
            time.sleep(0.01)
        check("two dispatchers keep both engine stages busy",
              engine.overlapped, "the two stages never overlapped")
        check("...and a 30-block document is still warmed exactly once through",
              two.cache.ready("p") == 30 and len(engine.calls) == 30,
              f"{two.cache.ready('p')} cached, {len(engine.calls)} calls")
        two.shutdown()

        # 2. A seek cancels the block that is already inside the engine, not
        #    just the ones still queued - and does it with no lock held.
        engine = _PipelinedEngine(tts_delay=0.2, asr_delay=0.2)
        seeking = pipelined(engine, warm_window=4, max_pending=32)
        seeking.set_text_provider(lambda d, i: f"block {i}")
        seeking.warm("s", 40, start=0, voice="old")
        time.sleep(0.15)                      # blocks are in the engine now
        seeking.bump("s")
        check("a seek cancels work already inside the engine",
              engine.cancels, "nothing was cancelled")
        check("...and never while holding the coordinator lock",
              not engine.cancelled_under_lock, str(engine.cancelled_under_lock))
        seeking.cache.drop_after("s", -1)
        time.sleep(0.6)                       # let every cancelled call unwind
        stale = [i for i in range(40)
                 if (entry := seeking.cache.get("s", i)) is not None
                 and entry.voice == "old"]
        check("...and nothing synthesized before the seek lands in the cache after it",
              not stale, f"blocks {stale[:5]} still hold pre-seek audio")
        seeking.shutdown()

        # 3. One block that will not synthesize is one block, not one book.
        #    The old behaviour dropped the whole warm plan on the first error.
        engine = _PipelinedEngine(tts_delay=0.005, asr_delay=0.005,
                                  fail=("block 3",))
        patchy = pipelined(engine, warm_window=4, max_pending=32)
        patchy.set_text_provider(lambda d, i: f"block {i}")
        patchy.warm("f", 12, start=0, voice="v")
        for _ in range(400):
            if patchy.cache.ready("f") >= 11:
                break
            time.sleep(0.01)
        attempts = [t for t, _v in engine.calls].count("block 3")
        check("a block that fails does not abandon the rest of the document",
              patchy.cache.ready("f") == 11, f"{patchy.cache.ready('f')}/11 others")
        check("...and speculative work does not retry it forever",
              attempts == 1, f"{attempts} attempts at the failing block")
        patchy.shutdown()

        # 4. A broken *engine* is different from a broken block: every block
        #    would fail the same way, so warming stops instead of failing once
        #    per paragraph for the rest of a book.
        engine = _PipelinedEngine(tts_delay=0.005, asr_delay=0.005,
                                  fatal_at="block 0")
        broken = pipelined(engine, warm_window=2, max_pending=16)
        broken.set_text_provider(lambda d, i: f"block {i}")
        broken.warm("b", 200, start=0, voice="v")
        time.sleep(0.5)
        check("a fatal engine failure stops warming instead of failing 200 times",
              len(engine.calls) < 20, f"{len(engine.calls)} calls after a fatal error")
        broken.shutdown()

        # 5. extend(): blocks that appear after the plan was built still get
        #    warmed, and the ones already made are not made again. This is the
        #    seam progressive ingestion appends through.
        engine = _PipelinedEngine(tts_delay=0.005, asr_delay=0.005)
        growing = pipelined(engine, warm_window=4, max_pending=32)
        growing.set_text_provider(lambda d, i: f"block {i}")
        growing.warm("g", 4, start=0, voice="v")
        for _ in range(200):
            if growing.cache.ready("g") >= 4:
                break
            time.sleep(0.01)
        before = len(engine.calls)
        growing.extend("g", 4, 9, voice="v")
        for _ in range(200):
            if growing.cache.ready("g") >= 9:
                break
            time.sleep(0.01)
        check("blocks appended after the warm plan was built are warmed too",
              growing.cache.ready("g") == 9, f"{growing.cache.ready('g')}/9")
        check("...and the ones already made are not made again",
              len(engine.calls) - before == 5,
              f"{len(engine.calls) - before} calls for 5 new blocks")
        growing.shutdown()

        # 6. A dispatcher that has already claimed speculative work steps
        #    aside when somebody is actually waiting. With one dispatcher the
        #    priority queue did all of this on its own; with two there is a
        #    window between claiming a request and starting it.
        engine = _PipelinedEngine(tts_delay=0.02, asr_delay=0.02)
        urgent = pipelined(engine, warm_window=8, max_pending=64)
        urgent.set_text_provider(lambda d, i: f"block {i}")
        urgent.warm("w", 400, start=0, voice="v")
        began = time.time()
        urgent.synth("w", 399, "block 399", "v",
                     priority=coordinator_module.PRIORITY_INTERACTIVE, timeout=30)
        waited = time.time() - began
        check("a block someone is waiting to hear still beats a warm backlog"
              " with two dispatchers",
              waited < 1.5, f"waited {waited:.2f}s")
        urgent.shutdown()


        print("pipelined synthesis: two blocks in the worker, replies matched by id")
        import queue as queue_module
        import engines as engines_module
        import kokoro_worker as worker_module

        # The worker's two stages are injectable, so everything about
        # admission, ordering, cancellation and per-stage failure can be
        # driven here without Kokoro, Whisper or a subprocess. If this needed
        # models it would never run, and this is the state machine that fails
        # quietly.
        class _Feed:
            """Stands in for the worker's stdin: a blocking line iterator the
            test feeds requests into, ending on close()."""

            def __init__(self):
                self.q = queue_module.Queue()

            def send(self, obj):
                self.q.put(json.dumps(obj) + "\n")

            def raw(self, line):
                self.q.put(line)

            def close(self):
                self.q.put(None)

            def __iter__(self):
                return self

            def __next__(self):
                item = self.q.get()
                if item is None:
                    raise StopIteration
                return item

        class _Sink:
            def __init__(self):
                self.q = queue_module.Queue()

            def write(self, s):
                self.q.put(json.loads(s))

            def flush(self):
                pass

            def take(self, timeout=10):
                return self.q.get(timeout=timeout)

        def run_worker(tts, asr, limits=None):
            feed, sink, stats = _Feed(), _Sink(), {}
            t = threading_module.Thread(
                target=worker_module.serve, args=(feed, sink),
                kwargs=dict(tts=tts, asr=asr, stats=stats,
                            limits=limits or worker_module.Limits()),
                daemon=True)
            t.start()
            return feed, sink, stats, t

        def ask(feed, rid, text="hello there"):
            feed.send({"op": "synth", "req_id": rid, "text": text})

        # 1. The point of the whole feature: while block 0 is in Whisper,
        #    block 1 is already through Kokoro. Serialised, that second TTS
        #    could not have started.
        seen, seen_lock = [], threading_module.Lock()
        asr_entered = {r: threading_module.Event() for r in ("0", "1")}
        asr_release = {r: threading_module.Event() for r in ("0", "1")}

        def overlap_tts(req):
            with seen_lock:
                seen.append(("tts", req.req_id))
            return b"WAV" + req.req_id.encode()

        def overlap_asr(req, wav):
            with seen_lock:
                seen.append(("asr", req.req_id))
            asr_entered[req.req_id].set()
            asr_release[req.req_id].wait(10)
            return [{"w": "hello", "s": 0.0, "e": 0.2},
                    {"w": "there", "s": 0.2, "e": 0.4}]

        feed, sink, stats, worker_thread = run_worker(overlap_tts, overlap_asr)
        ask(feed, "0")
        ask(feed, "1")
        entered = asr_entered["0"].wait(10)
        time.sleep(0.3)                       # let the TTS stage get ahead
        with seen_lock:
            snapshot = list(seen)
        check("block 1 goes through TTS while block 0 is still in Whisper",
              entered and ("tts", "1") in snapshot, str(snapshot))
        asr_release["0"].set()
        asr_release["1"].set()
        replies = {sink.take()["req_id"]: True for _ in range(2)}
        check("...and both blocks come back, each answered exactly once",
              set(replies) == {"0", "1"}, str(replies))
        feed.close()
        worker_thread.join(10)

        # 2. A lone request still runs TTS then ASR immediately - no batching
        #    delay, because a reader waiting on one paragraph is the case that
        #    must not get slower.
        lone = []

        def lone_tts(req):
            lone.append("tts")
            return b"WAV"

        def lone_asr(req, wav):
            lone.append("asr")
            return [{"w": "hello", "s": 0.0, "e": 0.2}]

        feed, sink, stats, worker_thread = run_worker(lone_tts, lone_asr)
        began = time.time()
        ask(feed, "solo")
        reply = sink.take()
        check("a single block is not delayed waiting for a second to pair with",
              reply["ok"] and lone == ["tts", "asr"] and time.time() - began < 2.0,
              f"{lone} in {time.time() - began:.2f}s")
        feed.close()
        worker_thread.join(10)

        # Native duration output takes the same bounded path but never calls
        # Whisper. This is the latency win Feature 3 exists to deliver.
        native_calls = []

        def native_tts(req):
            return worker_module._TTSOutput(
                b"WAV",
                [{"w": "hello", "s": 0.02, "e": 0.20},
                 {"w": "there", "s": 0.22, "e": 0.43}],
                None, 1.0)

        def forbidden_asr(req, wav):
            native_calls.append(req.req_id)
            raise AssertionError("Whisper ran for valid native timing")

        feed, sink, stats, worker_thread = run_worker(native_tts, forbidden_asr)
        ask(feed, "native")
        reply = sink.take()
        check("valid Kokoro duration marks bypass Whisper entirely",
              reply["ok"] and not native_calls and stats["asr_calls"] == 0
              and reply["alignment_method"] == "kokoro-pytorch-duration-v1",
              f"{reply} / {native_calls} / {stats}")
        check("...and report complete native coverage without changing audio transport",
              reply["native_direct_coverage"] == 1.0
              and "audio_b64" in reply and len(reply["heard"]) == 2,
              str(reply))
        feed.close()
        worker_thread.join(10)

        # Invalid duration evidence must take the old, trusted path once.
        fallback_calls = []

        def invalid_native_tts(req):
            return worker_module._TTSOutput(
                b"WAV", None, "native timing is not monotonic", 0.5)

        def fallback_asr(req, wav):
            fallback_calls.append(req.req_id)
            return [{"w": "hello", "s": 0.0, "e": 0.2}]

        feed, sink, stats, worker_thread = run_worker(
            invalid_native_tts, fallback_asr)
        ask(feed, "fallback", "hello")
        reply = sink.take()
        check("invalid native timing invokes the Whisper fallback exactly once",
              reply["ok"] and fallback_calls == ["fallback"]
              and stats["asr_calls"] == 1
              and reply["alignment_method"] == "whisper-v1",
              f"{reply} / {fallback_calls} / {stats}")
        check("...and keeps the non-sensitive fallback reason for diagnostics",
              reply["alignment_fallback_reason"] == "native timing is not monotonic",
              str(reply))
        feed.close()
        worker_thread.join(10)

        # Pure validation catches the failures that otherwise become silent
        # highlight drift without requiring either model in this test.
        groups = [{"w": "hello,", "s": 0.02, "e": 0.20, "partial": False},
                  {"w": "world", "s": 0.22, "e": 0.43, "partial": False}]
        native, reason, coverage = worker_module._validated_native_words(
            "hello, world", groups, 0.5)
        check("native validation accepts punctuation-attached monotonic words",
              reason is None and coverage == 1.0
              and [w["w"] for w in native] == ["hello,", "world"],
              f"{reason} / {native}")
        broken_groups = [dict(groups[0]), dict(groups[1], s=0.1)]
        native, reason, _ = worker_module._validated_native_words(
            "hello, world", broken_groups, 0.5)
        check("native validation rejects overlapping marks as one whole alignment",
              native is None and reason == "native timing is not monotonic", reason)

        # 3. Cancel between the stages. Kokoro's samples exist; Whisper is the
        #    expensive half and must not run for audio nobody will hear.
        tts_hold = threading_module.Event()
        counted = {"tts": 0, "asr": 0}

        def held_tts(req):
            counted["tts"] += 1
            tts_hold.wait(10)
            return b"WAV"

        def held_asr(req, wav):
            counted["asr"] += 1
            return []

        feed, sink, stats, worker_thread = run_worker(held_tts, held_asr)
        ask(feed, "doomed")
        time.sleep(0.2)                       # it is inside Kokoro now
        feed.send({"op": "cancel", "req_id": "doomed"})
        time.sleep(0.1)
        tts_hold.set()
        reply = sink.take()
        check("a block cancelled during TTS never reaches Whisper",
              reply["ok"] is False and reply["error"]["code"] == "cancelled"
              and counted["asr"] == 0, f"{reply} / {counted}")
        feed.close()
        worker_thread.join(10)

        # 4. Cancel *during* ASR. Nothing can safely interrupt faster-whisper
        #    mid-call, so the call finishes and the complete pair is thrown
        #    away - the one thing that must never happen is publishing it.
        asr_started = threading_module.Event()
        asr_hold = threading_module.Event()

        def slow_asr(req, wav):
            asr_started.set()
            asr_hold.wait(10)
            return [{"w": "hello", "s": 0.0, "e": 0.2}]

        feed, sink, stats, worker_thread = run_worker(lambda r: b"WAV", slow_asr)
        ask(feed, "late")
        check("...and one that is already inside Whisper is not killed there",
              asr_started.wait(10), "ASR never started")
        feed.send({"op": "cancel", "req_id": "late"})
        time.sleep(0.1)
        asr_hold.set()
        reply = sink.take()
        check("a block cancelled during Whisper returns nothing to publish",
              reply["ok"] is False and reply["error"]["code"] == "cancelled"
              and "audio_b64" not in reply, str(reply))
        feed.close()
        worker_thread.join(10)

        # 5. Which stage failed is part of the reply, and one bad block does
        #    not take the worker (or the next block) with it.
        def flaky_tts(req):
            if req.req_id == "bad-tts":
                raise ValueError("kokoro said no")
            return b"WAV"

        def flaky_asr(req, wav):
            if req.req_id == "bad-asr":
                raise ValueError("whisper said no")
            return [{"w": "hello", "s": 0.0, "e": 0.2}]

        feed, sink, stats, worker_thread = run_worker(flaky_tts, flaky_asr)
        ask(feed, "bad-tts")
        first = sink.take()
        ask(feed, "bad-asr")
        second = sink.take()
        ask(feed, "good")
        third = sink.take()
        check("a TTS failure is reported as a TTS failure",
              first["error"]["stage"] == "tts"
              and first["error"]["code"] == "tts_failed", str(first))
        check("an alignment failure discards the audio rather than publishing it alone",
              second["error"]["stage"] == "asr"
              and second["error"]["code"] == "alignment_failed"
              and "audio_b64" not in second, str(second))
        check("...and the block after two failures still succeeds",
              third["ok"] and third["req_id"] == "good", str(third))
        feed.close()
        worker_thread.join(10)

        # 6. The bounds are what keep a 900-page book from becoming gigabytes
        #    of decoded audio. Push far more at it than the parent ever would
        #    and nothing may exceed them - and every request still gets
        #    exactly one terminal reply, even the refused ones.
        def quick_tts(req):
            time.sleep(0.002)
            return b"W"

        def quick_asr(req, wav):
            time.sleep(0.002)
            return []

        feed, sink, stats, worker_thread = run_worker(quick_tts, quick_asr)
        for i in range(100):
            ask(feed, f"r{i}")
        answered, duplicated = {}, []
        for _ in range(100):
            reply = sink.take(timeout=20)
            if reply["req_id"] in answered:
                duplicated.append(reply["req_id"])
            answered[reply["req_id"]] = reply
        check("100 blocks at once never put more than the configured limit in flight",
              stats["live_max"] <= worker_module.Limits().live_requests,
              f"high-water {stats['live_max']}")
        check("...and every one of them is answered exactly once",
              len(answered) == 100 and not duplicated, f"{len(answered)}, dups {duplicated}")
        check("...with the overflow refused rather than queued",
              stats["busy"] > 0 and all(
                  r["ok"] or r["error"]["code"] == "busy"
                  for r in answered.values()), f"busy {stats['busy']}")
        feed.close()
        worker_thread.join(10)

        # 7. Protocol noise must never crash the worker.
        feed, sink, stats, worker_thread = run_worker(quick_tts, quick_asr)
        feed.raw("{ this is not json\n")
        garbled = sink.take()
        check("a malformed request is answered with a null id, not a crash",
              garbled["req_id"] is None and garbled["error"]["code"] == "bad_request",
              str(garbled))
        feed.send({"op": "cancel", "req_id": "never-existed"})
        ask(feed, "survivor")
        reply = sink.take()
        check("...and a cancel for a request that does not exist is ignored",
              reply["ok"] and reply["req_id"] == "survivor", str(reply))
        feed.close()
        worker_thread.join(10)

        # 7b. Shutdown must answer what it is abandoning. The sentinel sorts
        #     ahead of real work on purpose, so a request still queued behind
        #     it would otherwise get no reply at all and its caller would wait
        #     for a process that has already gone.
        gate = threading_module.Event()

        def gated_tts(req):
            gate.wait(20)
            return b"WAV"

        feed, sink, stats, worker_thread = run_worker(gated_tts, lambda r, w: [])
        ask(feed, "in-flight")
        time.sleep(0.2)
        ask(feed, "queued")
        feed.send({"op": "shutdown"})
        time.sleep(0.2)
        gate.set()
        settled = {}
        for _ in range(2):
            reply = sink.take(20)
            settled[reply["req_id"]] = reply
        worker_thread.join(20)
        check("a shutdown answers the request still queued behind it",
              set(settled) == {"in-flight", "queued"}
              and settled["queued"]["error"]["code"] == "cancelled",
              str({k: v.get("error", {}).get("code") for k, v in settled.items()}))
        check("...and the worker loop actually ends",
              not worker_thread.is_alive(), "serve() never returned")

        # 8b. A fatal model failure is not a failed block: it is reported once,
        #     marked fatal, and the process stops rather than answering every
        #     request the same way for the rest of the session. The stage
        #     thread must not simply return, or the request behind it would be
        #     left with nobody to reply to it.
        stopped = threading_module.Event()

        def dead_tts(req):
            raise worker_module.FatalWorkerError("the model would not load")

        feed, sink, stats = _Feed(), _Sink(), {}
        worker_thread = threading_module.Thread(
            target=worker_module.serve, args=(feed, sink),
            kwargs=dict(tts=dead_tts, asr=lambda r, w: [], stats=stats,
                        on_fatal=stopped.set), daemon=True)
        worker_thread.start()
        ask(feed, "first")
        ask(feed, "second")
        after_fatal = {}
        for _ in range(2):
            reply = sink.take(20)
            after_fatal[reply["req_id"]] = reply
        check("a fatal failure still answers every request that was live",
              set(after_fatal) == {"first", "second"}, str(list(after_fatal)))
        check("...marks the reply fatal, so the parent restarts instead of retrying",
              after_fatal["first"]["error"]["fatal"] is True,
              str(after_fatal["first"]["error"]))
        check("...and stops the process only once that reply has been written",
              stopped.wait(10), "on_fatal never ran")
        feed.close()
        worker_thread.join(20)
        check("...leaving no stage thread behind",
              not worker_thread.is_alive(), "serve() never returned")

        # 9b. A duplicate id must not evict the request that legitimately owns
        #     it - that request would then be answered twice, and the parent
        #     would attach the first answer to a block that had not been made.
        held_open = threading_module.Event()

        feed, sink, stats, worker_thread = run_worker(
            lambda r: (held_open.wait(20), b"WAV")[1], lambda r, w: [])
        ask(feed, "a1")
        time.sleep(0.2)
        feed.send({"op": "synth", "req_id": "a1"})      # duplicate, and no text
        rejected = sink.take(20)
        held_open.set()
        genuine = sink.take(20)
        check("a duplicate request id is refused under a null id",
              rejected["req_id"] is None
              and rejected["error"]["code"] == "duplicate_req_id", str(rejected))
        check("...and the request that owns that id still gets its own reply",
              genuine["req_id"] == "a1" and genuine["ok"], str(genuine)[:120])
        feed.close()
        worker_thread.join(20)

        # 10b. A cancel that arrives after Whisper returned but before the
        #      reply is written still publishes nothing. (A cancel sent after
        #      the line is on the wire cannot be caught by anyone, which is
        #      why the coordinator re-checks generation before it caches.)
        release_write = threading_module.Event()

        class _BlockingSink(_Sink):
            def __init__(self):
                super().__init__()
                self.first = True

            def write(self, s):
                if self.first:
                    self.first = False
                    release_write.wait(20)
                super().write(s)

        feed, sink, stats = _Feed(), _BlockingSink(), {}
        worker_thread = threading_module.Thread(
            target=worker_module.serve, args=(feed, sink),
            kwargs=dict(tts=lambda r: b"WAV",
                        asr=lambda r, w: [{"w": "hi", "s": 0.0, "e": 0.1}],
                        stats=stats), daemon=True)
        worker_thread.start()
        ask(feed, "one")
        time.sleep(0.3)                    # "one" is now stuck in the writer
        ask(feed, "two")
        time.sleep(0.3)                    # "two" has finished ASR and is queued
        feed.send({"op": "cancel", "req_id": "two"})
        time.sleep(0.2)
        release_write.set()
        late = {}
        for _ in range(2):
            reply = sink.take(20)
            late[reply["req_id"]] = reply
        check("a cancel landing before the reply is written still publishes nothing",
              late["two"]["ok"] is False
              and late["two"]["error"]["code"] == "cancelled"
              and "audio_b64" not in late["two"], str(late["two"]))
        check("...and the block that was not cancelled is unaffected",
              late["one"]["ok"], str(late["one"])[:120])
        feed.close()
        worker_thread.join(20)

        # 11b. Where models get loaded is a correctness question on Windows,
        #      not a performance one: a thread blocked reading the stdin pipe
        #      stops other threads loading DLLs, so a stage that reached for a
        #      model would hang until the next request arrived. prepare() runs
        #      on the reader thread, before admission, which is the one place
        #      nothing is blocked on the pipe.
        loaded = []

        def watching_prepare(req):
            loaded.append(("prepare", req.req_id))

        def watched_tts(req):
            loaded.append(("tts", req.req_id))
            return b"WAV"

        feed, sink, stats = _Feed(), _Sink(), {}
        worker_thread = threading_module.Thread(
            target=worker_module.serve, args=(feed, sink),
            kwargs=dict(tts=watched_tts, asr=lambda r, w: [],
                        prepare=watching_prepare, stats=stats), daemon=True)
        worker_thread.start()
        ask(feed, "m1")
        sink.take(20)
        check("models are loaded before the request reaches a stage, never inside one",
              loaded == [("prepare", "m1"), ("tts", "m1")], str(loaded))
        feed.close()
        worker_thread.join(20)

        def broken_prepare(req):
            raise RuntimeError("kokoro would not load")

        feed, sink, stats, _unused = None, None, None, None
        feed, sink = _Feed(), _Sink()
        stopped_loading = threading_module.Event()
        worker_thread = threading_module.Thread(
            target=worker_module.serve, args=(feed, sink),
            kwargs=dict(tts=watched_tts, asr=lambda r, w: [],
                        prepare=broken_prepare, on_fatal=stopped_loading.set),
            daemon=True)
        worker_thread.start()
        ask(feed, "m2")
        reply = sink.take(20)
        check("...and a model that will not load is fatal, not one failed block",
              reply["ok"] is False and reply["error"]["fatal"] is True
              and reply["error"]["stage"] == "prepare", str(reply))
        check("...which stops the worker rather than failing every block the same way",
              stopped_loading.wait(10), "on_fatal never ran")
        feed.close()
        worker_thread.join(20)

        # 8. The parent half. A fake worker subprocess exercises the response
        #    dispatcher against exactly the awkward cases a real one produces
        #    only by accident: replies out of order, stray library output on
        #    stdout, an id nobody is waiting on, and dying mid-conversation.
        fake_worker = os.path.join(tempfile.mkdtemp(prefix="talkify_smoke_worker_"),
                                   "fake_worker.py")
        with open(fake_worker, "w", encoding="utf-8") as f:
            f.write(FAKE_WORKER)

        def spawn_fake():
            proc = subprocess.Popen(
                [sys.executable, fake_worker], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                text=True, encoding="utf-8", bufsize=1)
            return proc, None

        client = engines_module.WorkerClient(spawn_fake, name="fake")
        held = client.submit({"text": "hold"})
        freed = client.submit({"text": "release"})
        second_result = freed.result(20)
        first_result = held.result(20)
        import base64 as base64_module
        check("a reply that arrives second is still matched to the request that asked",
              base64_module.b64decode(first_result["audio_b64"]) == b"WAVEhold"
              and base64_module.b64decode(second_result["audio_b64"]) == b"WAVErelease",
              f"{first_result.get('req_id')} / {second_result.get('req_id')}")

        noisy = client.submit({"text": "stray"})
        noisy_result = noisy.result(20)
        check("stray output on the worker's stdout is ignored, not fatal",
              base64_module.b64decode(noisy_result["audio_b64"]) == b"WAVEstray",
              str(noisy_result)[:120])
        check("...and a reply nobody is waiting on is dropped, never handed to another caller",
              any("unknown request" in n for n in client.notes), str(list(client.notes)))

        broken = client.submit({"text": "fail"})
        try:
            broken.result(20)
            raised = None
        except engines_module.SynthError as e:
            raised = e
        check("a per-request failure names its stage",
              raised is not None and raised.code == "alignment_failed"
              and raised.stage == "asr", str(raised))

        epoch_before = client._epoch
        orphans = [client.submit({"text": "hold"}) for _ in range(3)]
        orphans.append(client.submit({"text": "die"}))
        deaths = []
        for h in orphans:
            try:
                h.result(20)
                deaths.append(None)
            except engines_module.WorkerDied as e:
                deaths.append(e)
        check("when the worker dies every request in flight fails, none hang",
              all(isinstance(e, engines_module.WorkerDied) for e in deaths),
              str(deaths))
        check("...all with the same error, so 'the worker died' is distinguishable"
              " from 'this block failed'",
              len({id(e) for e in deaths}) == 1, str(deaths))
        check("...and nothing is left in the pending map",
              client.inflight() == 0, str(client.inflight()))
        revived = client.submit({"text": "after the funeral"})
        check("the next request starts a fresh worker rather than staying dead",
              revived.result(20)["ok"] and client._epoch == epoch_before + 1,
              f"epoch {client._epoch} was {epoch_before}")
        client.shutdown()

        # 9. And the engine on top of it: same public contract as before, so
        #    the routes, the player and the export path cannot tell.
        previous_client = engines_module.KokoroEngine._client
        engines_module.KokoroEngine._client = engines_module.WorkerClient(
            spawn_fake, name="kokoro")
        try:
            kok = engines_module.KokoroEngine()
            audio, mime, words = kok.synth("Hello there world")
            check("KokoroEngine.synth still returns (audio, mime, words) unchanged",
                  mime == "audio/wav" and audio.startswith(b"WAVE")
                  and [w.w for w in words] == ["Hello", "there", "world"],
                  str(words))
            handle = kok.submit("one two")
            audio2, mime2, words2 = handle.result(20)
            check("...and submit() is the same synthesis reached through a handle",
                  mime2 == "audio/wav" and [w.w for w in words2] == ["one", "two"],
                  str(words2))
            _, _, native_words = kok.synth("PSI pressure")
            check("native timings collapse a spoken P S I onto one displayed PSI",
                  [w.w for w in native_words] == ["PSI", "pressure"]
                  and abs(native_words[0].s - 0.0) < 1e-9
                  and abs(native_words[0].e - 0.55) < 1e-9
                  and abs(native_words[1].s - 0.6) < 1e-9,
                  str(native_words))
        finally:
            engines_module.KokoroEngine.shutdown()
            engines_module.KokoroEngine._client = previous_client


        print("durable state: settings and library survive concurrent writers")
        store_dir = tempfile.mkdtemp(prefix="talkify_smoke_store_")
        config_file = os.path.join(store_dir, "config.json")
        store = storage_module.SettingsStore(config_file)
        store.set("default_voice", "af_heart")
        check("a setting round-trips through the store",
              store.get("default_voice") == "af_heart", str(store.all()))
        with open(config_file, "w", encoding="utf-8") as f:
            f.write("{not json at all")
        check("a corrupt config reads as empty rather than raising - a mangled "
              "preference must never cost you the ability to read a book",
              store.all() == {} and store.get("default_voice") is None, str(store.all()))
        store.set("default_voice", "af_bella")

        torn = []
        blew_up = []
        clean_reads = [0]

        def hammer_settings(n):
            try:
                for k in range(25):
                    store.set(f"writer{n}", k)
            except Exception as e:
                blew_up.append(f"{type(e).__name__}: {e}")

        def watch_settings():
            for _ in range(400):
                try:
                    with open(config_file, encoding="utf-8") as f:
                        json.load(f)
                    clean_reads[0] += 1
                except PermissionError:
                    pass          # Windows: the file is mid-replace, not corrupt
                except FileNotFoundError:
                    torn.append("missing")
                except ValueError as e:
                    torn.append(str(e))
                time.sleep(0.001)

        writers = [threading_module.Thread(target=hammer_settings, args=(n,)) for n in range(4)]
        watcher = threading_module.Thread(target=watch_settings)
        watcher.start()
        for t in writers:
            t.start()
        for t in writers:
            t.join(30)
        watcher.join(30)
        check("concurrent writers all complete - no save is lost to a locked file",
              not blew_up, str(blew_up[:2]))
        check("config.json is never observed half-written under concurrent writes",
              not torn and clean_reads[0] > 50, f"torn={torn[:2]} clean={clean_reads[0]}")
        check("...and no writer's updates were lost to another's read-modify-write",
              all(store.get(f"writer{n}") == 24 for n in range(4)), str(store.all()))

        repo = storage_module.LibraryRepository(store_dir)
        try:
            sample_blocks = [f"Block number {i} of a document." for i in range(12)]

            def record(title, opened, position=0):
                return {"title": title, "blocks": sample_blocks, "voice": "af_heart",
                        "position": {"block": position, "word": 0, "within": 0.0},
                        "last_opened": opened, "block_seconds": {}, "source": None,
                        "chapters": []}

            repo.save("aaa", record("First book", "2026-09-01T10:00:00+00:00"))
            repo.save("bbb", record("Second book", "2026-09-03T10:00:00+00:00"))
            names = repo.tables()
            check("the library index has its tables",
                  {"documents", "document_blocks", "playback_progress"} <= set(names),
                  str(names))
            check("saving a document indexes one row per block",
                  repo.block_count("aaa") == len(sample_blocks), str(repo.block_count("aaa")))
            shelf = repo.recent(limit=3)
            check("the recent shelf comes back most-recently-opened first",
                  [r["id"] for r in shelf] == ["bbb", "aaa"], str(shelf))

            saved_torn = []
            save_failed = []
            saved_reads = [0]

            def hammer_library():
                try:
                    for k in range(30):
                        repo.save("aaa", record("First book", "2026-09-01T10:00:00+00:00",
                                                 position=k % len(sample_blocks)))
                except Exception as e:
                    save_failed.append(f"{type(e).__name__}: {e}")

            def watch_library():
                path = repo.path_for("aaa")
                for _ in range(400):
                    try:
                        with open(path, encoding="utf-8") as f:
                            loaded = json.load(f)
                        if len(loaded.get("blocks") or []) != len(sample_blocks):
                            saved_torn.append("short")
                        else:
                            saved_reads[0] += 1
                    except PermissionError:
                        pass      # Windows: mid-replace, not corrupt
                    except FileNotFoundError:
                        saved_torn.append("missing")
                    except ValueError as e:
                        saved_torn.append(str(e))
                    time.sleep(0.001)

            savers = [threading_module.Thread(target=hammer_library) for _ in range(3)]
            lib_watcher = threading_module.Thread(target=watch_library)
            lib_watcher.start()
            for t in savers:
                t.start()
            for t in savers:
                t.join(30)
            lib_watcher.join(30)
            check("concurrent saves of one document all complete",
                  not save_failed, str(save_failed[:2]))
            check("a document's JSON is always whole - concurrent saves leave "
                  "the old copy or the new one, never a truncated one",
                  not saved_torn and saved_reads[0] > 50,
                  f"torn={saved_torn[:2]} clean={saved_reads[0]}")

            repo.delete("aaa")
            check("deleting removes the JSON file", not os.path.exists(repo.path_for("aaa")))
            check("...and cascades the indexed blocks away with it",
                  repo.block_count("aaa") == 0 and [r["id"] for r in repo.recent(3)] == ["bbb"],
                  str(repo.recent(3)))
            check("nothing in any of that made the SQLite mirror fail",
                  repo.sqlite_errors == 0, str(repo.last_sqlite_error))
        finally:
            repo.close()
            shutil.rmtree(store_dir, ignore_errors=True)

        print("chapter detection (pure logic - no PDF or Ollama needed here)")
        import chapters as chapters_module
        check("a clean monotonic chapter list verifies",
              chapters_module._verify([10, 40, 90, 150], 900))
        check("a non-monotonic list is rejected",
              not chapters_module._verify([10, 90, 40, 150], 900))
        check("a list with a page beyond the book's length is rejected",
              not chapters_module._verify([10, 40, 9999], 900))
        check("a single-item list is rejected (nothing to navigate)",
              not chapters_module._verify([10], 900))
        check("chapters crammed onto ~1 page are rejected",
              not chapters_module._verify([10, 10, 11], 900))
        good_json = '[{"title": "Introduction", "page": 1}, {"title": "Methods", "page": 12}]'
        parsed = chapters_module._parse_model_json(good_json)
        check("well-formed model JSON parses", parsed == [
            {"title": "Introduction", "page": 1}, {"title": "Methods", "page": 12}])
        check("prose wrapped around the JSON array is tolerated",
              chapters_module._parse_model_json(f"Sure, here you go:\n{good_json}\nHope that helps!") == parsed)
        check("a model reply with no JSON array at all yields None, not a crash",
              chapters_module._parse_model_json("I cannot do that.") is None)
        with open(tmp_config, "w", encoding="utf-8") as f:
            json.dump({"gemini_api_key": "dummy"}, f)
        check("background chapter detection declines when only a cloud provider is configured (privacy guard)",
              chapters_module.detect("nonexistent.pdf") == [])


        print("figures (enhancement 1: show the figure the text points at)")
        import figures as figures_module

        check("a caption label normalizes to a canonical key",
              figures_module.parse_caption_label("Figure 3.20 Crystal lattice")
              == ("figure:3.20", "Figure 3.20"),
              str(figures_module.parse_caption_label("Figure 3.20 Crystal lattice")))
        check("...and an abbreviated caption reaches the same key, so a "
              "sentence may name either form",
              figures_module.parse_caption_label("Fig. 3.20 Crystal lattice")[0]
              == "figure:3.20")
        check("a table label parses too, case-folded",
              figures_module.parse_caption_label("Table 2A Alloy properties")
              == ("table:2a", "Table 2A"))
        check("ordinary prose is not a caption label",
              figures_module.parse_caption_label("The lattice repeats.") is None)

        fig_items = [figures_module.FigureItem(
            id="f-aaa", kind="figure", label_key="figure:3.20",
            label="Figure 3.20", page=11)]
        bound = figures_module.find_references(
            ["As shown in Figure 3.20, the lattice repeats."], [11], fig_items)
        check("a contextual Figure reference produces word bounds",
              bound and bound[0][0].start_word == 3, str(bound))
        check("...and the bound range ends after the number, exclusive",
              bound and bound[0][0].end_word == 5, str(bound))
        check("...and it points at the opaque figure id, never a page or xref",
              bound and bound[0][0].figure_id == "f-aaa", str(bound))
        check("a bare mention with no pointing verb binds nothing - that is a "
              "caption's shape, not a reference's",
              figures_module.find_references(
                  ["Figure 3.20 shows a crystal lattice."], [11], fig_items) == {})
        check("a reference to a figure that was never extracted binds nothing",
              figures_module.find_references(
                  ["As shown in Figure 9.99, nothing happens."], [11], fig_items) == {})

        dup_items = [figures_module.FigureItem(id="f-one", kind="figure",
                                               label_key="figure:1", label="Figure 1",
                                               page=10),
                     figures_module.FigureItem(id="f-two", kind="figure",
                                               label_key="figure:1", label="Figure 1",
                                               page=30)]
        ambiguous = figures_module.RawRef("figure:1", "Figure 1", 0, 20, 0, 2)
        check("an ambiguous repeated Figure number produces no binding",
              figures_module.resolve_reference(ambiguous, dup_items) is None)
        near = figures_module.RawRef("figure:1", "Figure 1", 0, 12, 0, 2)
        check("...but one that is clearly nearer on the page resolves",
              figures_module.resolve_reference(near, dup_items).id == "f-one")
        unknown_page = figures_module.RawRef("figure:1", "Figure 1", 0, None, 0, 2)
        check("...and a block with no page provenance never guesses between them",
              figures_module.resolve_reference(unknown_page, dup_items) is None)

        # Found in review of Phase 5: the below-branch's flat +MAX_GAP_ABOVE_PT
        # penalty used to let a merely-plausible image far above a caption
        # beat an obviously correct one just a few points below it.
        narrow_caption = {"x0": 100, "y0": 300, "x1": 160, "y1": 310}
        far_above = pymupdf.Rect(100, 100, 200, 200)     # 100pt above: within the allowance, but not the right one
        near_below = pymupdf.Rect(100, 315, 200, 400)    # 5pt below: the actually correct placement
        check("the closer placement wins regardless of direction, not 'above' "
              "unconditionally",
              figures_module._match_placement(narrow_caption, [far_above, near_below])
              == near_below)

        print("...and the whole pipeline, through a real PDF and the real routes")
        import struct as struct_module
        import zlib as zlib_module

        def _png(w, h, rgb):
            def chunk(tag, data):
                c = tag + data
                return (struct_module.pack(">I", len(data)) + c
                        + struct_module.pack(">I", zlib_module.crc32(c) & 0xFFFFFFFF))
            raw = b"".join(b"\x00" + bytes(rgb) * w for _ in range(h))
            return (b"\x89PNG\r\n\x1a\n"
                    + chunk(b"IHDR", struct_module.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                    + chunk(b"IDAT", zlib_module.compress(raw))
                    + chunk(b"IEND", b""))

        fig_pdf = os.path.join(tmp_library, "smoke_figures.pdf")
        fdoc = pymupdf.open()
        fpage = fdoc.new_page()
        fpage.insert_image(pymupdf.Rect(72, 100, 300, 250), stream=_png(64, 48, (40, 90, 200)))
        fpage.insert_text((72, 270), "Figure 1.1 A regular crystal lattice.", fontsize=9)
        fpage.insert_text((72, 320),
                          "The lattice of a metal is regular and repeating throughout.",
                          fontsize=11)
        fpage.insert_text((72, 336),
                          "As shown in Figure 1.1, the pattern repeats across the whole",
                          fontsize=11)
        fpage.insert_text((72, 352),
                          "page and does not vary from one region to the next at all.",
                          fontsize=11)
        fdoc.save(fig_pdf)
        fdoc.close()

        st, body = post("/api/open", {"path": fig_pdf})
        fjob_id = body.get("job")
        fjob = None
        for _ in range(60):
            st, jbody, _ = get(f"/api/open/status?job={fjob_id}")
            fjob = json.loads(jbody)
            if fjob["status"] != "running":
                break
            time.sleep(0.25)
        check("the figure PDF opens", fjob and fjob["status"] == "done", str(fjob)[:160])
        fig_doc_id = fjob["doc"]["id"] if fjob and fjob["status"] == "done" else None
        if fig_doc_id:
            joined = " ".join(fjob["doc"]["blocks"])
            check("the narrative sentence is spoken",
                  "As shown in Figure 1.1" in joined, joined[:160])
            check("...and the caption is not - the figure index must not change "
                  "one word of what gets read aloud",
                  "A regular crystal lattice" not in joined, joined[:160])

            manifest = None
            for _ in range(80):
                st, mbody, _ = get(f"/api/figures?doc={fig_doc_id}")
                manifest = json.loads(mbody)
                if manifest.get("state") != "pending":
                    break
                time.sleep(0.25)
            check("figure indexing settles on an answer rather than polling forever",
                  manifest and manifest.get("state") == "ready", str(manifest)[:200])
            items = (manifest or {}).get("items") or []
            check("exactly one figure was extracted, from page 0",
                  len(items) == 1 and items[0]["page"] == 0, str(items)[:200])
            check("...labelled as the book labels it",
                  items and items[0]["label"] == "Figure 1.1", str(items)[:200])
            check("...and carrying its caption, which the narration dropped",
                  items and "regular crystal lattice" in items[0]["caption"],
                  str(items)[:200])
            refs = (manifest or {}).get("refs") or {}
            ref = (refs.get("0") or [None])[0]
            check("the reference binds the narrating block to that figure",
                  ref and ref["figure_id"] == items[0]["id"], str(refs)[:200])
            check("...at a word range inside the block",
                  ref and 0 <= ref["start_word"] < ref["end_word"]
                  <= len(fjob["doc"]["blocks"][0].split()), str(ref))

            fid = items[0]["id"] if items else "nothing"
            st, png, ctype, headers = get_full(f"/api/figure?doc={fig_doc_id}&id={fid}")
            check("the figure asset is served", st == 200 and len(png) > 200,
                  f"{st}, {len(png)} bytes")
            check("...as a PNG", ctype == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n",
                  f"{ctype}, {png[:8]!r}")
            check("...with an ETag", bool(headers.get("ETag")), str(headers.get("ETag")))
            check("...and cached immutably, since the id is content-addressed",
                  "immutable" in (headers.get("Cache-Control") or ""),
                  str(headers.get("Cache-Control")))

            for bad, why in ((f"?doc={fig_doc_id}&id=f-doesnotexist", "an unknown id"),
                             (f"?doc={fig_doc_id}&id=", "an empty id"),
                             (f"?doc={fig_doc_id}&id=..%2F..%2Fconfig.json",
                              "a traversal-shaped id"),
                             (f"?doc={fig_doc_id}&id=manifest.json", "the manifest itself"),
                             ("?doc=nonexistent&id=f-aaa", "an unknown document")):
                st, fbody, _ = get(f"/api/figure{bad}")
                check(f"{why} 404s from the figure route", st == 404, f"got {st}")
                check(f"...and {why} leaks no filesystem path",
                      b"library" not in fbody and b":\\" not in fbody,
                      fbody[:80].decode("utf-8", "replace"))

            # The manifest is the only thing that makes an id servable, so a
            # PNG sitting in the cache under a name nobody indexed must not be
            # reachable either.
            st, fbody, _ = get(f"/api/figure?doc={fig_doc_id}&id=%2E%2E%5Cmanifest")
            check("a backslash-traversal id 404s too", st == 404, f"got {st}")

        st, body, _ = get(f"/api/figures?doc={doc_id}")
        quiet = json.loads(body)
        check("a document with no PDF behind it reports no figures, quietly - "
              "no error for the player to have to hide",
              quiet.get("state") == "none" and quiet.get("items") == []
              and quiet.get("refs") == {}, str(quiet)[:160])
        st, body, _ = get("/api/figures?doc=nonexistent")
        check("figures for an unknown document 404 cleanly", st == 404, str(body)[:80])

        if pdf_doc_id:
            st, body, _ = get(f"/api/figures?doc={pdf_doc_id}")
            nofig = json.loads(body)
            check("a PDF with no figures still opens and simply binds nothing",
                  nofig.get("state") in ("none", "pending")
                  and not (nofig.get("items") or []), str(nofig)[:160])
            st, body, _ = get(f"/api/block?doc={pdf_doc_id}&i=0")
            check("...and its blocks and audio are entirely unaffected",
                  st == 200 and json.loads(body).get("words"), str(body)[:120])

        print("running-head chapter detection (local, no model, no network)")
        check("a running head parses, number, title and printed page together",
              chapters_module.parse_head("82 • Chapter 3 / Structures of Metals", 101)
              == chapters_module.HeadHit(101, 3, "Structures of Metals", 82),
              str(chapters_module.parse_head("82 • Chapter 3 / Structures of Metals", 101)))
        check("a head with no space after the word parses",
              chapters_module.parse_head("Chapter1 | Structures of Metals", 5).number == 1)
        for sep, sep_name in (("|", "pipe"), ("-", "hyphen"), (":", "colon"),
                              ("–", "en dash")):
            check(f"a {sep_name} separator parses",
                  chapters_module.parse_head(f"Chapter 4 {sep} Imperfections", 9) is not None)
        check("narrative prose containing a chapter reference does NOT parse - "
              "the pattern is anchored at both ends for exactly this reason",
              chapters_module.parse_head(
                  "As we saw in Chapter 3 / results were mixed, the alloy failed.", 5)
              is None)
        check("a head with no title at all does not parse",
              chapters_module.parse_head("Chapter 3", 5) is None)
        check("OCR noise around a real head is tolerated without being 'corrected'",
              chapters_module.parse_head("  82 * Chapter 3 / Structures of Metals  ", 101)
              is not None)

        check("mode rejects an OCR page-number outlier",
              chapters_module.mode_offset(
                  [(101, 82), (115, 96), (129, 110), (200, 3)]) == 19)
        check("...and declines entirely when too few observations agree",
              chapters_module.mode_offset([(101, 82), (115, 90)]) is None)
        check("...and when no head carried a printed page at all",
              chapters_module.mode_offset([(101, None), (115, None)]) is None)

        # A fake 300-page book: heads only on verso pages, front matter blank -
        # exactly the shape measured on real books, and none of it needs Tesseract.
        rh_starts = {1: 30, 2: 80, 3: 150, 4: 220}
        rh_titles = {1: "Introduction", 2: "Atomic Structure",
                     3: "Structures of Metals", 4: "Imperfections"}

        def rh_read(p):
            cur = None
            for n, start in rh_starts.items():
                if p >= start:
                    cur = n
            if cur is None:
                return "front matter"
            if p % 2:
                return "some recto section head"
            return f"{p - 19} • Chapter {cur} / {rh_titles[cur]}"

        prober = chapters_module._Prober(read=rh_read)
        found = chapters_module._detect_running_heads(None, probe=prober, page_count=300)
        check("running heads find every chapter in a fake book",
              [c["number"] for c in found] == [1, 2, 3, 4], str(found)[:200])
        check("...with the titles the heads carried",
              [c["title"] for c in found] == list(rh_titles.values()), str(found)[:200])
        check("...and binary search lands on the first page that really shows "
              "each new head",
              [c["head_page"] for c in found] == [30, 80, 150, 220], str(found)[:200])
        check("...with a bounded, honest margin for the chapter opener",
              all(c["page"] == c["head_page"] - 2 for c in found)
              and all(c["confidence"] == "estimated" for c in found), str(found)[:200])
        check("...and the printed page reported is the one the published start "
              "sits on, not the one the head was read from",
              all(c["printed_page"] == c["page"] - 19 for c in found),
              str(found)[:200])
        check("negative running-head probes are cached - no page is ever read twice",
              prober.calls == len(prober.seen), f"{prober.calls} calls, {len(prober.seen)} pages")
        check("...and the scan is a sample, not a page-by-page walk of the book",
              prober.calls < 150, f"{prober.calls} probes over 300 pages")

        check("no heads anywhere yields nothing at all, not a guess",
              chapters_module._detect_running_heads(
                  None, probe=chapters_module._Prober(read=lambda p: "just body text"),
                  page_count=300) == [])
        check("a cancelled scan publishes nothing",
              chapters_module._detect_running_heads(
                  None, probe=chapters_module._Prober(read=rh_read), page_count=300,
                  cancelled=lambda: True) == [])
        check("a missing source file returns [] rather than raising",
              chapters_module._detect_running_heads("no_such_book.pdf") == [])

        print("a page range means a smaller book, and every pass has to agree")
        # Measured on the real 980-page Callister scan opened as pages 20-100:
        # the running-head pass OCR-probed 428 pages of it, and the machine was
        # unusable while it did. Nothing here needs OCR - the prober is fake -
        # but it pins the bound that was missing.
        probed = []

        def windowed_read(page):
            probed.append(page)
            if page < 200 or page > 280:
                return "outside the selection - should never be read"
            n = 1 if page < 240 else 2
            return f"{page - 19} • Chapter {n} / Windowed Title {n}"

        chapters_module._detect_running_heads(
            None, probe=chapters_module._Prober(read=windowed_read),
            page_count=980, pages=range(200, 281))
        check("the running-head scan never reads a page outside the selection",
              probed and min(probed) >= 200 and max(probed) <= 280,
              f"probed {min(probed) if probed else '-'}..{max(probed) if probed else '-'}")
        check("...and probes a fraction of what the whole file would cost",
              len(set(probed)) < 90, f"{len(set(probed))} pages for an 81-page window")

        whole = []
        chapters_module._detect_running_heads(
            None, probe=chapters_module._Prober(read=lambda p: whole.append(p) or "x"),
            page_count=980)
        check("...while a document that really is the whole file still scans it all",
              whole and max(whole) > 900, f"max page {max(whole) if whole else '-'}")

        window_names = {1: "Atomic Structure", 2: "Crystal Geometry",
                        3: "Imperfections"}

        def edge_read(page):
            if page < 20 or page > 300:
                return "outside the selection"
            n = 1 if page < 120 else (2 if page < 220 else 3)
            return f"{page - 19} • Chapter {n} / {window_names[n]}"

        edge = chapters_module._detect_running_heads(
            None, probe=chapters_module._Prober(read=edge_read),
            page_count=980, pages=range(20, 301))
        check("a chapter whose head sits on the first page of the selection"
              " does not publish a start before it",
              edge and all(c["page"] >= 20 for c in edge)
              and edge[0]["page"] == 20,
              str([(c["number"], c["page"]) for c in edge]))

        with open(os.path.join(here, "talkify.py"), encoding="utf-8") as fh:
            talkify_source = fh.read()
        # The one an auditor caught: a span would have let "pages 20, 980"
        # read the 958 pages nobody selected, which is the original bug again
        # wearing a comma.
        gapped = []
        chapters_module._detect_running_heads(
            None, probe=chapters_module._Prober(
                read=lambda p: gapped.append(p) or "body"),
            page_count=980, pages=[19, 979])
        check("a selection with a gap in it reads only its own pages, never"
              " the hundreds in between",
              set(gapped) <= {19, 979}, str(sorted(set(gapped))[:12]))

        check("a page window is the ends of the selection, and nonsense falls"
              " back to the whole book",
              chapters_module._page_window(980, range(200, 281)) == (200, 280)
              and chapters_module._page_window(980, [900, 5000]) == (900, 900)
              and chapters_module._page_window(980, None) == (0, 979)
              and chapters_module._page_window(980, ["x", "y"]) == (0, 979)
              and chapters_module._page_set(980, [5000, 6000]) is None)

        check("the selection the user typed is what the browser is told,"
              " gaps and all",
              talkify_module._page_runs(talkify_module._selected_pages(
                  "b.pdf", {"pages": "20-100"}, 980)) == [[19, 99]]
              and talkify_module._page_runs(talkify_module._selected_pages(
                  "b.pdf", {"pages": "20,980"}, 980)) == [[19, 19], [979, 979]]
              and talkify_module._selected_pages("b.pdf", {}, 980) is None
              and talkify_module._selected_pages("b.pdf", {"pages": "junk"}, 980)
              is None)
        check("chapter detection prefers the selection over block provenance,"
              " so a document with no provenance still cannot scan the file",
              "def _chapter_scan_pages(" in talkify_source
              and "pages=_chapter_scan_pages(doc)" in talkify_source
              and talkify_source.index("chosen = _doc_selected_pages(doc)")
              < talkify_source.index("return _loaded_page_range_pages(doc)"))

        check("the page view is scoped to those pages, not to the file",
              "function loadedPages()" in app_js
              and "state.pageRuns = j.page_runs || null;" in app_js
              and "nearestLoadedPage(" in app_js)
        check("...it opens on the first page the reader actually has",
              "function currentPdfPage()" in app_js
              and "return loadedPageRange()[0];" in app_js
              and "state.currentPage = null;" in app_js)
        check("...Next steps over a gap instead of walking through it",
              "const p = nearestLoadedPage(state.currentPage + 1, 1);" in app_js
              and "const p = nearestLoadedPage(state.currentPage - 1, -1);" in app_js)
        check("...and prefetches only neighbours the document has",
              "if (next !== null && next > p) {" in app_js
              and "if (previous !== null && previous < p) {" in app_js)

        pages_cache = talkify_module._PdfCache(max_pages=64, max_bytes=4096)
        cache_pdf = os.path.join(tmp_library, "smoke_cache_pages.pdf")
        cdoc = pymupdf.open()
        for n in range(4):
            cdoc.new_page().insert_text((72, 72), f"Cache page {n}")
        cdoc.save(cache_pdf)
        cdoc.close()
        for n in range(4):
            pages_cache.render(cache_pdf, n, 72)
        check("the rendered-page cache is bounded by bytes as well as by count,"
              " because a scanned page is a hundred times a text page",
              pages_cache._bytes <= pages_cache.max_bytes,
              f"{pages_cache._bytes} bytes over a {pages_cache.max_bytes} cap")
        tiny = talkify_module._PdfCache(max_pages=8, max_bytes=1)
        got = tiny.render(cache_pdf, 0, 72)
        check("...and a single page bigger than the whole budget is returned"
              " but not kept, rather than sitting over the cap forever",
              got and len(got) > 1 and not tiny._pages and tiny._bytes == 0,
              f"{len(tiny._pages)} entries, {tiny._bytes} bytes")
        pages_cache.render(cache_pdf, 0, 72)
        pages_cache.close()
        check("...and closing the cache leaves its byte count balanced",
              pages_cache._bytes == 0 and not pages_cache._pages)

        def rh_chapter(n, page, head, obs=3):
            return {"number": n, "title": rh_titles.get(n, "T"), "page": page,
                    "head_page": head, "observations": obs}
        good_seq = [rh_chapter(1, 18, 20), rh_chapter(2, 78, 80)]
        check("a clean running-head sequence verifies",
              chapters_module._verify_running_heads(good_seq, 300))
        check("a non-monotonic sequence is rejected whole",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 80, 82), rh_chapter(2, 18, 20)], 300))
        check("a duplicate chapter number is rejected whole",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20), rh_chapter(1, 78, 80)], 300))
        check("chapters packed onto adjacent pages are rejected",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20), rh_chapter(2, 19, 21)], 300))
        check("a start beyond the book's length is rejected",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20), rh_chapter(2, 780, 782)], 300))
        check("a single chapter is rejected - there is nothing to navigate",
              not chapters_module._verify_running_heads([rh_chapter(1, 18, 20)], 300))
        check("a one-off sighting per chapter is rejected: a running head "
              "repeats, so seeing one once is a misread",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20, obs=1), rh_chapter(2, 78, 80, obs=1)], 300))
        check("...and ONE well-supported chapter cannot carry a single-sighting "
              "one along with it - the list is all-or-nothing",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20, obs=6), rh_chapter(2, 78, 80, obs=1)], 300))
        check("head pages one apart are rejected even though the published "
              "starts were clamped apart",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20), rh_chapter(2, 21, 21)], 300))
        check("contradictory title sightings with < 2 title-consistent observations are rejected",
              not chapters_module._verify_running_heads(
                  [rh_chapter(1, 18, 20, obs=2), rh_chapter(2, 78, 80, obs=1)], 300))

        # Gemini flagged this one in review: a long run of unreadable pages
        # around the search midpoint used to make the boundary search abandon
        # everything below it, so a chapter whose head really started before
        # the plates was reported pages late.
        def plate_run(page):
            if page < 12:
                return "front matter"
            if 14 <= page <= 22:
                return ""                 # full-page plates, no head to read
            return f"{page} - Chapter 2 / Alloys"

        check("a run of headless plate pages does not hide the real boundary "
              "below it",
              chapters_module._first_head_page(
                  chapters_module._Prober(read=plate_run), 2, 5, 23) == 12,
              str(chapters_module._first_head_page(
                  chapters_module._Prober(read=plate_run), 2, 5, 23)))
        check("...and refinement stays bounded rather than walking the book",
              chapters_module.MAX_REFINE_PROBES <= 60,
              str(chapters_module.MAX_REFINE_PROBES))
        check("detect() returns [] rather than raising on a file that is not a PDF",
              chapters_module.detect(os.path.join(here, "smoke.py")) == [])

        print("...and the real detector against a real text-layer PDF (no Tesseract)")
        rh_pdf = os.path.join(tmp_library, "smoke_running_heads.pdf")
        rdoc = pymupdf.open()
        for p in range(120):
            rpage = rdoc.new_page()
            cur = None
            for n, start in {1: 10, 2: 50, 3: 90}.items():
                if p >= start:
                    cur = n
            if cur is not None and p % 2 == 0:
                rpage.insert_text((72, 40),
                                  f"{p - 5} • Chapter {cur} / {rh_titles[cur]}",
                                  fontsize=9)
            else:
                # A folio in the same strip, as a real book prints on every
                # page. It also keeps the text layer non-empty, so this
                # fixture never reaches the OCR branch.
                rpage.insert_text((72, 40), f"{p - 5}", fontsize=9)
            rpage.insert_text((72, 300), f"Body text of page {p + 1} of the book.",
                              fontsize=11)
        rdoc.save(rh_pdf)
        rdoc.close()

        real_heads = chapters_module._detect_running_heads(rh_pdf)
        check("the real running-head reader finds the chapters in a text-layer PDF",
              [c["number"] for c in real_heads] == [1, 2, 3], str(real_heads)[:220])
        check("...at the right pages, read from the page's own text layer",
              [c["head_page"] for c in real_heads] == [10, 50, 90], str(real_heads)[:220])
        check("...and says which method found them",
              all(c["method"] == "running_head" for c in real_heads), str(real_heads)[:220])

        import ai as ai_module
        ask_calls = []
        real_ask = ai_module.ask
        # In-process, ai.py reads the config path from the environment the
        # same way the server does - point it at the throwaway one, which the
        # chapter-detection section above filled with a cloud-only key.
        real_cfg_env = os.environ.get("TALKIFY_CONFIG_PATH")
        os.environ["TALKIFY_CONFIG_PATH"] = tmp_config
        check("the throwaway config really is cloud-only, so the guard below "
              "is testing something",
              ai_module.provider_info().get("provider") == "gemini",
              str(ai_module.provider_info())[:160])

        def counting_ask(*a, **kw):
            ask_calls.append(a)
            return "[]", None

        ai_module.ask = counting_ask
        try:
            via_detect = chapters_module.detect(rh_pdf)
            check("detect() prefers running heads over the contents page",
                  [c["number"] for c in via_detect] == [1, 2, 3], str(via_detect)[:200])
            check("...and never consults a language model to do it - the local "
                  "path is the fast path AND the private one",
                  ask_calls == [], str(ask_calls)[:120])
            headless = chapters_module.detect(pdf_path)
            check("a book with no running heads falls through to the contents "
                  "page, which is still refused for a cloud-only provider",
                  headless == [] and ask_calls == [], str(headless)[:120])
            check("detect() with cancelled=lambda: True returns [] immediately without calling TOC fallback",
                  chapters_module.detect(rh_pdf, cancelled=lambda: True) == [] and ask_calls == [])

            # Found in review of Phase 5: chapter detection for every open document
            # shares one serial worker, so a slow ai.ask() call for a book with
            # no running heads holds up the *next* book's chapters too.
            # _detect_toc couldn't be told the document had already moved on -
            # it had no cancelled parameter at all. Isolate just that one
            # check from the PDF/geometry work that surrounds it.
            real_provider_info = ai_module.provider_info
            real_calibrate = chapters_module._calibrate_offset
            real_find_contents = chapters_module._find_contents_page
            real_contents_text = chapters_module._contents_page_text
            ai_module.provider_info = lambda: {"available": True, "local": True}
            chapters_module._calibrate_offset = (
                lambda doc, samples=14, first_page=0, last_page=None: 0)
            chapters_module._find_contents_page = (
                lambda doc, first_page=0, last_page=None: first_page)
            chapters_module._contents_page_text = lambda doc, idx: "1 Introduction .... 1"
            try:
                check("_detect_toc checks cancellation right before the "
                      "network call, instead of always spending the full "
                      "150s timeout on a document that has already moved on",
                      chapters_module._detect_toc(pdf_path, cancelled=lambda: True) == []
                      and ask_calls == [])
            finally:
                ai_module.provider_info = real_provider_info
                chapters_module._calibrate_offset = real_calibrate
                chapters_module._find_contents_page = real_find_contents
                chapters_module._contents_page_text = real_contents_text
        finally:
            ai_module.ask = real_ask
            if real_cfg_env is None:
                os.environ.pop("TALKIFY_CONFIG_PATH", None)
            else:
                os.environ["TALKIFY_CONFIG_PATH"] = real_cfg_env

        print("audio identity (timings and bytes must name one synthesis)")
        st, bbody, _ = get(f"/api/block?doc={doc_id}&i=0")
        bjson = json.loads(bbody)
        st, abytes, actype, aheaders = get_full(f"/api/audio?doc={doc_id}&i=0")
        check("/api/block reports the voice it actually synthesized in",
              "voice" in bjson, str(bjson)[:120])
        check("/api/block reports an asset key", bool(bjson.get("asset_key")), str(bjson)[:120])
        check("audio and timings identify the same cache asset",
              bjson.get("asset_key") == aheaders.get("X-Talkify-Asset-Key"),
              f"{bjson.get('asset_key')} vs {aheaders.get('X-Talkify-Asset-Key')}")
        check("...and the same voice",
              (bjson.get("voice") or "") == (aheaders.get("X-Talkify-Voice") or ""),
              f"{bjson.get('voice')} vs {aheaders.get('X-Talkify-Voice')}")
        check("audio names the document and block it belongs to",
              aheaders.get("X-Talkify-Doc") == doc_id
              and aheaders.get("X-Talkify-Block") == "0",
              f"{aheaders.get('X-Talkify-Doc')}, {aheaders.get('X-Talkify-Block')}")

        st, gbody, _ = get(f"/api/doc?doc={doc_id}")
        gen = json.loads(gbody).get("generation")
        check("/api/doc reports the session generation", isinstance(gen, int), str(gen))
        alt_voice = next((v["id"] for v in json.loads(get("/api/voices")[1])["voices"]), None)
        if alt_voice is not None and isinstance(gen, int):
            st, vbody = post("/api/voice", {"doc": doc_id, "voice": alt_voice,
                                            "expected_generation": gen + 99})
            check("stale voice changes cannot overwrite a newer request",
                  st == 409, f"got {st}: {str(vbody)[:120]}")
            check("...and the refusal says what the generation actually is",
                  vbody.get("generation") == gen, str(vbody)[:120])
            st, vbody = post("/api/voice", {"doc": doc_id, "voice": alt_voice,
                                            "expected_generation": gen})
            check("...while a current one is accepted and reports the new number",
                  st == 200 and isinstance(vbody.get("generation"), int), str(vbody)[:120])

        from session import PlaybackSession
        import threading
        s_lock = threading.Lock()
        s_sess = PlaybackSession("test_doc", s_lock, voice="v1")
        s_ok, s_gen1 = s_sess.check_and_set_voice("v2", expected_generation=0)
        check("session check_and_set_voice succeeds atomically with matching expected_generation",
              s_ok and s_gen1 == 1 and s_sess.voice == "v2")
        s_bad, s_gen2 = s_sess.check_and_set_voice("v3", expected_generation=0)
        check("...and fails atomically when expected_generation is stale",
              not s_bad and s_gen2 == 1 and s_sess.voice == "v2")

        print("player wiring (two audio slots and a figure dock - invisible without a browser)")
        check("the player really has two audio slots, not one element renamed",
              "slots = {" in app_js and "standby()" in app_js and "activeSlot" in app_js)
        check("the next block is primed while the current one is still playing",
              "primeStandby" in app_js and app_js.count("primeStandby(") >= 4,
              str(app_js.count("primeStandby(")))
        check("the boundary hands over to the standby slot rather than "
              "starting a fetch at 'ended'",
              "function handoff(" in app_js and "onSlotEnded" in app_js)
        check("a standby that is not ready enters a waiting state instead of "
              "replaying whatever the slot still holds",
              "waiting for ${next + 1}" in app_js)
        check("every slot load carries an epoch, so a stale fetch cannot arm a slot",
              "playbackEpoch" in app_js and "state.playbackEpoch++" in app_js)
        check("slots are released before their object URLs are revoked",
              "releaseSlot" in app_js and "releaseAllSlots" in app_js)
        check("keysAgree is enforced before arming playback slot",
              "keysAgree(i)" in app_js and "delete state.words[i]" in app_js)

        print("gapless transitions (Web Audio) - the part a browser cannot show you")
        # The whole point of the pass: block N+1 begins at the context time
        # block N runs out, handed to the audio thread rather than started from
        # a main-thread event. If waScheduleNext stops taking its start time
        # from the sounding block's own endAt, the join silently goes back to
        # being an event-loop turn wide and nothing else here would notice.
        check("the next block is scheduled at the exact context time this one ends",
              "function waScheduleNext(" in app_js
              and "waStart(i, buffer, head.endAt, 0)" in app_js
              and "head.i !== i - 1" in app_js)
        check("...and a block's end is its own start plus its own duration",
              "endAt: startAt + Math.max(0, buffer.duration - off) / wa.rate" in app_js)
        check("...and a boundary less than one ramp away is refused rather than "
              "scheduled into the past, where the fade-in would be skipped",
              "if (head.endAt <= audioCtx.currentTime + WA_RAMP) return false;"
              in app_js)
        check("...and the barrier and the asset key are checked there too, not "
              "only at the caller",
              "if (beyondVoiceBarrier(i) || !keysAgree(i)) return false;" in app_js)
        # A ramp at each seam, and blocks that still abut exactly - a crossfade
        # built by overlapping them would shift every word timing after the
        # first one.
        check("the seam is a micro-ramp, not a cut",
              "const WA_RAMP = 0.006;" in app_js
              and "linearRampToValueAtTime(1, e.startAt + WA_RAMP)" in app_js
              and "linearRampToValueAtTime(0, e.endAt)" in app_js)
        check("...and stopping a node ramps it down too, so a seek does not click",
              "function waStopEntry(" in app_js
              and "linearRampToValueAtTime(0, t + WA_RAMP)" in app_js)
        # The context has to be built inside a real gesture or it stays
        # suspended, and a suspended context plays silence while reporting
        # nothing at all - the worst failure shape available here.
        check("the AudioContext is armed from a user gesture, not from playback",
              "function getAudioContext(" in app_js
              and '"pointerdown", "keydown", "touchend"' in app_js
              and "capture: true" in app_js)
        check("...and a context that will not leave 'suspended' is unusable",
              "async function waReady(" in app_js
              and 'audioCtx.state === "running"' in app_js)
        # Degrade to nothing, per the design rule: each of these says no
        # by returning false or null, and no reading path asks whether it worked.
        check("no context, or a clip that will not decode, falls back to the slots",
              "if (!started) {" in app_js
              and "armSlot(active(), i, audioUrl, state.voice, epoch);" in app_js
              and "if (buffer) started = waBegin(i, buffer, seek);" in app_js)
        check("...and the two paths can never sound at once",
              "  waStopAll();\n  releaseSlot(slots.A);" in app_js
              and "releaseSlot(slots.A);\n  releaseSlot(slots.B);\n  wa.queue = [e];" in app_js)
        check("...with the graph stopped wherever an element used to just be paused",
              app_js.count("waPause();") >= 2
              and app_js.count("waStopAll();") >= 6,
              str(app_js.count("waStopAll();")))
        # Timings and audio still come from the one synth() call, which
        # is non-negotiable: the decode reads the blob the fetch already
        # produced, never a second /api/audio another synthesis could serve.
        check("the decode reads the blob the fetch already made, not a second "
              "request the server could answer from another synthesis",
              "function ensureBuffer(" in app_js
              and "const url = state.audioUrls[i];" in app_js
              and "await (await fetch(url)).arrayBuffer()" in app_js
              and "/api/audio" not in app_js.split("function ensureBuffer(")[1]
                                            .split("\nfunction ")[0])
        check("...and a decoded buffer is voice-tagged like every other half-block",
              "bufferVoices" in app_js and "state.bufferVoices[i] === v" in app_js)
        # An epoch bump or a voice switch has to reach into the audio thread: a
        # scheduled node plays at a known time whether or not anything on the
        # main thread is still watching for it.
        check("a voice switch stops nodes already handed to the audio thread",
              "function waDropAfter(" in app_js
              and "waDropAfter(afterIndex);" in app_js
              and app_js.index("waDropAfter(afterIndex);")
                  < app_js.index("abortStale(afterIndex);"))
        check("...and flushes the decoded buffers past the barrier with them",
              "delete state.audioBuffers[k];" in app_js
              and "delete state.bufferVoices[k];" in app_js)
        check("...and a deliberate start stops whatever was scheduled behind it",
              app_js.index("async function playBlock(")
              < app_js.index("  waStopAll();\n  clearHighlight();"))
        # Decoded audio is PCM in RAM, several times the size of the WAV it came
        # from. Unbounded, a long book is gigabytes of it.
        check("the decoded cache is a window around the reading position, not "
              "the whole book",
              "function waTrimBuffers(" in app_js
              and "WA_KEEP_BACK" in app_js and "WA_KEEP_AHEAD" in app_js
              and app_js.count("waTrimBuffers(") >= 3,
              str(app_js.count("waTrimBuffers(")))
        check("...and it never drops a node the audio thread still holds",
              "if (waScheduled(i) || (wa.paused && wa.pausedAt === i)) continue;"
              in app_js)
        # The highlight and the saved position come off the context clock when
        # the graph is driving, because the elements hold nothing then -
        # activeEl().currentTime would be a flat zero all the way through.
        check("the highlighter reads the context clock when the graph is driving",
              "function waTick(" in app_js
              and "(audioCtx.currentTime - e.startAt) * wa.rate + e.offset" in app_js)
        check("...and so does the position the heartbeat saves",
              "const within = currentWithin();" in app_js
              and "function currentWithin(" in app_js)
        check("...including at the instant a block ends, when no node is left "
              "to ask and a zero would replay the paragraph on reopen",
              "wa.lastPos" in app_js
              and "if (wa.lastPos.i === state.i && !active().url) "
                  "return wa.lastPos.within;" in app_js)
        check("...and the graph's boundary goes where the slots' 'ended' goes, "
              "so the scanned-book case cannot be forgotten by one of them",
              "function advancePastEnd(" in app_js
              and app_js.count("advancePastEnd()") >= 3,
              str(app_js.count("advancePastEnd()")))
        check("a speed change re-anchors the chain instead of leaving every "
              "scheduled end at the old rate",
              "function waRetime(" in app_js and "waRetime(v);" in app_js
              and "cur.startAt = t - spoken / v;" in app_js)

        print("...and the holes an adversarial review found in it, pinned")
        # Each of these was a real bug, verified by a regression test that fails
        # against the code as it stood before the fix.
        #
        # The one this pass introduced: the two new awaits in playBlock sit
        # *after* the audio URL has been validated, which the old code's single
        # synchronous run never did. A document opened, or a voice confirmed,
        # while a decode was outstanding armed and played the old one.
        check("playBlock guards its new awaits on more than playToken - an "
              "epoch, a document or a voice moving under them arms nothing",
              "const stale = () => token !== playToken "
              "|| epoch !== state.playbackEpoch" in app_js
              and "|| doc !== state.docId || v !== state.voice" in app_js
              and "if (stale()) { if (started) waStopAll(); return; }" in app_js)
        check("...and they are wrapped, so 'the graph threw' reaches the "
              "element fallback exactly like 'the graph declined'",
              "} catch { started = false; }" in app_js)
        # A suspended context decodes and accepts start() without complaint
        # while currentTime never advances. Believing it is the one failure with
        # no way back: both elements have been emptied by then.
        check("a suspended AudioContext is never treated as playable",
              "const waLive = () =>" in app_js
              and 'audioCtx.state === "running";' in app_js
              and "&& state.i >= 0 && waLive()" in app_js
              and "const buffer = waLive() ? await ensureBuffer(state.i) : null;"
                  in app_js)
        check("...and Play on one reaches playBlock at the offset Pause kept, "
              "rather than silence with 'Pause' showing",
              "} else if (wa.paused && wa.pausedAt === state.i && state.i >= 0) {"
              in app_js
              and "playBlock(state.i, wa.offset, { explicit: true });" in app_js)
        # waDropAfter emptying the queue left wa.on true with no node left to
        # report ending: silent, with nothing to restart it.
        check("waDropAfter cannot stop the block that is sounding",
              "const cur = waSounding();" in app_js
              and "if (e.i > afterIndex && e !== cur) waStopEntry(e); "
                  "else keep.push(e);" in app_js)
        # waBegin schedules one ramp into the future, so for 6ms there is a node
        # that has not started. Pause read it as zero; a speed change retimed
        # the clock but not the node.
        check("a node still in its lead-in is not read as zero seconds in",
              "const cur = waSounding() || wa.queue[0] || null;" in app_js
              and "(cur.startAt > audioCtx.currentTime ? cur.offset : waWithin(cur))"
                  in app_js)
        check("...and a speed change retimes it rather than only the clock the "
              "highlighter reads",
              "const running = cur.startAt <= t;" in app_js
              and "if (running) cur.startAt = t - spoken / v;" in app_js
              and "else { waHold(cur.gain.gain, t); waEnvelope(cur); }" in app_js)
        # The same block can be synthesized twice in the same voice - the
        # keysAgree retry path does exactly that - so the voice alone cannot
        # decide whether a decoded buffer is still the audio the timings match.
        check("a decoded buffer carries the asset key it was decoded from",
              "bufferKeys: {}," in app_js
              and "state.bufferKeys[i] === state.audioKeys[i]" in app_js
              and "state.bufferKeys[i] = state.audioKeys[i];" in app_js)
        check("...and the keysAgree retry drops it along with the audio it is "
              "retrying",
              "delete state.audioUrls[i];\n    delete state.audioBuffers[i];"
              in app_js)
        # A gain frozen part-way down a fade-out must not snap back to full.
        check("retiming at the bottom of a fade rides it down instead of "
              "snapping back to full gain",
              "g.linearRampToValueAtTime(0, Math.max(e.endAt, t + 0.001));"
              in app_js)
        # A node stopped before it ever started is not guaranteed to report
        # ending, and it holds megabytes of PCM attached to the graph.
        check("a stopped node is disconnected on a timer as well as on 'ended'",
              "setTimeout(drop, 250);" in app_js)
        check("...and a key mismatch retries instead of leaving the player "
              "stuck on 'preparing' forever (review finding)",
              "if (token === playToken) playBlock(i, seek);" in app_js)
        check("the figure dock is created lazily, not left empty in the page",
              "ensureFigureDock" in app_js and "figureDock" not in index_html)
        check("...and it has a dismiss control",
              "figure-dock-close" in app_js)
        check("...and a stale image cannot appear after the reader has moved on",
              "figureGen" in app_js)
        check("updateFigureForWord displays the latest reached reference",
              "reached[reached.length - 1]" in app_js)
        check("stale poll callbacks do not cancel active poll timers",
              "if (docId !== state.docId) return;" in app_js)
        with open(os.path.join(here, "web", "style.css"), encoding="utf-8") as fh:
            app_css = fh.read()
        check("the dock is positioned over the page, so revealing a figure "
              "shifts no line of the text being read",
              ".figure-dock {" in app_css and "position: fixed" in app_css)

        print("a growing document in the browser (append-only, invisible without one)")
        check("paragraphs are built by one function, so an appended block matches"
              " the ones rendered at open",
              "function makeBlock(" in app_js
              and app_js.count("makeBlock(") >= 3, str(app_js.count("makeBlock(")))
        check("the suffix is appended, never spliced or re-rendered",
              "function appendBlocks(" in app_js
              and "reader.appendChild(frag)" in app_js
              and "state.blocks.concat(blocks)" in app_js)
        check("...and refused unless it lines up exactly with what is on screen",
              "if (from !== state.blocks.length) return 0;" in app_js
              and "if (docId !== state.docId) return 0;" in app_js)
        check("...leaving the search index to be rebuilt rather than left short",
              "state.lowerBlocks = null;" in app_js
              and app_js.count("state.lowerBlocks = null;") >= 2)
        check("the poll asks for a suffix by position, so a retried poll cannot"
              " lose an update",
              "/api/doc/blocks?doc=${docId}&from=${state.blocks.length}" in app_js)
        check("...and stops when the document changes, or the server says it is done",
              "if (docId !== state.docId) return;" in app_js
              and 'state.ingestion.state !== "running"' in app_js)
        check("reaching the last block that exists waits for the next page"
              " instead of stopping",
              "waitForMoreText" in app_js
              and "waiting for the next scanned page" in app_js)
        check("...and a reader who paused in the meantime is not restarted",
              "if (state.playing) playBlock(next);" in app_js)
        check("scanning progress has its own line, so it cannot overwrite"
              " playback status",
              'id="ingestNote"' in index_html and "playerSub" in index_html
              and 'aria-live="polite"' in index_html)
        check("...and is announced politely rather than on every tick",
              index_html.count('aria-live="polite"') >= 1)
        check("export is disabled while a book is still being scanned",
              'id="exportIngesting"' in index_html
              and '$("exportStartBtn").disabled = scanning;' in app_js)
        check("...and a 409 from the server does not offer an immediate retry",
              'j.code === "ingestion_running"' in app_js)
        check("practice cards are applied again as more of the book arrives,"
              " without doubling up on paragraphs that already have one",
              "function applyPractice(" in app_js
              and "state.practiceMarked.has(item.block)" in app_js
              and "applyPractice(reply.practice)" in app_js)
        check("the scanning line is styled quietly and respects reduced motion",
              ".ingest-note" in style_css
              and "prefers-reduced-motion" in style_css)

        print("pronunciation overrides (spoken vs displayed text)")
        src_text = "The gauge read 45 PSI on the dial."
        check("PSI gets spelled out for speech",
              "P S I" in spoken_form(src_text), spoken_form(src_text))
        check("...but the source tokens - what's displayed and highlighted - keep it as one word",
              "PSI" in words_of(src_text), str(words_of(src_text)))
        check("lowercase 'psi' (the Greek letter) is left alone",
              spoken_form("the psi function") == "the psi function")

        print("acoustic de-noising: what is spoken, and what is still displayed")
        import engines as engines_mod
        from engines import (load_pronunciations, make_spoken_plan, plan_words,
                             refresh_speech_settings, silent_block)

        def plan_of(text, table=None):
            return make_spoken_plan(text, pronunciations=table)

        def timed(plan, *, native=True, seconds=2.0):
            """Every displayed token, timed the way a real reply would time
            it: marks describe the spoken text, never the displayed text."""
            heard = [Word(w, i * 0.2, i * 0.2 + 0.15)
                     for i, w in enumerate(plan.spoken_words)]
            indices = list(range(len(heard))) if native else None
            return plan_words(plan, heard, spoken_indices=indices,
                              audio_seconds=seconds)

        def well_formed(words, plan, seconds=2.0):
            if len(words) != len(plan.source_words):
                return False, f"{len(words)} timings for {len(plan.source_words)} tokens"
            for i, w in enumerate(words):
                if w.w != plan.source_words[i]:
                    return False, f"token {i} is {w.w!r}"
                if not (w.s == w.s and w.e == w.e):        # NaN
                    return False, f"token {i} is not finite"
                if w.e <= w.s:
                    return False, f"token {i} has no duration"
                if w.s < 0 or w.e > seconds + 1e-6:
                    return False, f"token {i} is outside the audio"
                if i and w.s + 1e-9 < words[i - 1].e:
                    return False, f"token {i} goes backwards"
            return True, ""

        cite = plan_of("Strength rises (Smith et al., 2020), rapidly.")
        check("an author-date citation is not spoken",
              cite.spoken_text == "Strength rises rapidly.", cite.spoken_text)
        check("...and the words around it keep their sentence, comma and all",
              cite.source_words[-1] == "rapidly." and "," not in cite.spoken_text,
              cite.spoken_text)
        check("...while every displayed token, citation included, is still there",
              len(cite.source_words) == 7 and cite.silent_sources == frozenset({2, 3, 4, 5}),
              f"{cite.source_words} {sorted(cite.silent_sources)}")
        okay, why = well_formed(timed(cite), cite)
        check("...and every one of them gets a finite, monotonic, in-audio timing",
              okay, why)

        numeric = plan_of("See [12-14].")
        check("a bracketed reference is dropped but its sentence stop is kept",
              numeric.spoken_text == "See.", numeric.spoken_text)
        multi = plan_of("Compare [12, 14; 18] and [12–14] closely.")
        check("...including comma, semicolon and en-dash number lists",
              multi.spoken_text == "Compare and closely.", multi.spoken_text)

        semis = plan_of("Both (Smith, 2020; Jones and Lee, 2021) agree.")
        check("semicolon-separated author-year references go together",
              semis.spoken_text == "Both agree.", semis.spoken_text)
        pages = plan_of("As in (Callister & Rethwisch, 2018, pp. 45–48) exactly.")
        check("...and so do page locators inside them",
              pages.spoken_text == "As in exactly.", pages.spoken_text)

        apollo = plan_of("(Apollo 11 launched in 1969) is a fact.")
        check("an ordinary parenthetical with a year in it is still spoken",
              apollo.spoken_text == " ".join(apollo.source_words),
              apollo.spoken_text)
        check("...and nothing in it is marked silent",
              not apollo.silent_sources, sorted(apollo.silent_sources))
        maths = plan_of("Let f(x, 2020) be (a + b) times c.")
        check("maths parentheses are not citations",
              not maths.silent_sources, maths.spoken_text)
        crossref = plan_of("Compare (See 2020) here.")
        check("a capitalized cross-reference is not mistaken for an author",
              not crossref.silent_sources, crossref.spoken_text)
        room = plan_of("Meet at (Bay 4200) tomorrow.")
        check("...nor is a number that is not a plausible publication year",
              not room.silent_sources, room.spoken_text)

        url = plan_of("Visit https://example.com/a_b. Then stop.")
        check("a URL is spoken as one word",
              url.spoken_text == "Visit link. Then stop.", url.spoken_text)
        check("...mapped to the one displayed token it replaced",
              url.source_words[1] == "https://example.com/a_b."
              and url.source_for_spoken == (0, 1, 2, 3),
              f"{url.source_words} {url.source_for_spoken}")
        okay, why = well_formed(timed(url), url)
        check("...and the displayed URL still gets its own timing", okay, why)
        bare = plan_of("www.example.com")
        check("a block that is only a URL still speaks", bare.spoken_text == "link",
              bare.spoken_text)

        empty = plan_of("(Smith et al., 2020)")
        check("a citation-only block has nothing to say",
              empty.spoken_text == "" and not empty.spoken_words, empty.spoken_text)
        wav, mime, words = silent_block(empty)
        check("...so it answers with brief silence instead of waking a model",
              mime == "audio/wav" and len(wav) > 44
              and len(words) == len(empty.source_words), f"{mime} {len(words)}")
        okay, why = well_formed(words, empty, engines_mod.wav_seconds(wav))
        check("...and still times every displayed token, so the block advances",
              okay, why)

        table = load_pronunciations({"Euler": "Oiler", "SQL": "S Q L",
                                     "navier": "Nah-vee-ay"})
        dictated = plan_of("Euler and SQL and Navier and euler.", table)
        check("a dictionary entry is spoken as its replacement",
              dictated.spoken_text == "Oiler and S Q L and Nah-vee-ay and euler.",
              dictated.spoken_text)
        check("...a multi-word value still collapses to one displayed word",
              dictated.source_for_spoken == (0, 1, 2, 2, 2, 3, 4, 5, 6),
              str(dictated.source_for_spoken))
        okay, why = well_formed(timed(dictated), dictated)
        check("...and the expansion's timings collapse onto that word", okay, why)
        check("a cased entry stays case-sensitive; an all-lowercase one does not",
              dictated.spoken_words[-1] == "euler."
              and "Nah-vee-ay" in dictated.spoken_text, dictated.spoken_text)
        check("the dictionary overrides the built-in spell-out",
              plan_of("45 PSI", load_pronunciations(
                  {"PSI": "pounds per square inch"})).spoken_text
              == "45 pounds per square inch")
        check("...and SPELL_OUT still applies when it says nothing",
              plan_of("45 PSI").spoken_text == "45 P S I")

        for bad in (None, [], "nope", 7, {"": "x"}, {"k": ""}, {"a": 5},
                    {5: "a"}, {"a\nb": "c"}, {"a": "b\x00c"},
                    {"k" * 200: "v"}, {"k": "v" * 400}):
            if load_pronunciations(bad).entries:
                check("a malformed pronunciation map is invisible", False, repr(bad))
                break
        else:
            check("a malformed pronunciation map is invisible", True)
        refresh_speech_settings({"Euler": "Oiler"})
        first = engines_mod.pronunciation_table()
        refresh_speech_settings({"Euler": "Oiler"})
        check("an unchanged config is not recompiled",
              engines_mod.pronunciation_table() is first)
        refresh_speech_settings("not a map")
        check("...and an unreadable one reverts to ordinary narration",
              not engines_mod.pronunciation_table().entries
              and plan_of("Euler").spoken_text == "Euler")

        whispered = plan_of("The gauge read 45 PSI on the dial.")
        heard = [Word(w, i * 0.3, i * 0.3 + 0.25)
                 for i, w in enumerate("the gauge read 45 P S I on the dial".split())]
        out = plan_words(whispered, heard, audio_seconds=3.5)
        okay, why = well_formed(out, whispered, 3.5)
        check("a Whisper transcript is aligned against the spoken text, never"
              " the displayed text", okay, why)
        check("...so the spelled-out word keeps its single highlight",
              len(out) == 8 and out[4].w == "PSI"
              and out[4].e - out[4].s > 0.5, str(out[4]))
        check("no marks at all still times every displayed token",
              well_formed(plan_words(whispered, [], audio_seconds=3.5),
                          whispered, 3.5)[0])

        print("TKF1: the framed worker protocol")
        import io as io_mod
        import struct as struct_mod
        import worker_protocol as wp

        def frames_of_stream(stream):
            out = []
            while True:
                try:
                    out.append(wp.read_frame(stream))
                except wp.StreamClosed:
                    return out

        def frames_of(data):
            return frames_of_stream(io_mod.BytesIO(data))

        check("the header is the documented 20-byte struct",
              wp.HEADER.format == ">4sBBHIQ" and wp.HEADER.size == 20,
              wp.HEADER.format)
        check("the caps are 1 MiB of metadata and 32 MiB of body",
              wp.MAX_METADATA == 1 << 20 and wp.MAX_BODY == 32 << 20)

        buf = io_mod.BytesIO()
        writer = wp.FrameWriter(buf)
        # Everything a WAV can legally contain and a naive reader would choke
        # on: a NUL, a newline, and the frame magic itself.
        nasty = b"RIFF" + bytes([0]) + b"\n" + wp.MAGIC + b"\r\n" + bytes(range(256))
        writer.write(wp.RESULT, {"req_id": "r1", "sample_rate": 24000,
                                 "mime_type": "audio/wav",
                                 "words": [{"w": "café", "s": 0.0, "e": 0.4}],
                                 "alignment_method": "kokoro-pytorch-duration-v1"},
                     nasty)
        writer.write(wp.CANCEL, {"req_id": "r1"})
        writer.write(wp.SHUTDOWN, {})
        wire = buf.getvalue()
        got = frames_of(wire)
        check("a frame round-trips with its body byte for byte",
              len(got) == 3 and got[0].body == nasty, str(len(got)))
        check("...and a body containing NUL, CRLF and TKF1 is not resynced onto",
              got[0].req_id == "r1" and got[1].kind == wp.CANCEL
              and got[2].kind == wp.SHUTDOWN)
        check("...and non-ASCII metadata survives as text, not escapes",
              got[0].meta["words"][0]["w"] == "café")
        check("the audio is the body, never a base64 field in the metadata",
              "audio_b64" not in got[0].meta and nasty not in
              json.dumps(got[0].meta).encode("utf-8"))

        class _Drip:
            """A pipe that never gives you what you asked for in one go."""

            def __init__(self, data, chunk):
                self.data, self.chunk, self.pos = data, chunk, 0

            def read(self, n):
                take = min(n, self.chunk, len(self.data) - self.pos)
                out = self.data[self.pos:self.pos + take]
                self.pos += take
                return out

        check("a frame split one byte at a time still reads whole",
              [f.body for f in frames_of_stream(_Drip(wire, 1))] == [nasty, b"", b""])
        check("...and so does one delivered all at once",
              [f.body for f in frames_of_stream(_Drip(wire, len(wire)))]
              == [nasty, b"", b""])

        class _Trickle:
            """A pipe that accepts a few bytes per write, like a full one."""

            def __init__(self, limit):
                self.buf = bytearray()
                self.limit = limit

            def write(self, view):
                take = min(len(view), self.limit)
                self.buf += bytes(view[:take])
                return take

            def flush(self):
                pass

        trickle = _Trickle(7)
        wp.FrameWriter(trickle).write(
            wp.RESULT, {"req_id": "t", "sample_rate": 24000,
                        "mime_type": "audio/wav", "words": []}, nasty)
        check("a short write is looped until every byte is out",
              frames_of(bytes(trickle.buf))[0].body == nasty)

        def refuses(data, label):
            try:
                frames_of(data)
            except wp.ProtocolError:
                return True
            return False

        head = bytearray(wire[:wp.HEADER_SIZE])
        bad_magic = bytes(b"XXXX") + wire[4:]
        bad_version = bytes(head[:4] + bytearray([9]) + head[5:]) + wire[wp.HEADER_SIZE:]
        bad_kind = bytes(head[:5] + bytearray([0x7f]) + head[6:]) + wire[wp.HEADER_SIZE:]
        bad_flags = bytes(head[:6] + bytearray([0, 1]) + head[8:]) + wire[wp.HEADER_SIZE:]
        over_meta = wp.HEADER.pack(wp.MAGIC, 1, wp.RESULT, 0, wp.MAX_METADATA + 1, 0)
        over_body = wp.HEADER.pack(wp.MAGIC, 1, wp.RESULT, 0, 2, wp.MAX_BODY + 1) + b"{}"
        body_on_control = wp.HEADER.pack(wp.MAGIC, 1, wp.CANCEL, 0, 15, 3) \
            + b'{"req_id":"a"}' + b" " + b"abc"
        not_json = wp.HEADER.pack(wp.MAGIC, 1, wp.SHUTDOWN, 0, 3, 0) + b"{{{"
        not_object = wp.HEADER.pack(wp.MAGIC, 1, wp.SHUTDOWN, 0, 2, 0) + b"[]"
        bad_utf8 = wp.HEADER.pack(wp.MAGIC, 1, wp.SHUTDOWN, 0, 3, 0) + b'"\xff"'
        dupes = wp.HEADER.pack(wp.MAGIC, 1, wp.SHUTDOWN, 0, 21, 0) \
            + b'{"a": 1, "a": 2}     '
        no_id = wp.HEADER.pack(wp.MAGIC, 1, wp.RESULT, 0, 2, 0) + b"{}"
        thin_result = wp.HEADER.pack(wp.MAGIC, 1, wp.RESULT, 0, 15, 0) \
            + b'{"req_id":"a"} '
        bad_ack = wp.HEADER.pack(wp.MAGIC, 1, wp.HELLO_ACK, 0, 15, 0) \
            + b'{"version":999}'
        for data, label in ((bad_magic, "bad magic"), (bad_version, "bad version"),
                            (bad_kind, "unknown kind"), (bad_flags, "unknown flags"),
                            (over_meta, "oversized metadata"),
                            (over_body, "oversized body"),
                            (body_on_control, "a body on a control frame"),
                            (not_json, "unparseable metadata"),
                            (not_object, "metadata that is not an object"),
                            (bad_utf8, "metadata that is not UTF-8"),
                            (no_id, "a result with no request id"),
                            (thin_result, "a result with no audio"),
                            (bad_ack, "an acknowledgement for another version")):
            if not refuses(data, label):
                check("every malformed frame is refused, never resynced", False, label)
                break
        else:
            check("every malformed frame is refused, never resynced", True)
        check("duplicate metadata keys are refused rather than last-one-wins",
              refuses(dupes, "duplicate keys"))
        check("a structurally valid result with no audio in it is still refused",
              refuses(thin_result, "empty result"))
        check("EOF between frames is clean; EOF inside one is not",
              frames_of(b"") == []
              and refuses(wire[:wp.HEADER_SIZE + 4], "torn frame"))

        writers_buf = io_mod.BytesIO()
        shared = wp.FrameWriter(writers_buf)
        payloads = {}

        def spam(tag):
            for n in range(40):
                rid = f"{tag}-{n}"
                body = (tag * 3).encode() + bytes([n]) * 500
                payloads[rid] = body
                shared.write(wp.RESULT, {"req_id": rid, "sample_rate": 24000,
                                         "mime_type": "audio/wav",
                                         "words": []}, body)

        spammers = [threading_module.Thread(target=spam, args=(t,))
                    for t in ("a", "b", "c")]
        for t in spammers:
            t.start()
        for t in spammers:
            t.join(20)
        written = frames_of(writers_buf.getvalue())
        check("concurrent writers never interleave a frame",
              len(written) == 120
              and all(f.body == payloads[f.req_id] for f in written),
              str(len(written)))

        print("worker transports: both, and the rollback between them")
        import worker_protocol as _wp
        import engines as _eng

        FAKE_BINARY = r'''import os, sys
sys.path.insert(0, %r)
import worker_protocol as wp
reply = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)
sys.stdout = sys.stderr
w = wp.FrameWriter(reply)
w.write(wp.HELLO, wp.hello("binary-v1", "fake", True, "cafe1234"))
held = []
acked = False
while True:
    try:
        f = wp.read_frame(sys.stdin.buffer)
    except wp.StreamClosed:
        break
    if f.kind == wp.HELLO_ACK:
        acked = True
        continue
    if f.kind == wp.SHUTDOWN:
        break
    if f.kind == wp.CANCEL:
        for i, (rid, _t) in enumerate(held):
            if rid == f.req_id:
                held.pop(i)
                w.write(wp.ERROR, {"req_id": rid, "error": {
                    "code": "cancelled", "stage": "queued",
                    "message": "request superseded", "retryable": False,
                    "fatal": False}})
                break
        continue
    text = f.meta.get("text", "")
    if text == "hold":
        held.append((f.req_id, text))
        continue
    if text == "torn":
        # Half a header, then die: a worker that fell over mid-frame.
        reply.write(wp.MAGIC + bytes([1, wp.RESULT]))
        reply.flush()
        os._exit(1)
    if text == "nonative":
        method = "whisper-v1"
    else:
        method = "kokoro-pytorch-duration-v1"
    words = [{"w": t, "s": i * 0.2, "e": i * 0.2 + 0.15, "i": i}
             for i, t in enumerate(text.split())]
    body = b"RIFF" + bytes([0]) + b"\n" + wp.MAGIC + text.encode("utf-8")
    w.write(wp.RESULT, {"req_id": f.req_id, "sample_rate": 24000,
                        "mime_type": "audio/wav", "words": words,
                        "alignment_method": method,
                        "diagnostics": {"native_direct_coverage": 1.0}}, body)
''' % (os.path.abspath("."),)
        fake_dir = tempfile.mkdtemp(prefix="talkify_smoke_worker_")
        fake_path = os.path.join(fake_dir, "fake_binary_worker.py")
        with open(fake_path, "w", encoding="utf-8") as fh:
            fh.write(FAKE_BINARY)

        def binary_spawn():
            proc = subprocess.Popen(
                [sys.executable, fake_path], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            return proc, None, "binary-v1"

        client = _eng.WorkerClient(binary_spawn, name="tkf1")
        try:
            reply = client.submit({"text": "hello there world"}).result(timeout=30)
            check("a binary worker's audio arrives as raw bytes, not base64",
                  isinstance(reply.get("audio"), bytes)
                  and "audio_b64" not in reply
                  and reply["audio"].startswith(b"RIFF"), str(reply)[:120])
            check("...with the frame magic inside the body left alone",
                  _wp.MAGIC in reply["audio"])
            check("the handshake names the backend and its model identity",
                  client.identity == "binary-v1:fake:cafe1234", client.identity)

            held = client.submit({"text": "hold"})
            held.cancel()
            try:
                held.result(timeout=20)
                cancelled = False
            except _eng.CancelledSynth:
                cancelled = True
            check("a cancel is answered over the binary transport too", cancelled)

            live = [client.submit({"text": "hold"}) for _ in range(2)]
            torn = client.submit({"text": "torn"})
            failures = 0
            for handle in live + [torn]:
                try:
                    handle.result(timeout=20)
                except _eng.SynthError:
                    failures += 1
            check("a worker that dies mid-frame retires its epoch and completes"
                  " every outstanding request exactly once",
                  failures == 3, f"{failures} of 3")
            after = client.submit({"text": "back again"}).result(timeout=30)
            check("...and the next request starts a fresh worker",
                  after["audio"].endswith(b"back again"))
        finally:
            client.shutdown()

        cache = coordinator_module.AudioCache(max_bytes=1 << 20)
        cache.put("d", 0, "v", b"aaaa", "audio/wav", [])
        cache.note_identity("binary-v1:pytorch:pytorch")
        kept = cache.get("d", 0, voice="v") is not None
        cache.note_identity("binary-v1:onnx:cafe1234")
        check("audio made by one backend is never served as if the other had"
              " made it",
              kept and cache.get("d", 0, voice="v") is None)
        cache.put("d", 1, "v", b"bbbb", "audio/wav", [])
        cache.note_identity("")
        check("...and an engine that cannot say what it is empties nothing",
              cache.get("d", 1, voice="v") is not None)
        # The coordinator holds prefetched blocks outside this cache, made by
        # the same producer. It learns to drop them from this return value, so
        # the one call that empties the cache has to be the one that says so.
        check("...while the call that does empty it says so exactly once",
              cache.note_identity("binary-v1:pytorch:pytorch") is True
              and cache.note_identity("binary-v1:pytorch:pytorch") is False)

        def silent_spawn():
            proc = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.exit(0)"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL)
            return proc, None, "binary-v1"

        mute = _eng.WorkerClient(silent_spawn, name="mute")
        began = time.time()
        try:
            mute.submit({"text": "hello"}).result(timeout=10)
            failed = False
        except _eng.SynthError:
            failed = True
        waited = time.time() - began
        mute.shutdown()
        check("a worker that never says hello fails fast rather than sitting"
              " out the handshake timeout",
              failed and waited < 10, f"{failed} after {waited:.1f}s")

        plan = make_spoken_plan("hello there world")
        marks = [Word(w["w"], w["s"], w["e"]) for w in [
            {"w": "hello", "s": 0.0, "e": 0.15},
            {"w": "there", "s": 0.2, "e": 0.35},
            {"w": "world", "s": 0.4, "e": 0.55}]]
        check("a reply with no native duration marks falls back and still"
              " times every displayed token",
              well_formed(plan_words(plan, marks, spoken_indices=None,
                                     audio_seconds=0.6), plan, 0.6)[0])

        check("the transport is chosen by name, never guessed from a byte",
              _eng.worker_protocol_name() in _eng.WORKER_PROTOCOLS
              and "--protocol" in open("engines.py", encoding="utf-8").read())
        import kokoro_worker as _kw
        check("...and the worker reads the same name off its own argv",
              _kw._protocol_choice(["w", "--protocol", "binary-v1"]) == "binary-v1"
              and _kw._protocol_choice(["w"]) == "jsonl-v1")
        check("the JSONL transport is still what serve() defaults to",
              isinstance(_kw.JsonlTransport(None, None), _kw.JsonlTransport)
              and "JsonlTransport(inp, out)" in open(
                  "kokoro_worker.py", encoding="utf-8").read())
        check("a result carries the sample rate, so timings are clamped against"
              " the audio actually returned",
              _kw._Response("x", audio=b"", heard=[], sample_rate=24000)
              .frame()[1]["sample_rate"] == 24000)

        print("ONNX assets: verified, or not used")
        import kokoro_assets
        asset_dir = tempfile.mkdtemp(prefix="talkify_smoke_assets_")
        payload = b"pretend this is a model graph" * 100
        source = os.path.join(asset_dir, "incoming.bin")
        with open(source, "wb") as fh:
            fh.write(payload)
        import hashlib as _hashlib
        digest = _hashlib.sha256(payload).hexdigest()
        good = {"model": {"file": "m.onnx", "bytes": len(payload), "sha256": digest}}
        unset = {"model": {"file": "m.onnx", "bytes": len(payload), "sha256": None}}
        wrong = {"model": {"file": "m.onnx", "bytes": len(payload),
                           "sha256": "0" * 64}}
        try:
            kokoro_assets.install("model", source, unset, asset_dir)
            refused_unset = False
        except kokoro_assets.AssetError:
            refused_unset = True
        check("an asset with no recorded hash is refused, not trusted",
              refused_unset and not os.path.exists(os.path.join(asset_dir, "m.onnx")))
        try:
            kokoro_assets.install("model", source, wrong, asset_dir)
            refused_wrong = False
        except kokoro_assets.AssetError:
            refused_wrong = True
        check("...and so is one whose hash does not match",
              refused_wrong
              and not os.path.exists(os.path.join(asset_dir, "m.onnx.incoming")))
        installed = kokoro_assets.install("model", source, good, asset_dir)
        check("a verified asset is renamed into place atomically",
              os.path.exists(installed)
              and kokoro_assets.verified_path("model", good, asset_dir) == installed)
        with open(installed, "ab") as fh:
            fh.write(b"tampered")
        try:
            kokoro_assets.verified_path("model", good, asset_dir)
            caught = False
        except kokoro_assets.AssetError:
            caught = True
        check("...and a file that changed underneath is refused afterwards", caught)
        check("the shipped manifest records verified SHA-256 hashes",
              all(isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64
                  for entry in kokoro_assets.load_manifest().values()))
        shutil.rmtree(asset_dir, ignore_errors=True)
        shutil.rmtree(fake_dir, ignore_errors=True)

        print("figures: one caption, one figure, decided across the whole page")
        import figures as figures_mod

        class _Rect:
            def __init__(self, x0, y0, x1, y1):
                self.x0, self.y0, self.x1, self.y1 = x0, y0, x1, y1

            @property
            def width(self):
                return self.x1 - self.x0

            @property
            def height(self):
                return self.y1 - self.y0

            def __repr__(self):
                return f"Rect({self.x0},{self.y0},{self.x1},{self.y1})"

        def _cap(x0, y0, x1, y1):
            return {"x0": x0, "y0": y0, "x1": x1, "y1": y1}

        # The debrief fixture: a 60pt figure 15pt above its caption, and an
        # unrelated 250pt-wide rect that ends 1pt above the same caption.
        cap_small = _cap(50, 300, 110, 310)
        true_rect = _Rect(50, 225, 110, 285)
        wide_rect = _Rect(40, 150, 290, 299)
        cap_wide = _cap(40, 139, 290, 149)        # the wide rect's own caption
        page_rects = [wide_rect, true_rect]
        matched = figures_mod._match_placements([cap_small, cap_wide], page_rects)
        check("an unrelated wide rect does not win a caption it is merely"
              " nearest to, once the whole page competes for it",
              matched.get(0) == 1 and matched.get(1) == 0, str(matched))
        check("...and the rect that does own a caption still gets it",
              page_rects[matched[1]] is wide_rect)
        check("a short caption under a genuine full-width figure still matches",
              figures_mod._match_placement(
                  _cap(120, 300, 180, 310), [_Rect(40, 150, 290, 290)]) is not None)
        check("two candidates that are a real tie yield no figure at all",
              figures_mod._match_placement(
                  cap_small, [_Rect(50, 225, 110, 285),
                              _Rect(50, 225, 110, 285)]) is None)
        left_cap, right_cap = _cap(40, 300, 140, 310), _cap(200, 300, 300, 310)
        left_rect, right_rect = _Rect(40, 200, 140, 290), _Rect(200, 200, 300, 290)
        columns = [right_rect, left_rect]
        column_match = figures_mod._match_placements([left_cap, right_cap], columns)
        check("columns do not cross-match",
              columns[column_match[0]] is left_rect
              and columns[column_match[1]] is right_rect, str(column_match))
        check("a figure with no plausible placement simply has none",
              figures_mod._match_placements(
                  [cap_small], [_Rect(50, 0, 110, 40)]) == {})

        print("voice switching: the reservation is a barrier, not a hope")
        check("a pending switch is held as its own state, not as the active voice",
              "pendingVoice: null" in app_js and "voiceRequestEpoch: 0" in app_js)
        check("the barrier goes up before the POST, not after the reply",
              app_js.index("state.pendingVoice = pending;")
              < app_js.index('await fetch("/api/voice"'))
        check("...and old work past it is invalidated at the same moment",
              app_js.index("invalidateAhead(barrier);")
              < app_js.index('await fetch("/api/voice"'))
        check("every path that could start a block consults the barrier",
              all("beyondVoiceBarrier" in chunk for chunk in [
                  app_js[app_js.index("async function primeStandby"):
                         app_js.index("function markHere")],
                  app_js[app_js.index("async function loadBlock"):
                         app_js.index("async function loadAudio")],
                  app_js[app_js.index("async function loadAudio"):
                         app_js.index("function invalidateAhead")],
                  app_js[app_js.index("const prefetch ="):
                         app_js.index("let playToken")],
                  app_js[app_js.index("function onSlotEnded"):
                         app_js.index("function onSlotTime")],
                  app_js[app_js.index("async function playBlock"):
                         app_js.index("function stop()")],
              ]))
        check("slotHolds refuses a slot the switch has already invalidated",
              "&& !beyondVoiceBarrier(i)" in app_js)
        check("the ended handler waits rather than handing off the old voice",
              'state.pendingVoice.resumeAt = next;' in app_js
              and 'status("switching voice…")' in app_js)
        check("...and the block it was waiting for is started in the new voice",
              "if (pending.resumeAt !== null) {" in app_js
              and "playBlock(pending.resumeAt, 0);" in app_js)
        check("only the newest request may be applied",
              "const superseded = () => state.pendingVoice !== pending" in app_js
              and "if (superseded()) {" in app_js)
        check("a stale failure clears only its own reservation",
              "if (state.pendingVoice !== pending) return;" in app_js)
        check("a refused switch keeps the voice that is actually being spoken",
              "release();" in app_js and "state.voice = voiceId;" in app_js
              and app_js.index("release();") < app_js.index("state.voice = voiceId;"))
        # Both ends measured from the marker, not from the top of the file:
        # anchoring the end at the *first* "let r;" anywhere in app.js made
        # this assert fail the moment an unrelated function above declared
        # one, which says nothing about the voice-switch release path.
        release_at = app_js.index("const release = () => {")
        release_body = app_js[release_at:app_js.index("let r;", release_at)]
        check("...and restarts playback that was waiting at the barrier, rather"
              " than leaving Pause showing over silence",
              "pending.resumeAt !== null" in release_body
              and "playBlock(pending.resumeAt, 0);" in release_body,
              release_body[:80])
        check("the reservation comes down before anything is restarted",
              app_js.index("state.pendingVoice = null;      // before anything")
              < app_js.index("playBlock(pending.resumeAt, 0);"))
        check("the generation compare-and-set and the play token both survive",
              "expected_generation: state.generation" in app_js
              and "const token = playToken;" in app_js)

        print("durable saves: one document, one lock, no lost field")
        import storage as storage_mod
        race_dir = tempfile.mkdtemp(prefix="talkify_smoke_race_")
        try:
            repo = storage_mod.LibraryRepository(race_dir)
            repo.save("d1", {"blocks": ["a", "b"], "voice": "old",
                             "figure_index": {"state": "pending"}})
            repo.save("d2", {"blocks": ["z"], "voice": "old"})
            entered = threading_module.Event()
            release_voice = threading_module.Event()
            index_done = threading_module.Event()
            other_doc_done = threading_module.Event()

            def slow_voice(record):
                entered.set()
                release_voice.wait(10)
                record["voice"] = "new"

            def index_fields(record):
                record["figure_index"] = {"state": "ready", "count": 3}

            voice_thread = threading_module.Thread(
                target=lambda: repo.update_record("d1", slow_voice))
            voice_thread.start()
            check("an update holds its document's lock", entered.wait(10))

            def do_index():
                repo.update_record("d1", index_fields)
                index_done.set()

            def do_other():
                repo.update_record("d2", lambda r: r.__setitem__("voice", "other"))
                other_doc_done.set()

            index_thread = threading_module.Thread(target=do_index)
            other_thread = threading_module.Thread(target=do_other)
            index_thread.start()
            other_thread.start()
            check("...so a second write to the same document waits for it",
                  not index_done.wait(0.4))
            check("...but a different document is not blocked by it",
                  other_doc_done.wait(10))
            release_voice.set()
            voice_thread.join(10)
            index_thread.join(10)
            other_thread.join(10)
            with open(os.path.join(race_dir, "d1.json"), encoding="utf-8") as fh:
                final = json.load(fh)
            check("the record that lands keeps both changes - an index save"
                  " cannot revert a voice switch",
                  final.get("voice") == "new"
                  and final.get("figure_index") == {"state": "ready", "count": 3},
                  json.dumps(final)[:160])
            check("...and the blocks are untouched by either of them",
                  final.get("blocks") == ["a", "b"], str(final.get("blocks")))

            original_write = storage_mod._atomic_write
            failures = {"n": 0}

            def failing_write(path, data, mode="w"):
                failures["n"] += 1
                raise OSError("disk went away")

            storage_mod._atomic_write = failing_write
            try:
                repo.update_record("d1", lambda r: r.__setitem__("voice", "lost"))
                raised = False
            except OSError:
                raised = True
            finally:
                storage_mod._atomic_write = original_write
            check("a failed atomic write is reported, not swallowed",
                  raised and failures["n"] == 1)
            check("...and leaves the repository believing what is actually on disk",
                  repo.latest_record("d1").get("voice") == "new",
                  repo.latest_record("d1").get("voice"))
            repo.update_record("d1", lambda r: r.__setitem__("voice", "third"))
            with open(os.path.join(race_dir, "d1.json"), encoding="utf-8") as fh:
                check("...so the next save still writes every field",
                      json.load(fh).get("figure_index")
                      == {"state": "ready", "count": 3})
            repo.close()
        finally:
            shutil.rmtree(race_dir, ignore_errors=True)

        talkify_src = open("talkify.py", encoding="utf-8").read()
        check("every durable document mutation goes through update_record",
              "LIBRARY.update_record(doc_id, snapshot)" in talkify_src
              and "LIBRARY.update_record(doc_id, patch)" in talkify_src
              and "LIBRARY.save(doc_id, doc.record())" not in talkify_src)
        check("...and the snapshot is taken inside the lock, not before it",
              "def snapshot(record):" in talkify_src
              and talkify_src.index("def snapshot(record):")
              < talkify_src.index("LIBRARY.update_record(doc_id, snapshot)"))
        check("the voice route and the indexers write their own fields and"
              " nothing else",
              '_save_doc_fields(doc_id, "voice")' in talkify_src
              and '_save_doc_fields(doc_id, "figure_index")' in talkify_src
              and '_save_doc_fields(doc_id, "chapters", "chapter_detection")'
              in talkify_src)
        check("...but a partial save still takes the block list from the live"
              " document, so an index cannot restore a shorter book",
              'for name in ("blocks", "block_pages", "content_revision",'
              ' "ingestion"):' in talkify_src)

        print("adversarial review findings, pinned so they cannot come back")
        # Every check below is a defect an independent audit of this diff found
        # (one on the normalizer, one on the whole change). They are kept
        # together because their value is the failing input, not the code path.

        def spoken(text):
            return make_spoken_plan(text).spoken_text

        for text in ("The element at [0] is null.",
                     "The interval [0, 1] is closed.",
                     "Djokovic won [6-4, 7-5] today.",
                     "Rows [12-9] were removed."):
            if make_spoken_plan(text).silent_sources:
                check("a bracket that is maths or a score is not a citation",
                      False, text)
                break
        else:
            check("a bracket that is maths or a score is not a citation", True)
        check("...while a real reference list still is",
              spoken("See [12-14] and [12, 14; 18].") == "See and.",
              spoken("See [12-14] and [12, 14; 18]."))
        check("a page locator inside a citation is not read as its year",
              spoken("See (Smith, 2020, p. 1200) for details.")
              == "See for details.",
              spoken("See (Smith, 2020, p. 1200) for details."))
        check("...but a number that cannot be a year still keeps its text",
              spoken("Meet at (Bay 4200) tomorrow.")
              == "Meet at (Bay 4200) tomorrow.")
        check("a parenthetical that opens with a date word is not an author",
              spoken("Shakespeare (Born 1564) wrote plays.")
              == "Shakespeare (Born 1564) wrote plays.")

        wiki = "See https://en.wikipedia.org/wiki/Python_(programming_language) now."
        check("a closing bracket the address itself opened is not trimmed off",
              spoken(wiki) == "See link now.", spoken(wiki))
        check("a URL written inside brackets is still spoken as a link",
              spoken("Visit (https://example.com) now.") == "Visit (link) now.",
              spoken("Visit (https://example.com) now."))
        long_tail = "https://example.com" + ("." * 300)
        began = time.time()
        make_spoken_plan(long_tail)
        check("...and three hundred trailing dots do not become a quadratic scan",
              time.time() - began < 0.25, f"{time.time() - began:.3f}s")

        check("an abbreviation joined by a hyphen is left alone",
              spoken("A PSI-high reading and a low-PSI one.")
              == "A PSI-high reading and a low-PSI one.",
              spoken("A PSI-high reading and a low-PSI one."))
        check("...while the same abbreviation standing alone is still spelled out",
              spoken("The gauge read 45 PSI.") == "The gauge read 45 P S I.")

        tail_plan = make_spoken_plan("alpha beta gamma")
        partial = [Word("alpha", 0.0, 0.5), Word("beta", 0.5, 1.0)]
        filled = plan_words(tail_plan, partial, spoken_indices=[0, 1],
                            audio_seconds=3.0)
        check("a final word the backend did not mark is timed against the audio,"
              " not against the last mark it did give",
              len(filled) == 3 and filled[2].e > 2.5, str(filled[2]))
        okay, why = well_formed(filled, tail_plan, 3.0)
        check("...and the block is still monotonic and inside its clip", okay, why)

        print("...and the same for the parts a browser cannot check")
        race_dir = tempfile.mkdtemp(prefix="talkify_smoke_audit_")
        try:
            repo = storage_mod.LibraryRepository(race_dir)
            repo.save("n1", {"blocks": ["a"], "voice": "v",
                             "position": {"block": 1, "word": 0}})
            taken = repo.latest_record("n1")
            taken["position"]["block"] = 999
            repo.update_record("n1", lambda r: r.__setitem__("voice", "w"))
            with open(os.path.join(race_dir, "n1.json"), encoding="utf-8") as fh:
                saved = json.load(fh)
            check("a record handed out is detached, so changing it afterwards"
                  " cannot reach through to the next save",
                  saved["position"]["block"] == 1 and saved["voice"] == "w",
                  json.dumps(saved))

            entered = threading_module.Event()
            go = threading_module.Event()

            def slow(record):
                entered.set()
                go.wait(10)
                record["voice"] = "late"

            slow_thread = threading_module.Thread(
                target=lambda: repo.update_record("n1", slow))
            slow_thread.start()
            entered.wait(10)
            deleted = threading_module.Event()

            def do_delete():
                repo.delete("n1")
                deleted.set()

            delete_thread = threading_module.Thread(target=do_delete)
            delete_thread.start()
            check("a delete waits for a save already under way",
                  not deleted.wait(0.4))
            go.set()
            slow_thread.join(10)
            delete_thread.join(10)
            check("...so a deleted document cannot be written back into existence",
                  not os.path.exists(os.path.join(race_dir, "n1.json")))
            repo.close()
        finally:
            shutil.rmtree(race_dir, ignore_errors=True)

        rect_tie = figures_mod._match_placements(
            [_cap(50, 290, 110, 300), _cap(50, 291, 110, 301)],
            [_Rect(50, 200, 110, 289)])
        check("one figure that two captions both nearly own goes to neither",
              rect_tie == {}, str(rect_tie))

        check("the client catches up to the voice the server actually committed",
              "serverVoice: null" in app_js
              and "state.voice = state.serverVoice;" in app_js
              and "state.serverVoice = voiceId;" in app_js)
        check("...recorded even from a reply that arrived too late to apply",
              app_js.index("state.serverVoice = voiceId;")
              < app_js.index("if (superseded()) {"))
    finally:
        srv.terminate()
        srv.wait(timeout=5)
        shutil.rmtree(tmp_library, ignore_errors=True)
        shutil.rmtree(os.path.dirname(tmp_config), ignore_errors=True)
        shutil.rmtree(tmp_exports, ignore_errors=True)

    print(f"\n{ok} passed, {fail} failed")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
