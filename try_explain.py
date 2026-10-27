#!/usr/bin/env python3
"""
Is the explain feature worth building? Find out before writing any of it.

    python try_explain.py sometext.txt            # picks a middling paragraph
    python try_explain.py sometext.txt --block 12
    python try_explain.py sometext.txt --model qwen2.5:7b --ask "what does sigma mean here"

Sends one paragraph of your own material to Ollama with the prompt the real
feature would use, prints the answer, and checks whether the prompt was
silently truncated. Read the answer as a student: would it have helped, or
would it have made you doubt the book?
"""
import argparse, json, os, sys, urllib.error, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

SYSTEM = ("You explain passages from technical textbooks to an engineering "
          "student. Explain only what the passage says. If it states a result "
          "without deriving it, say so rather than inventing the derivation. "
          "Never introduce numbers or steps that are not in the passage. "
          "Be brief and concrete.")


def post(path, payload, timeout=180):
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file", help="a .txt (or .pdf/.epub) of your own")
    ap.add_argument("--model", default="llama3.2")
    ap.add_argument("--block", type=int, help="which paragraph; default is a middling one")
    ap.add_argument("--ask", default="Explain this passage.")
    ap.add_argument("--num-ctx", type=int, default=8192)
    a = ap.parse_args()

    import extract
    blocks = extract.load(a.file)
    if not blocks:
        sys.exit("no text found in that file")
    i = a.block if a.block is not None else len(blocks) // 2
    i = max(0, min(i, len(blocks) - 1))
    passage = blocks[i]

    print(f"{os.path.basename(a.file)}: {len(blocks)} paragraphs, using #{i}")
    print(f"model: {a.model}   num_ctx: {a.num_ctx}\n")
    print("PASSAGE " + "-" * 62)
    print(passage)
    print("-" * 70 + "\n")

    payload = {
        "model": a.model,
        "stream": False,
        "options": {"num_ctx": a.num_ctx, "temperature": 0.2},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": f"{a.ask}\n\nPassage:\n{passage}"},
        ],
    }

    try:
        r = post("/api/chat", payload)
    except urllib.error.URLError as e:
        sys.exit(f"could not reach Ollama at {HOST}\n  {e}\n"
                 "Is it running? Try:  ollama serve   (and: ollama pull " + a.model + ")")

    print("ANSWER " + "-" * 63)
    print(r.get("message", {}).get("content", "").strip())
    print("-" * 70)

    # the silent-truncation check: if far fewer tokens went in than we sent,
    # the head of the prompt was dropped and the answer is not trustworthy
    sent_chars = len(SYSTEM) + len(a.ask) + len(passage)
    used = r.get("prompt_eval_count")
    if used is not None:
        rough = sent_chars / 4          # ~4 chars per token, good enough to spot a cliff
        flag = "  <-- LIKELY TRUNCATED" if used < rough * 0.7 else ""
        print(f"\nprompt tokens counted: {used}  (roughly {rough:.0f} expected){flag}")
    dur = r.get("total_duration")
    if dur:
        print(f"took {dur / 1e9:.1f}s")


if __name__ == "__main__":
    main()
