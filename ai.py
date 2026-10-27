"""
Language model access for Talkify - see try_explain.py, which validated this prompt against real
Ollama output before any UI was built around it.

    explain(prompt, context) -> (answer, warning)

One interface, four implementations, chosen by config: Google Gemini,
Anthropic Claude and OpenAI, each of which needs a key the user supplies,
and local Ollama, which needs nothing but being installed.

Every call is plain urllib over the wire - no vendor SDK is imported, let
alone linked (talk to it over HTTP, never link it), so no
provider's licence follows this code and requirements.txt stays as short as
it is.

If no key is set and nothing answers on the Ollama port, `available()` is
False and the caller shows no button at all - same degrade-to-nothing rule
as everything else in this app.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))

PROBE_TIMEOUT = 1.5     # just checking Ollama is there - never worth a long wait
CALL_TIMEOUT = 60
MAX_TOKENS = 1024       # a paragraph of explanation; the cloud ones bill for these

SYSTEM_EXPLAIN = (
    "You explain a passage from a book to a reader who has it open in front "
    "of them. Explain only what the passage itself says. If it states a "
    "result without deriving it, say so rather than inventing the "
    "derivation. Never introduce facts, numbers, or steps that are not in "
    "the passage. Be brief, concrete, and conversational."
)

# provider -> (config key, env var, default model, vendor, friendly label)
CLOUD = {
    "gemini": ("gemini_api_key", "GEMINI_API_KEY",
               "gemini-2.5-flash", "Gemini", "Gemini 2.5 Flash"),
    "anthropic": ("anthropic_api_key", "ANTHROPIC_API_KEY",
                  "claude-3-5-haiku-20241022", "Claude", "Claude 3.5 Haiku"),
    "openai": ("openai_api_key", "OPENAI_API_KEY",
               "gpt-4o-mini", "OpenAI", "GPT-4o mini"),
}

# Auto-detection order. A key is something the user deliberately went and
# pasted into config.json, so it outranks whatever happens to be listening on
# the Ollama port. Ollama is last only in that sense - it stays the option
# that costs nothing and needs no configuration at all.
ORDER = ("gemini", "anthropic", "openai", "ollama")

DEFAULT_OLLAMA_MODEL = "llama3.2"
DEFAULT_OLLAMA_HOST = "http://localhost:11434"


def _config_path():
    return os.environ.get("TALKIFY_CONFIG_PATH") or os.path.join(HERE, "config.json")


_CONFIG_CACHE = None                    # (path, (size, mtime_ns), parsed dict)


def _config():
    """config.json, re-parsed only when it changes on disk.

    resolve() asks _setting() up to half a dozen times, and each of those used
    to open and parse the file again - so deciding whether the explain feature
    is available at all read the same few hundred bytes six times over. Keyed
    on path as well as stat, since the tests point TALKIFY_CONFIG_PATH at a
    different file mid-process.
    """
    global _CONFIG_CACHE
    path = _config_path()
    try:
        st = os.stat(path)
        key = (path, (st.st_size, st.st_mtime_ns))
    except OSError:
        _CONFIG_CACHE = None
        return {}
    cached = _CONFIG_CACHE
    if cached is not None and cached[0] == key:
        return cached[1]
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError):
        return {}
    data = data if isinstance(data, dict) else {}
    _CONFIG_CACHE = (key, data)
    return data


def _setting(key, env=None):
    """config.json first, then the environment. config.json is the documented
    home for a key (it is gitignored); the environment is the
    convenient one for a shell that already exports it."""
    for v in (_config().get(key), os.environ.get(env) if env else None):
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _key(provider):
    return _setting(CLOUD[provider][0], CLOUD[provider][1])


def _model_for(provider):
    if provider == "ollama":
        return _setting("ollama_model") or DEFAULT_OLLAMA_MODEL
    return _setting(f"{provider}_model") or CLOUD[provider][2]


def _label(provider, model):
    if provider == "ollama":
        return "Local (Ollama)"
    _, _, default, vendor, friendly = CLOUD[provider]
    return friendly if model == default else f"{vendor} ({model})"


def _host():
    return (os.environ.get("OLLAMA_HOST") or _config().get("ollama_host")
            or DEFAULT_OLLAMA_HOST)


def ollama_available():
    try:
        urllib.request.urlopen(_host() + "/api/tags", timeout=PROBE_TIMEOUT)
        return True
    except Exception:
        return False


def resolve():
    """Which provider would answer right now, or None.

    A cloud provider counts as available on the strength of its key alone:
    probing it costs a request and, on a metered key, money. Ollama is
    probed, because asking is free and because "installed" and "running" are
    different things."""
    want = (_setting("ai_provider", "TALKIFY_AI_PROVIDER") or "").lower()
    if want:
        # An explicit choice is honoured or it fails - never quietly swapped
        # for another provider, since a user who picked local may have picked
        # it precisely so that nothing leaves the machine.
        if want in CLOUD:
            return want if _key(want) else None
        if want == "ollama":
            return "ollama" if ollama_available() else None
        return None
    for p in ORDER:
        if p == "ollama":
            if ollama_available():
                return "ollama"
        elif _key(p):
            return p
    return None


def provider_info():
    """What the UI needs to decide whether to show the feature at all, and
    what to say about it when it does. `local` is the part the reader
    actually cares about: it is the difference between the passage staying
    on this machine and the passage going to somebody's API."""
    p = resolve()
    if p is None:
        return {"available": False, "provider": None, "model": None,
                "local": False, "label": None}
    model = _model_for(p)
    return {"available": True, "provider": p, "model": model,
            "local": p == "ollama", "label": _label(p, model)}


def available():
    """Whether the explain feature has anything to talk to at all."""
    return resolve() is not None


# ---------------------------------------------------------------- transport

def _redact(text, secret):
    """Gemini takes its key in the query string, so an error quoting the URL
    would put that key in the response body and in the log. Nothing here lets
    a key back out (never log it, never send it anywhere but the
    provider)."""
    return text.replace(secret, "***") if secret else text


def _post(url, body, headers, timeout, secret=None):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode("utf-8", "replace")[:300]
        except Exception:
            detail = ""
        raise RuntimeError(_redact(f"HTTP {e.code}: {detail}", secret)) from None
    except urllib.error.URLError as e:
        raise RuntimeError(_redact(f"couldn't reach the model: {e.reason}", secret)) from None
    except ValueError as e:
        raise RuntimeError(f"the model returned something that isn't JSON: {e}") from None


def _truncation_warning(prompt_tokens, sent_chars):
    """The one defence against silent truncation: compare what the
    model says it read against what was sent. Every provider here reports a
    prompt-token count, and Ollama - the one that actually truncates without
    saying so - is no exception. Six characters per token is deliberately
    generous (English runs about four), so this fires on a real drop rather
    than on a tokenizer quirk."""
    if prompt_tokens is None or prompt_tokens >= sent_chars // 6:
        return None
    return "This answer may be degraded - the model appears to have truncated the prompt."


def _text_from(resp, pick, provider):
    """Every provider buries the answer somewhere different, and every one of
    them can return a well-formed reply with no text in it - a safety block,
    a refusal, an empty choice. Missing text is an error with a name, not a
    KeyError escaping a request handler."""
    try:
        return pick(resp)
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"{provider} returned no usable text ({e})") from None


def _ask_ollama(system, prompt, num_ctx, temperature, timeout):
    options = {"num_ctx": num_ctx}
    if temperature is not None:
        options["temperature"] = temperature
    resp = _post(_host() + "/api/generate",
                 {"model": _model_for("ollama"), "system": system,
                  "prompt": prompt, "stream": False, "options": options},
                 {"Content-Type": "application/json"}, timeout)
    return resp.get("response") or "", resp.get("prompt_eval_count")


def _ask_gemini(system, prompt, temperature, timeout):
    key, model = _key("gemini"), _model_for("gemini")
    gen_config = {"maxOutputTokens": MAX_TOKENS}
    if temperature is not None:
        gen_config["temperature"] = temperature
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": gen_config}
    resp = _post("https://generativelanguage.googleapis.com/v1beta/models/"
                 f"{model}:generateContent?key={key}",
                 body, {"Content-Type": "application/json"}, timeout, secret=key)
    text = _text_from(resp, lambda r: "".join(
        p.get("text", "") for p in r["candidates"][0]["content"]["parts"]), "Gemini")
    return text, (resp.get("usageMetadata") or {}).get("promptTokenCount")


def _ask_anthropic(system, prompt, temperature, timeout):
    key, model = _key("anthropic"), _model_for("anthropic")
    body = {"model": model, "system": system, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    resp = _post("https://api.anthropic.com/v1/messages", body,
                 {"x-api-key": key, "anthropic-version": "2023-06-01",
                  "content-type": "application/json"}, timeout, secret=key)
    text = _text_from(resp, lambda r: "".join(
        b.get("text", "") for b in r["content"] if b.get("type") == "text"), "Claude")
    return text, (resp.get("usage") or {}).get("input_tokens")


def _ask_openai(system, prompt, temperature, timeout):
    key, model = _key("openai"), _model_for("openai")
    body = {"model": model, "max_tokens": MAX_TOKENS,
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}]}
    if temperature is not None:
        body["temperature"] = temperature
    resp = _post("https://api.openai.com/v1/chat/completions", body,
                 {"Authorization": f"Bearer {key}",
                  "Content-Type": "application/json"}, timeout, secret=key)
    text = _text_from(
        resp, lambda r: r["choices"][0]["message"]["content"] or "", "OpenAI")
    return text, (resp.get("usage") or {}).get("prompt_tokens")


_CLOUD_ASK = {"gemini": _ask_gemini, "anthropic": _ask_anthropic,
              "openai": _ask_openai}


def ask(system, prompt, num_ctx=4096, temperature=None, timeout=CALL_TIMEOUT):
    """Lower-level entry point: one system prompt, one user prompt, no
    assumptions about what the answer is for. explain() below is the common
    case built on this; chapters.py's contents-page extraction needs a
    different system prompt (a low temperature, and more patience - a
    structured-JSON generation over a whole page of rows genuinely takes
    longer than a short conversational answer, and it runs in a background
    thread anyway, so there's no UI waiting on it).

    `num_ctx` only means anything to Ollama, the one backend here whose
    context window is small enough to have to be told about. The cloud
    providers ignore it rather than rejecting it, so callers never have to
    know which one is in use.

    Returns (text, warning); the warning is None unless the answer looks
    degraded."""
    provider = resolve()
    if provider is None:
        raise RuntimeError("no language model is configured")
    if provider == "ollama":
        text, tokens = _ask_ollama(system, prompt, num_ctx, temperature, timeout)
    else:
        text, tokens = _CLOUD_ASK[provider](system, prompt, temperature, timeout)
    text = text.strip()
    if not text:
        raise RuntimeError(f"{provider} returned an empty response")
    return text, _truncation_warning(tokens, len(system) + len(prompt))


def explain(prompt, context):
    """`context` is one block's worth of text - a single paragraph, capped
    at MAX_BLOCK (320 chars) by extract.py - deliberately small: Ollama
    silently truncates from the *front* of an over-long prompt rather than
    erroring, which would drop the system prompt first and
    nobody would know why the answer got worse. One short passage per call
    keeps every request nowhere near that edge, on any backend."""
    full_prompt = f"Passage:\n\"\"\"\n{context}\n\"\"\"\n\nQuestion: {prompt}"
    return ask(SYSTEM_EXPLAIN, full_prompt)
