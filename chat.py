"""Chat that stays on the local Ollama model unless a question needs live data."""

import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

OLLAMA_HOST = "http://127.0.0.1:11434"
OFFLINE_MODEL = "llama3.1:8b"
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5-search-api")
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MAX_CONTEXT_CHARS = 12_000

OFFLINE_SYSTEM = (
    "You are a local assistant with no internet access. "
    "Answer from the conversation, the local notes, and general knowledge "
    "you already have. If the user needs a live fact that is not in the notes, "
    "say that it needs a current lookup."
)
ONLINE_SYSTEM = (
    "You answer questions that need current public information. "
    "Search the web and cite the sources you used. Be concise."
)
ROUTE_PROMPT = """Decide whether the latest question needs live internet data.

Reply with exactly one word: ONLINE or OFFLINE.

ONLINE: news, weather, prices, scores, releases, laws, people in office, or any fact that changes and is not answered by the local notes.
OFFLINE: reasoning, coding, math, general knowledge, conversation, or anything the local notes already answer.

Local notes:
{notes}

Question:
{question}
"""


def load_dotenv():
    path = Path(__file__).resolve().parent / ".env"
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def ollama(path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        OLLAMA_HOST + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def ensure_model():
    try:
        with ollama("/api/tags", timeout=5) as resp:
            body = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        print("Ollama is not reachable at 127.0.0.1:11434.")
        print("Start it, then run this script again: ollama serve")
        raise SystemExit(1) from exc

    names = [model.get("name", "") for model in body.get("models", [])]
    if OFFLINE_MODEL not in names:
        installed = ", ".join(names) if names else "none"
        print(f"{OFFLINE_MODEL} is not installed. Installed models: {installed}")
        raise SystemExit(1)


def load_notes():
    if not KNOWLEDGE_DIR.is_dir():
        return ""

    chunks = []
    total = 0
    files = sorted(
        path
        for path in KNOWLEDGE_DIR.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".md"}
    )
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            continue
        block = f"--- {path.relative_to(KNOWLEDGE_DIR).as_posix()} ---\n{text}\n"
        if total + len(block) > MAX_CONTEXT_CHARS:
            print(f"Skipped {path.name}: local notes are already at the size limit.")
            continue
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks)


def error_text(exc):
    if isinstance(exc, urllib.error.HTTPError):
        try:
            body = exc.read().decode("utf-8", errors="replace")
            parsed = json.loads(body)
            message = parsed.get("error", {}).get("message")
            if message:
                return message
        except (OSError, json.JSONDecodeError, AttributeError):
            pass
    return str(exc)


def needs_online(question, notes):
    payload = {
        "model": OFFLINE_MODEL,
        "messages": [
            {
                "role": "user",
                "content": ROUTE_PROMPT.format(
                    notes=notes or "(none)",
                    question=question,
                ),
            }
        ],
        "stream": False,
        "options": {"temperature": 0, "num_predict": 8},
    }
    with ollama("/api/chat", payload, timeout=120) as resp:
        body = json.load(resp)
    decision = body.get("message", {}).get("content", "")
    match = re.search(r"\b(ONLINE|OFFLINE)\b", decision.upper())
    return bool(match and match.group(1) == "ONLINE")


def stream_offline(history, notes):
    system = OFFLINE_SYSTEM
    if notes:
        system += "\n\nLocal notes:\n" + notes
    payload = {
        "model": OFFLINE_MODEL,
        "messages": [{"role": "system", "content": system}, *history],
        "stream": True,
        "options": {"temperature": 0.2},
    }
    with ollama("/api/chat", payload) as resp:
        for raw in resp:
            line = raw.strip()
            if not line:
                continue
            event = json.loads(line.decode("utf-8"))
            if event.get("error"):
                raise RuntimeError(event["error"])
            piece = event.get("message", {}).get("content", "")
            if piece:
                yield piece
            if event.get("done"):
                return


def stream_online(history):
    api_key = os.environ.get("OPENAI_API_KEY", "")
    payload = {
        "model": OPENAI_MODEL,
        "messages": [{"role": "system", "content": ONLINE_SYSTEM}, *history],
        "stream": True,
        "web_search_options": {"search_context_size": "medium"},
    }
    req = urllib.request.Request(
        OPENAI_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                return
            event = json.loads(data)
            if event.get("error"):
                raise RuntimeError(event["error"].get("message", event["error"]))
            choices = event.get("choices") or []
            if not choices:
                continue
            piece = (choices[0].get("delta") or {}).get("content") or ""
            if piece:
                yield piece


def answer(history, question, notes):
    try:
        online = needs_online(question, notes)
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        print(f"Routing failed, staying offline: {error_text(exc)}")
        online = False

    if online and not os.environ.get("OPENAI_API_KEY"):
        print("This question needs live data. Set OPENAI_API_KEY, then ask again.")
        return

    label = f"openai ({OPENAI_MODEL})" if online else f"offline ({OFFLINE_MODEL})"
    print(f"{label}> ", end="", flush=True)
    history.append({"role": "user", "content": question})
    parts = []
    try:
        stream = stream_online(history) if online else stream_offline(history, notes)
        for piece in stream:
            parts.append(piece)
            print(piece, end="", flush=True)
    except (urllib.error.URLError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
        history.pop()
        print(f"\nRequest failed: {error_text(exc)}")
        return
    print()
    history.append({"role": "assistant", "content": "".join(parts)})


def main():
    load_dotenv()
    global OPENAI_MODEL
    OPENAI_MODEL = os.environ.get("OPENAI_MODEL", OPENAI_MODEL)

    ensure_model()
    notes = load_notes()
    if notes:
        print(f"Using local notes from {KNOWLEDGE_DIR}")
    else:
        print(f"No notes yet. Add .txt or .md files to {KNOWLEDGE_DIR}")

    key_state = "ready" if os.environ.get("OPENAI_API_KEY") else "missing OPENAI_API_KEY"
    print(f"Offline model: {OFFLINE_MODEL}")
    print(f"Online model: {OPENAI_MODEL} ({key_state})")
    print("Commands: /exit  /clear\n")

    history = []
    if len(sys.argv) > 1:
        answer(history, " ".join(sys.argv[1:]), notes)
        return

    while True:
        try:
            user = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not user:
            continue
        if user in {"/exit", "/quit"}:
            return
        if user == "/clear":
            history = []
            print("Conversation cleared.")
            continue
        answer(history, user, notes)


if __name__ == "__main__":
    main()
