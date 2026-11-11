"""Offline chat with the local Ollama model. No network calls leave this machine."""

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

HOST = "http://127.0.0.1:11434"
MODEL = "llama3.1:8b"
KNOWLEDGE_DIR = Path(__file__).resolve().parent / "knowledge"
MAX_CONTEXT_CHARS = 12_000

SYSTEM = (
    "You are an offline assistant running on this computer. "
    "You have no internet access. Answer from the conversation and from the "
    "local notes included below. If the notes do not contain the answer, say "
    "that you do not have that information locally. Do not invent sources."
)


def request(path, payload=None, timeout=600):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        HOST + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    return urllib.request.urlopen(req, timeout=timeout)


def ensure_model():
    try:
        with request("/api/tags", timeout=5) as resp:
            body = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as exc:
        print("Ollama is not reachable at 127.0.0.1:11434.")
        print("Start it, then run this script again: ollama serve")
        raise SystemExit(1) from exc

    names = [model.get("name", "") for model in body.get("models", [])]
    if MODEL not in names:
        installed = ", ".join(names) if names else "none"
        print(f"{MODEL} is not installed. Installed models: {installed}")
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


def stream_reply(messages):
    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.2},
    }
    with request("/api/chat", payload) as resp:
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


def answer(messages, question):
    messages.append({"role": "user", "content": question})
    parts = []
    try:
        for piece in stream_reply(messages):
            parts.append(piece)
            print(piece, end="", flush=True)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as exc:
        messages.pop()
        print(f"\nRequest failed: {exc}")
        return
    print()
    messages.append({"role": "assistant", "content": "".join(parts)})


def main():
    ensure_model()
    notes = load_notes()
    system = SYSTEM
    if notes:
        system += "\n\nLocal notes:\n" + notes
        print(f"Using local notes from {KNOWLEDGE_DIR}")
    else:
        print(f"No notes yet. Add .txt or .md files to {KNOWLEDGE_DIR}")

    print(f"Offline model: {MODEL}")
    print("Commands: /exit  /clear\n")

    messages = [{"role": "system", "content": system}]

    if len(sys.argv) > 1:
        answer(messages, " ".join(sys.argv[1:]))
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
            messages = [{"role": "system", "content": system}]
            print("Conversation cleared.")
            continue
        print("model> ", end="", flush=True)
        answer(messages, user)


if __name__ == "__main__":
    main()
