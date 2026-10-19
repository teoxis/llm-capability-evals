#!/usr/bin/env python3
"""Measures how many tokens an opencode agent sends on every turn, broken down.

Starts a small proxy that forwards to llama-server and captures the request bodies,
runs one `opencode run` turn through it, and then counts tokens with the model's REAL
tokenizer (llama-server's /tokenize endpoint), not with estimates.

On a local model this matters: the fixed system prompt plus tool definitions are paid
on every single turn, out of a small context window.

Usage:  prompt_cost.py [label]

Environment:
  LLM_SERVER_URL   llama-server root URL         (default http://127.0.0.1:8080)
  LLM_API_KEY      bearer token for llama-server (optional)
  OPENCODE_CONFIG  opencode config to copy      (default ~/.config/opencode/opencode.json)
  OPENCODE_PROVIDER provider entry to redirect   (default llamacpp)
  PROXY_PORT       local proxy port              (default 8090)
"""
import http.server
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import urllib.request
from pathlib import Path

UPSTREAM = os.environ.get("LLM_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
KEY = os.environ.get("LLM_API_KEY", "")
CONFIG = Path(os.environ.get("OPENCODE_CONFIG", "~/.config/opencode/opencode.json")).expanduser()
PROVIDER = os.environ.get("OPENCODE_PROVIDER", "llamacpp")
PORT = int(os.environ.get("PROXY_PORT", "8090"))
captured = []
MODEL = ""  # filled from the captured request: llama-server in router mode needs it for /tokenize


class Proxy(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _forward(self, body=None):
        req = urllib.request.Request(UPSTREAM + self.path, data=body, method=self.command,
                                     headers={k: v for k, v in self.headers.items()
                                              if k.lower() not in ("host", "content-length")})
        with urllib.request.urlopen(req, timeout=600) as r:
            data = r.read()
            self.send_response(r.status)
            for k, v in r.headers.items():
                if k.lower() not in ("transfer-encoding", "content-length", "connection"):
                    self.send_header(k, v)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    def do_GET(self):
        self._forward()

    def do_POST(self):
        body = self.rfile.read(int(self.headers["Content-Length"]))
        if self.path.endswith("/chat/completions"):
            captured.append(json.loads(body))
        self._forward(body)


def ntok(text: str) -> int:
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {KEY}"} if KEY else {})}
    req = urllib.request.Request(UPSTREAM + "/tokenize", data=json.dumps({"content": text, **({"model": MODEL} if MODEL else {})}).encode(), headers=headers)
    return len(json.load(urllib.request.urlopen(req))["tokens"])


def main() -> None:
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Proxy)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    cfg = json.loads(CONFIG.read_text())
    cfg["provider"][PROVIDER]["options"]["baseURL"] = f"http://127.0.0.1:{PORT}/v1"
    tmp = Path(tempfile.mkdtemp())
    (tmp / "opencode.json").write_text(json.dumps(cfg))
    env = dict(os.environ, OPENCODE_CONFIG=str(tmp / "opencode.json"))
    # stdin closed on purpose: without a terminal, `opencode run` reads stdin until EOF and hangs.
    subprocess.run(["opencode", "run", "say only: hello"], cwd=tmp, env=env,
                   stdin=subprocess.DEVNULL, capture_output=True, timeout=600)
    srv.shutdown()
    if not captured:
        sys.exit("No chat request was captured: is the provider name right (OPENCODE_PROVIDER)?")

    global MODEL
    main_req = max(captured, key=lambda b: len(json.dumps(b)))  # the agent's request, not the title one
    MODEL = main_req.get("model", "")
    system = "\n".join(m["content"] if isinstance(m["content"], str) else json.dumps(m["content"])
                       for m in main_req["messages"] if m["role"] == "system")
    tools = main_req.get("tools", [])
    by_group: dict[str, int] = {}
    for t in tools:
        n = t["function"]["name"]
        g = n.split("_")[0] if "_" in n else "builtin"
        by_group[g] = by_group.get(g, 0) + ntok(json.dumps(t))
    total = ntok(system) + sum(by_group.values())
    label = sys.argv[1] if len(sys.argv) > 1 else ""
    print(f"[{label}] requests={len(captured)}  system={ntok(system)}  "
          f"tools={sum(by_group.values())} ({len(tools)})  TOTAL~{total}")
    detail = sorted(((t["function"]["name"], ntok(json.dumps(t))) for t in tools), key=lambda x: -x[1])
    print("  per tool:", ", ".join(f"{n}={c}" for n, c in detail))
    for m in main_req["messages"]:
        if m["role"] == "system":
            for block in re.split(r"\n(?=# |<)", m["content"] if isinstance(m["content"], str) else ""):
                if ntok(block) > 300:
                    print(f"  system> {ntok(block):5d}  {block[:70]!r}")
    print("  per group:", dict(sorted(by_group.items(), key=lambda x: -x[1])))


if __name__ == "__main__":
    main()
