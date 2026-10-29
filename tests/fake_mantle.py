"""Fake Bedrock Mantle (OpenAI-compatible) server for an offline end-to-end run.

Returns the function_minimization initial program with a random perturbation,
reports usage, and answers 429 to ~15% of requests to exercise retry.
"""
import json
import random
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROGRAM = Path(sys.argv[2]).read_text()
random.seed(0)


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        assert "seed" not in body, "seed must not be sent to Bedrock"
        assert body.get("top_p", 1) is not None, "top_p: null must not be sent"
        if random.random() < 0.15:
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Too many requests","type":"throttling"}}')
            return
        prog = PROGRAM.replace("step_size = 0.1", f"step_size = {random.uniform(0.02, 0.5):.3f}")
        prog = prog.replace("iterations=1000", f"iterations={random.randint(500, 3000)}")
        content = "Here is an improved version:\n```python\n" + prog + "\n```\n"
        ptoks = sum(len(m.get("content", "")) for m in body.get("messages", [])) // 4
        resp = {
            "id": "chatcmpl-fake", "object": "chat.completion", "created": int(time.time()),
            "model": body.get("model"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": ptoks, "completion_tokens": len(content) // 4,
                      "total_tokens": ptoks + len(content) // 4,
                      "prompt_tokens_details": {"cached_tokens": 0}},
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    port = int(sys.argv[1])
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
