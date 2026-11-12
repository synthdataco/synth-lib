import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from synth_lib.benchmark.agy_gateway import make_server


class Upstream(BaseHTTPRequestHandler):
    """Records what reaches the proxy; /fail answers 400, everything else echoes the request."""

    seen: list[dict] = []

    def _answer(self, body: bytes) -> None:
        Upstream.seen.append({"method": self.command, "path": self.path, "key": self.headers.get("x-goog-api-key")})
        status = 400 if self.path.startswith("/gemini/fail") else 200
        payload = json.dumps({"path": self.path, "body": body.decode()}).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        self._answer(b"")

    def do_POST(self) -> None:
        self._answer(self.rfile.read(int(self.headers["Content-Length"])))

    def log_message(self, *args) -> None:
        pass


def _serve(server: ThreadingHTTPServer) -> str:
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


@pytest.fixture
def relay():
    Upstream.seen.clear()
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    server = make_server(_serve(upstream) + "/gemini", "sk-v", port=0)
    yield _serve(server)
    server.shutdown()
    upstream.shutdown()


def test_relay_adds_the_key_and_keeps_path_and_body(relay):
    listing = json.load(urllib.request.urlopen(relay + "/v1beta/models?pageSize=1000"))
    assert listing["path"] == "/gemini/v1beta/models?pageSize=1000"
    request = urllib.request.Request(
        relay + "/v1beta/models/gemini-4-carbon:streamGenerateContent?alt=sse",
        data=b'{"contents": []}',
        headers={"Content-Type": "application/json"},
    )
    assert json.load(urllib.request.urlopen(request))["body"] == '{"contents": []}'
    assert [(s["method"], s["key"]) for s in Upstream.seen] == [("GET", "sk-v"), ("POST", "sk-v")]


def test_relay_passes_upstream_errors_through(relay):
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(relay + "/fail")
    assert exc.value.code == 400 and json.load(exc.value)["path"] == "/gemini/fail"


def test_relay_answers_502_when_upstream_is_down():
    closed = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    dead = f"http://127.0.0.1:{closed.server_address[1]}"
    closed.server_close()
    server = make_server(dead, "sk-v", port=0)
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(_serve(server) + "/v1beta/models")
    server.shutdown()
    assert exc.value.code == 502
