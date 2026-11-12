"""Localhost relay between the Antigravity CLI's model gateway and the proxy's /gemini passthrough.

Runs inside a leg's sandbox as a standalone script (standard library only): `agy` with AGY_LLM_GATEWAY_URL
sends Gemini-format requests with no credential, and the relay forwards each one upstream with the leg's
virtual key. Run as a script it binds, detaches, and returns, so `relay && exec agy …` starts agy only once
the port is listening.

    SYNTH_AGY_UPSTREAM=<proxy>/gemini SYNTH_AGY_KEY=<virtual key> python3 agy_gateway.py
"""

from __future__ import annotations

import os
import sys
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# agy accepts a plain-http gateway only on localhost.
PORT = 4100
UPSTREAM_ENV = "SYNTH_AGY_UPSTREAM"
KEY_ENV = "SYNTH_AGY_KEY"
# Seconds without a byte from upstream before the request fails; a high thinking level can stay silent for minutes.
UPSTREAM_TIMEOUT = 900
LOG_FILE = os.path.expanduser("~/.agy-gateway.log")


def make_server(upstream: str, key: str, port: int = PORT) -> ThreadingHTTPServer:
    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self) -> None:
            self._relay(None)

        def do_POST(self) -> None:
            self._relay(self.rfile.read(int(self.headers.get("Content-Length", 0))))

        def _relay(self, body: bytes | None) -> None:
            headers = {"x-goog-api-key": key}
            if body is not None:
                headers["Content-Type"] = self.headers.get("Content-Type", "application/json")
            request = urllib.request.Request(upstream + self.path, data=body, method=self.command, headers=headers)
            try:
                response = urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT)
            except urllib.error.HTTPError as exc:
                response = exc
            except OSError as exc:
                self._reply(502, f"relay: upstream unreachable: {exc}".encode())
                return
            with response:
                self.send_response(response.status)
                self.send_header("Content-Type", response.headers.get("Content-Type", "application/json"))
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                while chunk := response.read(8192):
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

        def _reply(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            # Errors only: every request is already in the proxy's spend log.
            if args and str(args[1]) not in ("200", "-"):
                sys.stderr.write(f"{self.log_date_time_string()} {format % args}\n")

    return ThreadingHTTPServer(("127.0.0.1", port), Relay)


def main() -> None:
    server = make_server(os.environ[UPSTREAM_ENV], os.environ.pop(KEY_ENV))
    if os.fork():
        os._exit(0)
    os.setsid()
    log = os.open(LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    devnull = os.open(os.devnull, os.O_RDWR)
    os.dup2(devnull, 0)
    os.dup2(log, 1)
    os.dup2(log, 2)
    server.serve_forever()


if __name__ == "__main__":
    main()
