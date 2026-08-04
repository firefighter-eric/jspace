"""Minimal local HTTP API for J-Space Observatory."""

from __future__ import annotations

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from jspace_server.runtime import AnalysisRuntime

RUNTIME = AnalysisRuntime()
ALLOWED_ORIGIN = "http://127.0.0.1:5174"
ANALYSIS_GATE = threading.BoundedSemaphore(1)


def _origin_allowed(origin: str | None) -> bool:
    """Allow direct local clients and the fixed Vite development origin."""
    return origin is None or origin == ALLOWED_ORIGIN


def _is_json_content_type(value: str | None) -> bool:
    return (
        value is not None
        and value.partition(";")[0].strip().lower() == "application/json"
    )


class ObservatoryHandler(BaseHTTPRequestHandler):
    server_version = "JSpaceObservatory/0.1"

    def _send_json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
            "utf-8"
        )
        try:
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            # The browser may abort a fetch when navigating away. The model
            # run cannot be interrupted safely, but the server should remain
            # quiet and healthy when the response is no longer wanted.
            return

    def do_OPTIONS(self) -> None:  # noqa: N802
        if not _origin_allowed(self.headers.get("Origin")):
            self._send_json({"error": "origin not allowed"}, HTTPStatus.FORBIDDEN)
            return
        self.send_response(HTTPStatus.NO_CONTENT)
        self.send_header("Access-Control-Allow-Origin", ALLOWED_ORIGIN)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            self._send_json(RUNTIME.status())
            return
        self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/analyze":
            self._send_json({"error": "not found"}, HTTPStatus.NOT_FOUND)
            return
        if not _origin_allowed(self.headers.get("Origin")):
            self._send_json({"error": "origin not allowed"}, HTTPStatus.FORBIDDEN)
            return
        if not _is_json_content_type(self.headers.get("Content-Type")):
            self._send_json(
                {"error": "Content-Type must be application/json"},
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
            )
            return
        if not ANALYSIS_GATE.acquire(blocking=False):
            self._send_json(
                {"error": "an analysis is already running; try again shortly"},
                HTTPStatus.TOO_MANY_REQUESTS,
            )
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 64 * 1024:
                raise ValueError("invalid request body size")
            payload = json.loads(self.rfile.read(length))
            if not isinstance(payload, dict):
                raise ValueError("request body must be an object")
            result = RUNTIME.analyze(
                str(payload.get("prompt", "")),
                top_k=int(payload.get("top_k", 8)),
                max_tokens=int(payload.get("max_tokens", 64)),
            )
            self._send_json(result)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
        except Exception as exc:
            self._send_json(
                {"error": f"{type(exc).__name__}: {exc}"},
                HTTPStatus.INTERNAL_SERVER_ERROR,
            )
        finally:
            ANALYSIS_GATE.release()

    def log_message(self, message: str, *args: object) -> None:
        print(f"[jspace-api] {self.address_string()} {message % args}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8765, type=int)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), ObservatoryHandler)
    print(f"J-Space API listening on http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
