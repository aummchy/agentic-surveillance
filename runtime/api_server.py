"""Dashboard API server lifecycle: asyncio loop + Uvicorn on a daemon thread.

Two-phase stop mirrors the original shutdown sequence: request_exit() flips
Uvicorn's should_exit early (serving stops while the track queue drains),
finalize() joins the thread and closes the loop at the very end.
"""

import asyncio
import threading

import structlog

logger = structlog.get_logger(__name__)

HOST = "0.0.0.0"
PORT = 8000


class ApiServer:
    def __init__(self):
        self.loop: asyncio.AbstractEventLoop | None = None
        self._server = None
        self._thread: threading.Thread | None = None

    def start(self):
        import uvicorn
        from dashboard.backend.main import app

        self.loop = asyncio.new_event_loop()

        config = uvicorn.Config(app, host=HOST, port=PORT, log_level="warning", access_log=False)
        self._server = uvicorn.Server(config)

        def run_server():
            self.loop.run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=run_server, daemon=True)
        self._thread.start()
        logger.info("dashboard_api_started", url=f"http://localhost:{PORT}")

    def request_exit(self):
        if self._server is not None:
            self._server.should_exit = True

    def finalize(self, timeout: float = 5):
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        if self.loop is not None:
            try:
                self.loop.close()
            except Exception:
                pass
