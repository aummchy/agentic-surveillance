"""Track queue + its consumer threads: the handoff between the camera path
and finalization.

The camera side enqueues finished tracks (via TrackProcessor.enqueue with
.workers.queue); num_workers consumers pull and run the late-bound
process_fn. The process function is bound at start() because the queue
consumers start before TrackProcessor is constructed — exactly the
original ordering from main.py.
"""

import queue
import threading
from typing import Callable

import structlog

from pipeline.models import Track

logger = structlog.get_logger(__name__)

NUM_WORKERS = 2


class TrackWorkers:
    def __init__(self, num_workers: int = NUM_WORKERS):
        self.queue: queue.Queue = queue.Queue()
        self._num_workers = num_workers
        self._shutdown_event = threading.Event()
        self._process_fn: Callable[[Track], None] | None = None
        self._threads: list[threading.Thread] = []

    def start(self, process_fn: Callable[[Track], None]):
        self._process_fn = process_fn
        for _ in range(self._num_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self._threads.append(t)

    def stop(self):
        self._shutdown_event.set()

    def drain(self):
        try:
            self.queue.join()
        except Exception:
            pass

    def _worker_loop(self):
        while not self._shutdown_event.is_set():
            track = None
            try:
                track = self.queue.get(timeout=1.0)
                self._process_fn(track)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error("worker_failed", track_id=track.track_id if track else None,
                             error=str(e), exc_info=True)
            finally:
                if track is not None:
                    self.queue.task_done()
