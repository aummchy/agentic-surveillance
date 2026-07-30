import threading
import time
import structlog

logger = structlog.get_logger(__name__)


class TimingCollector:
    """Thread-safe collector for recognition timing diagnostics.

    Encapsulates submit/start/done tracking with locking,
    keeping timing state out of CameraAgent.
    """

    def __init__(self):
        self._submit_times: dict = {}
        self._submit_lock = threading.Lock()
        self._start_times: dict = {}
        self._start_lock = threading.Lock()

    def record_submit(self, track_id: str) -> float:
        submit_time = time.perf_counter()
        with self._submit_lock:
            self._submit_times[track_id] = submit_time
        return submit_time

    def pop_submit(self, track_id: str):
        with self._submit_lock:
            return self._submit_times.pop(track_id, None)

    def submit_pending_count(self) -> int:
        with self._submit_lock:
            return len(self._submit_times)

    def record_start(self, track_id: str):
        with self._start_lock:
            self._start_times[track_id] = time.perf_counter()

    def pop_start(self, track_id: str):
        with self._start_lock:
            return self._start_times.pop(track_id, None)

    def get_duration_ms(self, track_id: str):
        start = self.pop_start(track_id)
        if start is None:
            return None
        return round((time.perf_counter() - start) * 1000, 1)