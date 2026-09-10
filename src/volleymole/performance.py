"""Thread-safe host timings. Overlapping intervals must not be added as wall time."""
from contextlib import contextmanager
from threading import Lock
from time import perf_counter


class Timings:
    def __init__(self):
        self.started = perf_counter()
        self.lock = Lock()
        self.stats = {}

    def add(self, name, seconds):
        with self.lock:
            row = self.stats.setdefault(name, {'calls': 0, 'wall_sec': 0.0})
            row['calls'] += 1
            row['wall_sec'] += seconds

    @contextmanager
    def measure(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self.add(name, perf_counter() - start)

    def report(self):
        with self.lock:
            return {'elapsed_sec': round(perf_counter() - self.started, 6),
                    'timings': {k: {'calls': v['calls'], 'wall_sec': round(v['wall_sec'], 6)}
                                for k, v in self.stats.items()},
                    'note': 'Host durations; overlapping stages are not additive. CUDA transfer submissions '
                            'are not kernel timings. ORT execution includes transfers unless separately bound.'}
