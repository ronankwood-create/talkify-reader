"""
Async jobs the browser polls: opening a PDF, exporting an MP3.

Both used to be a bare dict inside the global STATE, mutated from a worker
thread and read from an HTTP thread with nothing in between. That is a real
race, not a theoretical one - _run_export writes job["done"] on every block
while /api/export/status iterates the same dict to build its JSON response,
and a dict resized mid-iteration raises.

So: one lock, and get() hands back a copy. Callers can do whatever they like
with what they are given without holding anything.
"""
from __future__ import annotations

import threading
import uuid


class JobTracker:
    def __init__(self, kind="job"):
        self.kind = kind
        self._lock = threading.Lock()
        self._jobs = {}

    def create(self, **fields):
        job_id = uuid.uuid4().hex[:12]
        with self._lock:
            self._jobs[job_id] = dict(fields)
        return job_id

    def get(self, job_id):
        """A snapshot, or None. Never the live dict."""
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job is not None else None

    def update(self, job_id, **fields):
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            job.update(fields)
            return dict(job)
