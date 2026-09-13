"""Track observed Docker PIDs across non-atomic KFD/process teardown snapshots."""
from pathlib import Path


class Ownership:
    def __init__(self, proc=Path('/proc')):
        self.proc = proc
        self.starts = {}

    def start(self, pid):
        try:
            # comm can contain spaces and parentheses. Field 22 is starttime.
            return (self.proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]
        except (FileNotFoundError, ProcessLookupError):
            return None

    def remember(self, owned):
        for pid in owned:
            start = self.start(pid)
            if start is not None:
                self.starts[pid] = start

    def foreign(self, holders, current_owned):
        result = set()
        for pid in holders - current_owned:
            # A known owned process may already have exited while its KFD entry
            # is disappearing. Reject an unknown PID or a reused live PID.
            live_start = self.start(pid)
            if pid not in self.starts or (live_start is not None and live_start != self.starts[pid]):
                result.add(pid)
        return result
