from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks/e2e"))
from gpu_ownership import Ownership


class OwnershipTest(unittest.TestCase):
    def test_exit_race_does_not_hide_unknown_or_reused_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            stat = proc / '100/stat'
            stat.parent.mkdir()
            stat.write_text('100 (worker name (child)) S ' + '0 ' * 18 + '1234 0')
            owner = Ownership(proc)
            owner.remember({100})
            self.assertEqual(owner.starts[100], '1234')
            stat.unlink()
            self.assertEqual(owner.foreign({100, 200}, set()), {200})
            stat.write_text('100 (another process) S ' + '0 ' * 18 + '9999 0')
            self.assertEqual(owner.foreign({100, 200}, set()), {100, 200})
            self.assertEqual(owner.foreign({100, 200}, {100}), {200})


if __name__ == '__main__':
    unittest.main()
