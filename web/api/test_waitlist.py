"""Stdlib unittest suite for web/api/waitlist.py (importable module)."""
import importlib.util
import json
import sys
import tempfile
import threading
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "waitlist", Path(__file__).resolve().parent / "waitlist.py"
)
wl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wl)

import unittest


class SaveEmailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        wl.DATA_DIR = Path(self.tmp.name)
        wl.WAITLIST_CSV = wl.DATA_DIR / "waitlist.csv"

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_and_duplicate(self):
        self.assertTrue(wl.save_email("a@b.io"))
        self.assertFalse(wl.save_email("A@b.IO"))  # case-insensitive dup
        rows = wl.WAITLIST_CSV.read_text().strip().splitlines()
        self.assertEqual(len(rows), 2)  # header + one row
        self.assertTrue(rows[0].startswith("email,ts_utc,source"))

    def test_formula_injection_neutralized(self):
        for evil in ["=2+5@a.io", "+2+5@a.io", "-2+3@a.io", "=HYPERLINK(\"http://evil.example\")@a.io"]:
            self.assertTrue(wl.EMAIL_RE.match(evil), evil)
            wl.save_email(evil)
        content = wl.WAITLIST_CSV.read_text()
        for line in content.splitlines()[1:]:
            first_field = line.split(",", 1)[0]
            self.assertTrue(first_field.startswith("\"'") or first_field.startswith("'"),
                            f"not neutralized: {line}")

    def test_concurrent_writes_no_dup_no_double_header(self):
        results = []
        def worker():
            results.append(wl.save_email("race@test.io"))
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        self.assertEqual(results.count(True), 1, f"expected exactly one True, got {results}")
        lines = wl.WAITLIST_CSV.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(sum(1 for l in lines if l.startswith("email,")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
