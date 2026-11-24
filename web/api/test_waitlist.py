"""Stdlib unittest suite for web/api/waitlist.py (importable module)."""
import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "waitlist", Path(__file__).resolve().parent / "waitlist.py"
)
wl = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(wl)


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

    def test_direct_save_formula_prefixes(self):
        # every FORMULA_PREFIXES char must be neutralized even for direct callers
        for ch in wl.FORMULA_PREFIXES:
            addr = ch + "x@a.io"
            if not wl.EMAIL_RE.match(addr):
                continue  # tab/CR are whitespace and can't appear in a valid email anyway
            wl.save_email(addr)
            first = wl.WAITLIST_CSV.read_text().splitlines()[-1].split(",", 1)[0]
            self.assertTrue(first.startswith("\"'") or first.startswith("'"),
                            f"not neutralized: {addr} -> {first}")

    def test_concurrent_writes_no_dup_no_double_header(self):
        results = []
        def worker():
            results.append(wl.save_email("race@test.io"))
        threads = [threading.Thread(target=worker) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(results.count(True), 1, f"expected exactly one True, got {results}")
        lines = wl.WAITLIST_CSV.read_text().strip().splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(sum(1 for line in lines if line.startswith("email,")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
