import os
import sys
import types
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("DATA_DIR", "/tmp/bipod-test-data")

try:
    import faiss  # noqa: F401
except ModuleNotFoundError:
    fake_faiss = types.ModuleType("faiss")

    class _FakeIndexFlatL2:
        def __init__(self, dim):
            self.d = dim
            self.ntotal = 0

        def add(self, vecs):
            self.ntotal += len(vecs)

        def search(self, vec, k):
            return [[999.0] * k], [[-1] * k]

    def _fake_read_index(path):
        return _FakeIndexFlatL2(768)

    def _fake_write_index(index, path):
        return None

    fake_faiss.IndexFlatL2 = _FakeIndexFlatL2
    fake_faiss.read_index = _fake_read_index
    fake_faiss.write_index = _fake_write_index
    sys.modules["faiss"] = fake_faiss

try:
    import pypdf  # noqa: F401
except ModuleNotFoundError:
    fake_pypdf = types.ModuleType("pypdf")

    class _FakePage:
        def extract_text(self):
            return ""

    class _FakePdfReader:
        def __init__(self, *args, **kwargs):
            self.pages = [_FakePage()]

    fake_pypdf.PdfReader = _FakePdfReader
    sys.modules["pypdf"] = fake_pypdf
