import json
import re
from pathlib import Path

import pytest

from recsys import report


def test_readme_block_is_exactly_the_rendering_of_results_json():
    if not report.RES.exists():
        pytest.skip("results.json not generated yet")
    text = report.README.read_text()
    m = re.search(re.escape(report.BEGIN) + r"\n(.*?)\n" + re.escape(report.END), text, re.S)
    assert m, "README is missing the generated results block"
    assert m.group(1) == report.render(json.loads(report.RES.read_text()))


def test_no_dataset_files_tracked():
    root = Path(report.ROOT)
    assert not list(root.glob("**/ratings.dat")) or (root / "data").exists()
