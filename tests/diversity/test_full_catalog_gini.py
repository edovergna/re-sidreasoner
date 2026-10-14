"""Exercise the metric code shipped in the SIDReasoner working-tree patch.

The baseline submodule need not be downloaded for these CPU-only tests.
"""
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

PATCH = Path(__file__).resolve().parents[2] / 'patches/common/sidreasoner-local.patch'
FILES = ['calc.py', 'utils/summarize_item_side_diversity.py',
         'utils/make_rq5_diversity_tables.py']


def added_source(filename):
    text = PATCH.read_text()
    block = text.split(f'diff --git a/{filename} b/{filename}\n', 1)[1]
    block = block.split('\ndiff --git ', 1)[0]
    return '\n'.join(line[1:] for line in block.splitlines()
                     if line.startswith('+') and not line.startswith('+++')) + '\n'


def load_helper(filename):
    namespace = {'__name__': 'metric_under_test'}
    exec(compile(added_source(filename), filename, 'exec'), namespace)
    return namespace


@pytest.fixture(params=FILES)
def gini(request):
    # calc.py is modified rather than added in the patch; extract its complete
    # added function. The two new helpers can also be tested through their APIs.
    source = added_source(request.param)
    function = re.search(r'def _gini\(.*?(?=\n\ndef |\Z)', source, re.S).group(0)
    namespace = {'np': np}
    exec(compile(function, request.param, 'exec'), namespace)
    return namespace['_gini']


def test_unexposed_items_count(gini):
    assert gini(Counter({i: 1 for i in range(10)}), 100) == pytest.approx(0.9)


def test_full_coverage_uniform(gini):
    assert gini(Counter({i: 7 for i in range(10)}), 10) == pytest.approx(0)


def test_one_exposed_item(gini):
    assert gini(Counter({'item': 20}), 100) == pytest.approx(0.99)


def test_empty_exposure(gini):
    assert gini(Counter(), 100) == 0
    assert gini(Counter(), 0) == 0
    assert gini(Counter({'item': 0}), 100) == 0


def test_matches_pairwise_definition(gini):
    rng = np.random.default_rng(42)
    for size in (1, 7, 100):
        exposures = rng.integers(0, 8, size=size).astype(float)
        exposures[:size // 2] = 0
        expected = (np.abs(exposures[:, None] - exposures).sum()
                    / (2 * size * exposures.sum())) if exposures.sum() else 0
        counts = Counter({i: int(x) for i, x in enumerate(exposures) if x})
        assert gini(counts, size) == pytest.approx(expected)


def test_invalid_population_rejected(gini):
    with pytest.raises(ValueError):
        gini(Counter({'a': 1, 'b': 1}), 1)
    with pytest.raises(ValueError):
        gini(Counter({'a': -1}), 2)


def test_json_metrics_and_table_use_rl_predictions(tmp_path, monkeypatch):
    table = load_helper(FILES[2])
    summary = load_helper(FILES[1])
    monkeypatch.chdir(tmp_path)
    info = tmp_path / 'data/Amazon/info/Video_Games_5_2016-10-2018-11.txt'
    info.parent.mkdir(parents=True)
    info.write_text(''.join(f'item{i}\ttitle{i}\t{i}\n' for i in range(100)))
    root = tmp_path / 'results'
    for stage in ('sft', 'ra', 'rl'):
        for decoder in ('beam10', 'sample_t12_p095', 'sample_t10_k50'):
            for seed in (42, 43, 44):
                path = root / 'Video_Games' / stage / f'{decoder}_seed{seed}' / 'final_result_thinking_Video_Games.json'
                path.parent.mkdir(parents=True)
                preds = ['item0'] * 10 if stage == 'rl' else [f'item{i}' for i in range(10)]
                path.write_text(json.dumps([{'predict': preds, 'output': 'item0'}]))
    metrics = summary['_metrics'](path, info)
    assert metrics['Gini@10'] == pytest.approx(0.99)
    assert metrics['HR@10'] == 1
    out = tmp_path / 'table.md'
    table['build_table']('Video_Games', root, out)
    rows = out.read_text().splitlines()
    rl_rows = [r for r in rows if r.startswith('| SFT + RA + RL |')]
    assert len(rl_rows) == 3
    assert all('0.990000 +/- 0.000000' in r for r in rl_rows)
    assert all('0.900000 +/- 0.000000' in r for r in rows if r.startswith('| SFT |'))
    # Missing RL outputs must fail instead of falling back to old aggregates.
    path.unlink()
    with pytest.raises(FileNotFoundError):
        table['build_table']('Video_Games', root, out)
