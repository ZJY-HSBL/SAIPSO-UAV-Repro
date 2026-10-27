import csv
from saipso_uav.route3d_study import run, ARMS, SCENES
from saipso_uav.route3d import ENCODINGS, INITIALIZATIONS
from saipso_uav.model import WEIGHTS


def test_route3d_reproduction_smoke(tmp_path):
    m = run(tmp_path, seeds=[123456], budget=20, make_plots=False)
    count = len(SCENES)*len(WEIGHTS)*len(INITIALIZATIONS)*len(ENCODINGS)*len(ARMS)
    assert m['total_runs'] == count == 48
    with (tmp_path/'metrics.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == count
    assert all(int(r['evaluations'])==20 for r in rows)
    assert all(r['audited_feasible'] in ('True', 'False') for r in rows)
    assert len(m['paired_contrasts']) == len(SCENES)*len(WEIGHTS)*len(INITIALIZATIONS)*len(ARMS)
