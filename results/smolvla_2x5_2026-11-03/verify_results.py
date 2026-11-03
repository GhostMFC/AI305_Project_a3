"""Verify published metrics against episode records; no simulator or GPU required."""
import csv
import json
import math
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
MODELS = ['teacher', 'ft2000', 'ft6000', 'ft16000', 'ft40000']
SEEDS = list(range(22000, 22010))

def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))

def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def main():
    rows = read_csv(ROOT / 'data/episodes.csv')
    pairs = [(r['model'], int(r['seed'])) for r in rows]
    assert len(pairs) == len(set(pairs)) == 50
    assert set(pairs) == {(model, seed) for model in MODELS for seed in SEEDS}
    for r in rows:
        assert r['success'] in ('True', 'False')
        r.update(seed=int(r['seed']), score=int(r['score']),
                 success=r['success'] == 'True', steps=int(r['steps']))
    historical = read_json(ROOT / 'records/teacher/historical_final_result.json')
    teacher_episodes = {e['seed']: e for e in historical['episodes']}
    provenance = read_json(ROOT / 'provenance/teacher_baseline.json')
    teacher_details = {e['seed']: e for e in provenance['teacher_episode_details']}
    for r in rows:
        if r['model'] == 'teacher':
            d = teacher_details[r['seed']]
            first = d['first_success_step']
            e = teacher_episodes[r['seed']]
            assert d['trace_rows'] == e['steps'] == 1000
            assert d['historical_final_score'] == e['cookies_in_target']
            assert r['source'] == 'historical_trace_first_success'
            assert r['wall_seconds'] == ''
            assert (r['success'], r['steps'], r['score']) == (
                first is not None, first if first is not None else e['steps'],
                10 if first is not None else e['cookies_in_target'])
            continue
        record = read_json(ROOT / 'records' / r['model'] / f"seed_{r['seed']}.json")
        assert len(record['episodes']) == 1
        ep = record['episodes'][0]
        assert (r['seed'], r['score'], r['success'], r['steps']) == (
            ep['seed'], ep['cookies_in_target'], ep['success'], ep['steps'])
        assert float(r['wall_seconds']) == ep['wall_seconds']
        assert r['source'] == 'direct_stop_on_success'
        args = record['run']['arguments']
        for key, value in dict(model='smolvla', episodes=1, seed_start=r['seed'],
                               max_steps=1000, horizon=50, flow_steps=25,
                               inference_seed=1000, profile='basic',
                               require_released=True, no_early_stop=False).items():
            assert args[key] == value, (r['model'], r['seed'], key)
    summary = read_csv(ROOT / 'data/summary.csv')
    assert len(summary) == 5
    assert {r['model'] for r in summary} == set(MODELS)
    for item in summary:
        group = [r for r in rows if r['model'] == item['model']]
        wins = [r for r in group if r['success']]
        expected = dict(paired_episodes=len(group), successes=len(wins),
                        success_rate_percent=100 * len(wins) / len(group),
                        mean_score=mean(r['score'] for r in group),
                        mean_success_steps=mean(r['steps'] for r in wins))
        for key, value in expected.items():
            assert math.isclose(float(item[key]), value, abs_tol=1e-10), (item['model'], key)
        print(f"{item['model']:>8}: {len(wins)}/10 success, mean score {expected['mean_score']:.1f}/10")
    paired = read_csv(ROOT / 'data/comparison_by_seed.csv')
    assert sorted(int(r['seed']) for r in paired) == SEEDS
    index = {(r['model'], r['seed']): r for r in rows}
    for row in paired:
        for model in MODELS:
            expected = index[model, int(row['seed'])]
            for key in ['score', 'steps']:
                assert int(row[f'{model}_{key}']) == expected[key]
            assert row[f'{model}_success'] == str(expected['success'])
    print('PASS: 50 model/seed rows, 40 direct results, teacher provenance, summaries and paired table agree.')
    print('Teacher full trajectories are not included; this checks their documented reconstruction, not a fresh replay.')

if __name__ == '__main__':
    main()
