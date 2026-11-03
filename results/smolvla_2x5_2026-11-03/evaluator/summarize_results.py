"""Create small factual summaries, rejecting missing/incomplete runs."""
import argparse
from collections import Counter
import csv
import json
import math
from pathlib import Path
import statistics


def summarize(path):
    r = json.loads(path.read_text(encoding='utf-8'))
    episodes = r['episodes']
    n = r['summary']['total_episodes']
    assert n > 0 and len(episodes) == n, 'Incomplete episode list'
    seeds = [e['seed'] for e in episodes]
    assert len(set(seeds)) == n, 'Duplicate seeds'
    args = r.get('run', {}).get('arguments', {})
    if args:
        assert args['episodes'] == n
        assert seeds == list(range(args['seed_start'], args['seed_start']+n))
    values = [e['cookies_in_target'] for e in episodes]
    assert all(isinstance(e['success'], bool) for e in episodes)
    assert all(0 <= e['cookies_in_target'] <= e['max_score'] for e in episodes)
    k = sum(e['success'] for e in episodes)
    # Wilson interval for binary episode success, not for average cookie count.
    z, phat = 1.96, k/n
    d = 1 + z*z/n
    center = (phat + z*z/(2*n))/d
    half = z*math.sqrt(phat*(1-phat)/n + z*z/(4*n*n))/d
    summary = dict(file=str(path), episodes=n, successful_episodes=k, success_rate=k/n,
        success_wilson95=[center-half, center+half], mean_final_cookies=statistics.mean(values),
        mean_steps=statistics.mean(e['steps'] for e in episodes),
        mean_wall_seconds=statistics.mean(e['wall_seconds'] for e in episodes),
        distribution=dict(Counter(values)), failed_seeds=[e['seed'] for e in episodes if not e['success']],
        failure_reason_counts=dict(Counter(str(e.get('failure_reason')) for e in episodes if not e['success'])),
        arguments=args, actual_prompt=r.get('run',{}).get('prompt'),
        warning='Counts do not establish failure mechanism; inspect video/trace. Wall time is not model-only latency.')
    return summary, episodes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result', type=Path)
    a = p.parse_args()
    summary, episodes = summarize(a.result)
    out = a.result.parent
    (out/'analysis_summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    fields = ['seed','cookies_in_target','max_score','success','steps','wall_seconds','failure_reason']
    with (out/'episode_scores.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore')
        writer.writeheader(); writer.writerows(episodes)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
