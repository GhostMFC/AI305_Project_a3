"""Evaluate checkpoints with the repository's 2x5 judge and optional release requirement."""
import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def digest(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def configure_episode_end(benchmark, *, require_released=False, no_early_stop=False):
    """Configure judging/termination before reset without modifying policy actions."""
    original_create_env = benchmark.create_env

    def create_env(*, render_cameras=False):
        env = original_create_env(render_cameras=render_cameras)
        try:
            changes = {}
            if require_released:
                changes['require_released'] = True
            if no_early_stop:
                changes['terminate_on_success'] = False
            env.task_config = replace(env.task_config, **changes)
        except Exception:
            env.close()
            raise
        return env

    benchmark.create_env = create_env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['smolvla', 'pi05', 'expert'], required=True)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--seed-start', type=int, default=22000)
    p.add_argument('--max-steps', type=int, default=500)
    p.add_argument('--horizon', type=int, default=50)
    p.add_argument('--flow-steps', type=int)
    p.add_argument('--inference-seed', type=int, default=1000)
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    p.add_argument('--profile', choices=['basic', 'medium', 'advanced'], default='basic')
    p.add_argument('--diagnostic', action='store_true')
    p.add_argument('--require-released', action='store_true',
                   help='Require target cookies to be near the floor and free of left-finger contact; does not command release.')
    p.add_argument('--no-early-stop', action='store_true',
                   help='Continue policy actions after success until max-steps (safety/policy stops still apply); score the final state.')
    a = p.parse_args()
    if min(a.episodes, a.max_steps, a.horizon) < 1:
        p.error('episodes, max-steps and horizon must be positive')
    if a.model != 'expert' and a.checkpoint is None:
        p.error('--checkpoint required for learned policies')
    a.flow_steps = a.flow_steps or (25 if a.model == 'smolvla' else 10)
    if a.flow_steps < 1:
        p.error('flow-steps must be positive')
    a.out = a.out.resolve()
    a.out.mkdir(parents=True, exist_ok=False)
    from a3_dual_arm_sim.workflows.benchmark import make_cookie_benchmark
    from a3_dual_arm_sim.sim.randomization import load_randomization_config
    from a3_dual_arm_sim.workflows.policy_adapters import make_policy_adapter
    from teacher_policy import TeacherPolicy, PI_PROMPT, SMOL_PROMPT
    import a3_dual_arm_sim
    settings = load_randomization_config(None)
    settings.update(profile=a.profile, source_column='1')
    benchmark = make_cookie_benchmark('2x5', prompt_style='training',
        profile=a.profile, source_column='1', randomization_settings=settings,
        max_steps=a.max_steps, randomize_boxes=True, randomize_cookies=True,
        success_hold_steps=5, slot_tolerance_m=(0.015, 0.005), wall_contact_tolerance_m=0.030)
    if a.require_released or a.no_early_stop:
        configure_episode_end(benchmark, require_released=a.require_released,
                              no_early_stop=a.no_early_stop)
    prompt = PI_PROMPT if a.model == 'pi05' else SMOL_PROMPT
    # Record the actual instruction in both episode metadata and diagnostic traces.
    benchmark._episode_task = lambda column, seed: prompt
    if a.diagnostic:
        benchmark.diagnostic_dir = a.out / 'trace'
        benchmark.diagnostic_dir.mkdir()
    record = dict(arguments=vars(a), prompt=prompt, python=sys.version,
                  source_module=a3_dual_arm_sim.__file__, randomization=settings,
                  task='2x5', source_column=1, success_hold_steps=5,
                  require_released=a.require_released,
                  terminate_on_success=not a.no_early_stop,
                  score_semantics='final_state',
                  slot_tolerance_m=[0.015, 0.005], wall_contact_tolerance_m=0.030,
                  script_sha256={f.name:digest(f) for f in Path(__file__).parent.glob('*.py')})
    for name, command in [('git_commit',['git','rev-parse','HEAD']),
                          ('git_diff',['git','diff','HEAD']),
                          ('pip_freeze',[sys.executable,'-m','pip','freeze'])]:
        r = subprocess.run(command, capture_output=True, text=True)
        record[name] = r.stdout if r.returncode == 0 else r.stderr
    if a.checkpoint:
        record['weights_sha256'] = digest(a.checkpoint / 'model.safetensors')
        record['checkpoint_files_sha256'] = {f.name:digest(f) for f in a.checkpoint.iterdir()
            if f.is_file() and f.name != 'model.safetensors'}
    def save(name, obj):
        (a.out / name).write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str)+'\n', encoding='utf-8')
    save('run.json', record)
    policy = None
    start = time.perf_counter()
    try:
        policy = make_policy_adapter('same_column') if a.model == 'expert' else TeacherPolicy(
            a.model, a.checkpoint, a.device, a.horizon, a.flow_steps, a.inference_seed)
        record['load_seconds'] = time.perf_counter() - start
        if a.model != 'expert':
            record['effective_policy_config'] = policy.effective_config
        save('run.json', record)
        def on_episode(ep):
            with (a.out / 'episodes.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(ep.to_dict(), ensure_ascii=False)+'\n')
        result = benchmark.evaluate(policy, num_episodes=a.episodes, seed_start=a.seed_start,
                                    workers=1, on_episode_end=on_episode)
        report = result.to_dict()
        report['run'] = record
        if a.model != 'expert':
            import numpy as np
            times = policy.timing
            report['policy_call_timing'] = dict(count=len(times), mean_seconds=float(np.mean(times)),
                p95_seconds=float(np.percentile(times,95)),
                meaning='preprocess+select_action+postprocess; includes queued actions, not fresh-chunk-only latency')
        save('result.json', report)
        print(result.summary_table())
    except Exception as exc:
        save('error.json', dict(error_type=type(exc).__name__, error=str(exc),
                               completed=False, elapsed_seconds=time.perf_counter()-start))
        raise
    finally:
        if policy is not None and hasattr(policy, 'close'):
            policy.close()


if __name__ == '__main__':
    main()
