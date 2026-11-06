"""
Bulk evaluation of multiple checkpoints listed in a json file.

Each checkpoint is benchmarked by running `main.py --mode benchmark` in its own
subprocess (fresh ray instance per checkpoint), then a summary csv is built
from the per-checkpoint detail files.

Entries with a "planner" key instead of "path" are OMPL baselines, run through
`benchmark.py` (BITstar, RRTConnect (= BiRRT), RRT, PRM, LBKPIECE). They use
"timeout" (per-task planning budget in seconds, default 30) and need "config".

Usage:
    python -m helper.bulk_evaluate configs/bulk_evaluate.json

Checkpoint list format (per-checkpoint keys override the top-level defaults):
    {
        "tasks_path": "tasks/obstacle_evaluate",
        "config": "configs/evaluate.json",
        "num_processes": 10,
        "max_time": null,
        "checkpoints": [
            "runs/obstacle_v9/ckpt_multiarm_motion_planner_00370",
            {
                "name": "v5_992_dense",
                "path": "runs/obstacle_v5/ckpt_multiarm_motion_planner_00992",
                "tasks_path": "tasks/obstacle_evaluate_dense"
            },
            {"planner": "BITstar", "timeout": 30},
            {"name": "ompl_birrt", "planner": "RRTConnect"}
        ]
    }

Outputs (STAMP = run timestamp, e.g. 20261017_153000):
    benchmark/bulk/STAMP/STAMP_summary.csv       one row per checkpoint
    benchmark/bulk/STAMP/STAMP_<name>.csv/.pkl   detail file per checkpoint
    benchmark/bulk/STAMP/STAMP_checkpoints.json  copy of the checkpoint list
"""

import csv
import pickle
import subprocess
import sys
from argparse import ArgumentParser
from datetime import datetime
from json import load, dump
from os import makedirs
from os.path import exists, join
from pathlib import Path
from time import time

import numpy as np


DEFAULTS = {
    'tasks_path': None,
    'config': None,
    'num_processes': 10,
    'max_time': None,
    'gui': False,
    'timeout': 30.0,
}

OMPL_PLANNERS = ['RRTConnect', 'RRT', 'BITstar', 'PRM', 'LBKPIECE']
OMPL_DEFAULT_NAMES = {'RRTConnect': 'ompl_birrt', 'BITstar': 'ompl_bitstar'}

SUMMARY_FIELDS = [
    'run_timestamp',
    'name',
    'method',
    'checkpoint',
    'config',
    'tasks_path',
    'status',
    'task_count',
    'evaluated_count',
    'success_count',
    'success_rate',
    'collision_count',
    'collision_rate',
    'other_failure_count',
    'avg_path_length',
    'avg_path_length_success',
    'avg_total_joint_motion',
    'avg_planning_time',
    'avg_planning_time_success',
    'avg_time_to_success',
    'avg_steps_to_success',
    'avg_mean_pos_residual',
    'avg_mean_orn_residual',
    'success_rate_1_arm',
    'success_rate_2_arm',
    'success_rate_3_arm',
    'success_rate_4_arm',
    'wall_time_sec',
    'detail_file',
]


def parse_args():
    parser = ArgumentParser("Bulk Checkpoint Evaluation")
    parser.add_argument('checkpoint_list', type=str,
                        help='path to json file listing checkpoints')
    parser.add_argument('--output_dir', type=str, default='benchmark/bulk',
                        help='root directory for bulk evaluation outputs,' +
                        ' must be inside benchmark/ (where Logger saves)')
    return parser.parse_args()


def default_name(checkpoint_path):
    # runs/obstacle_v9/ckpt_multiarm_motion_planner_00370 -> obstacle_v9_00370
    path = Path(checkpoint_path)
    parts = path.parts
    run_name = parts[parts.index('runs') + 1] if 'runs' in parts[:-1]\
        else path.parent.name
    return '{}_{}'.format(run_name, path.name[-5:])


def load_jobs(checkpoint_list_path):
    spec = load(open(checkpoint_list_path))
    defaults = dict(DEFAULTS)
    defaults.update({k: v for k, v in spec.items() if k != 'checkpoints'})

    jobs = []
    names = set()
    for entry in spec['checkpoints']:
        if isinstance(entry, str):
            entry = {'path': entry}
        job = dict(defaults)
        job.update(entry)
        job.setdefault('planner', None)
        job.setdefault('path', None)
        label = job['planner'] or job['path']
        if job['planner'] is not None:
            if job['planner'] not in OMPL_PLANNERS:
                print('[BulkEvaluate] Unknown OMPL planner', job['planner'],
                      '- choose from', OMPL_PLANNERS)
                exit(1)
            if job['config'] is None:
                print('[BulkEvaluate] OMPL planner', job['planner'],
                      'needs a config')
                exit(1)
        elif job['path'] is None:
            print('[BulkEvaluate] Entry needs "path" or "planner":', entry)
            exit(1)
        if job['tasks_path'] is None:
            print('[BulkEvaluate] No tasks_path for', label)
            exit(1)
        if not job.get('name'):
            job['name'] = default_name(job['path']) if job['planner'] is None                else OMPL_DEFAULT_NAMES.get(
                    job['planner'], 'ompl_' + job['planner'].lower())
        # Keep detail file names unique within a run
        base_name, i = job['name'], 2
        while job['name'] in names:
            job['name'] = '{}_{}'.format(base_name, i)
            i += 1
        names.add(job['name'])
        jobs.append(job)
    return jobs


def count_tasks(tasks_path):
    return len([f for f in Path(tasks_path).rglob('*.json')
                if 'config' not in str(f)])


def run_benchmark(job, benchmark_name):
    if job['planner'] is not None:
        return run_ompl_benchmark(job, benchmark_name)
    command = [
        sys.executable, 'main.py',
        '--mode', 'benchmark',
        '--name', benchmark_name,
        '--load', job['path'],
        '--tasks_path', job['tasks_path'],
        '--num_processes', str(job['num_processes']),
    ]
    if job['config'] is not None:
        command.extend(['--config', job['config']])
    if job['max_time'] is not None:
        command.extend(['--max_time', str(job['max_time'])])
    if job['gui']:
        command.append('--gui')
    print('[BulkEvaluate] Running:', ' '.join(command), flush=True)
    return subprocess.call(command)


def run_ompl_benchmark(job, benchmark_name):
    # benchmark.py's Logger writes benchmark/<name>.pkl/.csv like main.py
    command = [
        sys.executable, '-m', 'helper.benchmark',
        '--planner', job['planner'],
        '--name', benchmark_name,
        '--config', job['config'],
        '--tasks_path', job['tasks_path'],
        '--timeout', str(job['timeout']),
    ]
    if job['gui']:
        command.append('--gui')
    print('[BulkEvaluate] Running:', ' '.join(command), flush=True)
    return subprocess.call(command)


def mean_of(scores, key):
    vals = [float(score[key]) for score in scores
            if score.get(key) is not None]
    return float(np.mean(vals)) if vals else None


def summarize(scores):
    successes = [s for s in scores if s['success'] > 0]
    collisions = [s for s in scores if s['success'] == 0 and s['collision'] > 0]
    evaluated = len(scores)
    row = {
        'evaluated_count': evaluated,
        'success_count': len(successes),
        'success_rate': len(successes) / evaluated if evaluated else None,
        'collision_count': len(collisions),
        'collision_rate': len(collisions) / evaluated if evaluated else None,
        'other_failure_count': evaluated - len(successes) - len(collisions),
        'avg_path_length': mean_of(scores, 'path_length'),
        'avg_path_length_success': mean_of(successes, 'path_length'),
        'avg_total_joint_motion': mean_of(scores, 'total_joint_motion'),
        'avg_planning_time': mean_of(scores, 'planning_time'),
        'avg_planning_time_success': mean_of(successes, 'planning_time'),
        'avg_time_to_success': mean_of(successes, 'time_to_success'),
        'avg_steps_to_success': mean_of(successes, 'steps_to_success'),
        'avg_mean_pos_residual': mean_of(scores, 'mean_pos_residual'),
        'avg_mean_orn_residual': mean_of(scores, 'mean_orn_residual'),
    }
    for ur5_count in range(1, 5):
        arm_scores = [s for s in scores
                      if s['task']['ur5_count'] == ur5_count]
        row['success_rate_{}_arm'.format(ur5_count)] = \
            sum(s['success'] > 0 for s in arm_scores) / len(arm_scores)\
            if arm_scores else None
    return row


def append_summary_row(summary_path, row):
    write_header = not exists(summary_path)
    with open(summary_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def print_summary(rows):
    print('\n=== Bulk Evaluation Summary ===')
    print('{:40s} {:>8s} {:>8s} {:>8s} {:>10s}'.format(
        'name', 'evaluated', 'success', 'collide', 'path_len'))
    for row in rows:
        print('{:40s} {:>8} {:>8} {:>8} {:>10}'.format(
            row['name'],
            str(row.get('evaluated_count', '-')),
            str(row.get('success_count', '-')),
            str(row.get('collision_count', '-')),
            '{:.4f}'.format(row['avg_path_length'])
            if row.get('avg_path_length') is not None else '-'))


if __name__ == "__main__":
    args = parse_args()
    jobs = load_jobs(args.checkpoint_list)

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    # Logger saves to benchmark/<name>.pkl, so name is relative to benchmark/
    run_dir = join(args.output_dir, stamp)
    makedirs(run_dir, exist_ok=True)
    summary_path = join(run_dir, '{}_summary.csv'.format(stamp))
    dump(load(open(args.checkpoint_list)),
         open(join(run_dir, '{}_checkpoints.json'.format(stamp)), 'w'),
         indent=4)
    print('[BulkEvaluate] Run timestamp {} | {} jobs | output: {}'
          .format(stamp, len(jobs), run_dir), flush=True)

    rows = []
    for i, job in enumerate(jobs):
        print('\n[BulkEvaluate] ({}/{}) {}'.format(
            i + 1, len(jobs), job['name']), flush=True)
        detail_name = '{}_{}'.format(stamp, job['name'])
        benchmark_name = str(Path(run_dir).relative_to('benchmark')
                             / detail_name).replace('\\', '/')
        detail_pkl = join(run_dir, detail_name + '.pkl')
        row = {
            'run_timestamp': stamp,
            'name': job['name'],
            'method': 'ompl_' + job['planner'] if job['planner']
            else 'rl_policy',
            'checkpoint': job['path'] or '',
            'config': job['config'] or '<checkpoint config.json>',
            'tasks_path': job['tasks_path'],
            'task_count': count_tasks(job['tasks_path']),
            'detail_file': join(run_dir, detail_name + '.csv'),
        }

        start = time()
        if job['planner'] is None and not exists(job['path']):
            row['status'] = 'missing_checkpoint'
        else:
            return_code = run_benchmark(job, benchmark_name)
            row['status'] = 'ok' if return_code == 0 \
                else 'exit_code_{}'.format(return_code)
        row['wall_time_sec'] = round(time() - start, 1)

        if exists(detail_pkl):
            scores = pickle.load(open(detail_pkl, 'rb'))
            if scores:
                row.update(summarize(scores))
        elif row['status'] == 'ok':
            row['status'] = 'no_results'

        append_summary_row(summary_path, row)
        rows.append(row)

    print_summary(rows)
    print('\n[BulkEvaluate] Summary saved to', summary_path)
