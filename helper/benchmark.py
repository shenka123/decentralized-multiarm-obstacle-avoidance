"""
Benchmark using OMPL planners instead of a trained RL policy.
Runs tasks sequentially with optional GUI so you can watch the simulation.

Usage:
    python -m helper.benchmark \
        --config configs/evaluate.json \
        --tasks_path tasks/base_test_obs \
        --planner RRTConnect \
        --timeout 30 \
        --name rrtconnect_benchmark \
        --gui

Available planners: RRTConnect, RRT, BITstar, PRM, LBKPIECE
"""

import ray
import pickle
import numpy as np
import argparse
from time import time
from signal import signal, SIGINT
from tqdm import tqdm

from utils import load_config, Logger
from environment.benchmarkEnv import BenchmarkEnv
from environment import TaskLoader
from environment.utils import perform_expert_actions
from environment.rrt.omplWrapper import OMPLWrapper


def parse_args():
    parser = argparse.ArgumentParser("OMPL Benchmark")
    parser.add_argument('--config',     required=True,
                        help='Path to config json')
    parser.add_argument('--tasks_path', required=True,
                        help='Path to directory containing tasks')
    parser.add_argument('--planner',    default='RRTConnect',
                        choices=['RRTConnect', 'RRT',
                                 'BITstar', 'PRM', 'LBKPIECE'],
                        help='OMPL planner to use')
    parser.add_argument('--timeout',    type=float, default=30.0,
                        help='Per-task planning time budget in seconds')
    parser.add_argument('--name',       type=str, default=None,
                        help='Name for output files')
    parser.add_argument('--gui',        action='store_true',
                        help='Render the simulation')
    return parser.parse_args()


def exit_gracefully(logger, output_path, benchmark_results):
    print("\nInterrupted — saving partial results to", output_path)
    pickle.dump(benchmark_results, open(output_path, 'wb'))
    ray.shutdown()
    exit(0)


if __name__ == "__main__":
    args = parse_args()

    config = load_config(args.config)
    env_config = config['environment']
    training_config = config['training']

    ray.init()

    planner_name = args.planner
    run_name = args.name or f'{planner_name}_benchmark'
    output_path = f'benchmark/{run_name}_score.pkl'

    logger = Logger.remote(
        logdir='benchmark',
        benchmark_mode=True,
        benchmark_name=run_name)

    # Local loader just for counting tasks
    local_task_loader = TaskLoader(
        root_dir=args.tasks_path,
        shuffle=False,
        repeat=False)
    num_tasks = len(local_task_loader)

    # Remote loader for the env to pull tasks from
    remote_task_loader = ray.remote(TaskLoader).remote(
        root_dir=args.tasks_path,
        shuffle=False,
        repeat=False)
    training_config['task_loader'] = remote_task_loader

    # BenchmarkEnv runs the actual simulation — unchanged from normal benchmark
    env = BenchmarkEnv(
        env_config=env_config,
        training_config=training_config,
        gui=args.gui,
        logger=logger)
    env.set_memory_cluster_map({'multiarm_motion_planner': None})

    # OMPL planner runs in its own isolated PyBullet instance
    # used only for collision checking during planning
    planner = OMPLWrapper.remote(
        env_config=env_config,
        gui=False,
        planner_name=planner_name)

    benchmark_results = []
    signal(SIGINT, lambda sig, frame: exit_gracefully(
        logger, output_path, benchmark_results))

    print(f"Running {planner_name} benchmark on {num_tasks} tasks...")

    with tqdm(range(num_tasks), dynamic_ncols=True, smoothing=0.01) as pbar:
        for _ in pbar:
            # Reset env and get next task
            env.reset()
            task = env.get_current_task()

            # Plan in isolated PyBullet — returns waypoints only
            t_start = time()
            waypoints = ray.get(
                planner.plan_from_task.remote(task, timeout=args.timeout))
            planning_time = time() - t_start

            if waypoints is None or len(waypoints) == 0:
                print(f"\n[skip] Planning failed for task: {task.id}")
                # Episode ends with no success, metrics still recorded
                env.on_episode_end()
            else:
                # Record planning time separately from execution time
                env.record_planning_time(planning_time)

                # Replay waypoints in BenchmarkEnv's PyBullet
                # All collision, reach, and timing metrics recorded here
                perform_expert_actions(
                    env=env,
                    expert_waypoints=np.array(waypoints),
                    expert_config=env_config['expert'])

                # env.step() already ends (and logs) the episode once it
                # terminates; only end it here if the replay stopped early
                if not env.terminate_episode:
                    env.on_episode_end()

            result = env.current_episode_score
            benchmark_results.append(result)

            success_rate = np.mean(
                [r.get('success', 0) for r in benchmark_results])
            pbar.set_description(
                f'{planner_name} | Success: {success_rate:.3f}')

    # Final save
    print(f"\nSaving results to {output_path}")
    pickle.dump(benchmark_results, open(output_path, 'wb'))

    # Summary
    if benchmark_results:
        keys = [k for k in benchmark_results[0]
                if k not in ('task', 'debug')]
        print(f"\n=== {planner_name} Benchmark Summary ===")
        for key in keys:
            vals = [r[key] for r in benchmark_results if key in r]
            try:
                print(f"  {key:30s}  mean={np.mean(vals):.4f}"
                      f"  std={np.std(vals):.4f}")
            except Exception:
                pass

    ray.get(logger.atexit.remote())
    ray.shutdown()