"""
Benchmark using pre-saved expert waypoints instead of a trained policy.
Runs tasks sequentially with GUI enabled so you can watch the simulation.

Usage:
    python -m helper.benchmark_expert \
        --config configs/default.json \
        --tasks_path benchmark/ \
        --expert_waypoints expert/ \
        --name expert_benchmark          # optional, names the output .pkl
"""

import ray
import pickle
import numpy as np
from signal import signal, SIGINT
from os.path import exists
from tqdm import tqdm

from utils import parse_args, load_config, Logger
from environment.benchmarkEnv import BenchmarkEnv
from environment import TaskLoader
from environment.utils import perform_expert_actions


def exit_gracefully(logger, output_path, benchmark_results):
    print("\nInterrupted — saving partial results to", output_path)
    pickle.dump(benchmark_results, open(output_path, 'wb'))
    ray.shutdown()
    exit(0)


if __name__ == "__main__":
    args = parse_args()

    # Always show the simulation
    args.gui = True
    # Single process — GUI requires main thread
    args.num_processes = 1

    if args.tasks_path is None:
        print("Please supply tasks with --tasks_path")
        exit(1)
    if args.expert_waypoints is None:
        print("Please supply expert waypoints with --expert_waypoints")
        exit(1)
    if args.config is None:
        print("Please supply a config with --config")
        exit(1)

    config = load_config(args.config)
    env_config = config['environment']
    training_config = config['training']

    ray.init()

    output_path = 'benchmark/expert_benchmark_score.pkl'
    if args.name:
        output_path = ' benchmark/{}_benchmark_score.pkl'.format(args.name)

    benchmark_results = []
    continue_benchmark = False
    finished_task_paths = set()
    # if exists(output_path):
    #     benchmark_results = pickle.load(open(output_path, 'rb'))
    #     continue_benchmark = False
    #     finished_task_paths = {r['task']['task_path']
    #                            for r in benchmark_results}
    #     print("Resuming benchmark — {} tasks already done.".format(
    #         len(benchmark_results)))

    # Logger Ray actor (benchmark_mode=True so it just collects scores)
    logdir = args.expert_waypoints.rstrip('/')
    logger = Logger.remote(
        logdir=logdir,
        benchmark_mode=True,
        benchmark_name=args.name or 'expert_benchmark'
    )

    # Plain local loader — used only to count / list tasks.
    local_task_loader = TaskLoader(
        root_dir=args.tasks_path,
        shuffle=False,
        repeat=False
    )
    num_tasks = len(local_task_loader)

    # TaskManager.setup_next_task calls task_loader.get_next_task.remote(),
    # so it must be a Ray remote actor. Point it at the same benchmark dir
    # so the env loads exactly the benchmark tasks in order.
    remote_task_loader = ray.remote(TaskLoader).remote(
        root_dir=args.tasks_path,
        shuffle=False,
        repeat=False
    )
    training_config['task_loader'] = remote_task_loader

    # Non-remote BenchmarkEnv so we can call perform_expert_actions directly
    env = BenchmarkEnv(
        env_config=env_config,
        training_config=training_config,
        gui=True,
        logger=logger
    )

    # BenchmarkEnv needs a memory_cluster_map with exactly 1 entry.
    # Without a policy manager we inject a dummy None entry so the
    # assert len(self.memory_cluster_map) == 1 in setup_task passes.
    env.set_memory_cluster_map({'multiarm_motion_planner': None})

    signal(SIGINT, lambda sig, frame: exit_gracefully(
        logger, output_path, benchmark_results))

    print("Running expert benchmark on {} tasks...".format(num_tasks))

    with tqdm(range(num_tasks), dynamic_ncols=True, smoothing=0.01) as pbar:
        for _ in pbar:
            # reset() -> setup_task() -> task_manager.setup_next_task()
            # -> task_loader.get_next_task.remote()
            # The remote loader auto-advances; read back which task landed.
            env.reset()
            task = env.get_current_task()

            if task.task_path in finished_task_paths:
                continue

            # Load pre-saved expert waypoints for this task
            waypoint_path = (args.expert_waypoints.rstrip('/')
                             + '/' + task.id + '.npy')

            if not exists(waypoint_path):
                print("\n[skip] No waypoints found for task:", task.id)
                continue

            expert_waypoints = np.load(waypoint_path)

            # Follow the expert waypoints
            perform_expert_actions(
                env=env,
                expert_waypoints=expert_waypoints,
                expert_config=env_config['expert']
            )

            # on_episode_end scores the run and sends to logger; env.step()
            # already calls it when the episode terminates, so don't log twice
            if not env.terminate_episode:
                env.on_episode_end()

            result = env.current_episode_score
            benchmark_results.append(result)

            success_rate = np.mean([r.get('success', 0)
                                    for r in benchmark_results])
            pbar.set_description(
                'Success Rate: {:.3f}'.format(success_rate))

    # Final save
    print("\nSaving results to", output_path)
    pickle.dump(benchmark_results, open(output_path, 'wb'))

    # Summary
    if benchmark_results:
        keys = [k for k in benchmark_results[0]
                if k not in ('task', 'debug')]
        print("\n=== Expert Benchmark Summary ===")
        for key in keys:
            vals = [r[key] for r in benchmark_results if key in r]
            print("  {:30s}  mean={:.4f}  std={:.4f}".format(
                key, np.mean(vals), np.std(vals)))

    ray.get(logger.atexit.remote())
    ray.shutdown()