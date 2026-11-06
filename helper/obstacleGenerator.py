import ray
import numpy as np
from environment.rrt import RRTWrapper
from environment.tasks import Task
import json
import sys
import os

def iter_json_files(root):
    with os.scandir(root) as it:
        for entry in it:
            if entry.is_dir(follow_symlinks=False):
                yield from iter_json_files(entry.path)
            elif entry.name.endswith('.json'):
                yield entry.path


def iter_tasks(tasks_dir, logs):
    """
    Streaming generator — yields (filename, task_data, task_file) one at a time.
    Each file is only opened and read when a worker is ready to consume it,
    so only num_workers task objects ever live in memory at once.
    Load errors are logged in-place and skipped.
    """
    for task_file in iter_json_files(tasks_dir):
        filename = os.path.basename(task_file)
        if filename in logs["runs"]:
            continue
        try:
            with open(task_file) as f:
                task_data = json.load(f)
            yield filename, task_data, task_file
        except Exception as e:
            print(f"[ERROR] Could not load {filename}: {e}")
            logs["runs"][filename] = "error:0"
            logs["stats"]["processed"] += 1
            logs["stats"]["errors"] += 1


def dump_json(dics, filename, save_write=False):

    # Use save_write to maintain write atomicity
    if save_write:

        temp_filename = 'temp/' + os.path.basename(filename)
        
        if not os.path.exists('temp/'):
            os.makedirs('temp/')

        with open(temp_filename, "w") as f:
            json.dump(dics, f, indent=4)

        os.replace(temp_filename, filename)

        
    else:
        with open(filename, "w") as f:
            json.dump(dics, f, indent=4)

    


def build_obstacles(task_data, obs_config):
    """Sample a random obstacle configuration for a task."""
    obstacles_count = int(np.random.choice(np.arange(0, 6), p=[.1, .15, .2, .2, .2, .15]))
    obstacles = {}
    
    for obs in list(obs_config)[:obstacles_count]:
        base_poses = [b[0] for b in task_data['base_poses']]
        base = base_poses[np.random.randint(0, len(base_poses))]
        r = np.random.uniform(0.2, 0.7)
        a = np.random.uniform(0, 2 * np.pi)
        b = np.random.uniform(0, np.pi)
        offset = [
            r * np.cos(a) * np.cos(b),
            r * np.sin(a) * np.cos(b),
            r * np.sin(b)
        ]
        obstacles[obs] = [base[i] + offset[i] for i in range(3)]
    return obstacles, obstacles_count


def build_task(task_data, task_file, obstacles):
    return Task(
        target_eff_poses=task_data['target_eff_poses'],
        base_poses=task_data['base_poses'],
        start_config=task_data['start_config'],
        goal_config=task_data['goal_config'],
        start_goal_config=task_data.get('goal_config'),
        obstacles=obstacles,
        difficulty=task_data.get('difficulty', 0.0),
        dynamic_speed=task_data.get('dynamic_speed'),
        task_path=str(task_file)
    )


def submit_task(worker, task_data, task_file, obs_config, birrt_timeout):
    """Build obstacles, create task, submit to a specific worker. Returns (future, obstacles, obstacles_count)."""
    obstacles, obstacles_count = build_obstacles(task_data, obs_config)
    task = build_task(task_data, task_file, obstacles)
    future = worker.birrt_from_task.remote(task, timeout=birrt_timeout)
    return future, obstacles, obstacles_count



def generate_expert_demonstrations(task_name='', target_name='', config_file='', 
        reset=False, max_attempts=10, num_workers=10, gui=False):

    """
    Generate obstacles and its RRT expert waypoints for all tasks in a directory 
    
    Tasks input located in tasks/{task_name}/
    Tasks with generated obstacles output in tasks/{target_name}/
    Experts output in experts/{target_name}/
    Training logs saved in logs/birrt_{target_name}.json
    
    reset: Whether to generate from scratch (True) or continue from logs (False)
    max_attempts: set attempts to generate obstacles
    gui: Whether to visualize RRT (slow, use False for batch processing)
    """

    
    # Initialize Ray
    print("Initializing Ray...")
    ray.init()
    
    # Environment config (from configs/default.json)
    
    with open(config_file) as f:
        env_config = json.load(f)['environment']
        obs_config = env_config['obstacles']
    
    tasks_dir = f'tasks/{task_name}/'
    target_dir = f'tasks/{target_name}/'
    failed_dir = f'tasks/{target_name}_failed/'
    experts_dir = f'experts/{target_name}/'
    logs_file = f'logs/birrt_{target_name}.json'

    birrt_timeout = 30
    done_count = 0

    logs = {
        "stats":{
            "processed":0,
            "success":0,
            "failed":0,
            "errors":0
        },
        "runs":{}
    }
    
    if not reset:
        try: 
            logs = json.load(open(logs_file))
        except: 
            pass

    
    # Create output directory if needed
    for dir in [experts_dir, target_dir, failed_dir, 'logs/']:
        if not os.path.exists(dir):
            os.makedirs(dir)

    
    # ── Build pending task queue ───────────────
    pending = iter_tasks(tasks_dir, logs)

    # ── Create worker pool ────────────────────
    print(f"Creating {num_workers} RRTWrapper workers...")
    workers = [
        RRTWrapper.remote(env_config=env_config, gui=gui)
        for _ in range(num_workers)
    ]

    # ── worker_state tracks what each worker is doing ─────────────────────
    # {
    #   worker: {
    #     "future":          ObjectRef,
    #     "filename":        str,
    #     "task_data":       dict,
    #     "task_file":       str,
    #     "attempt":         int,
    #     "obstacles":       dict,
    #     "obstacles_count": int,
    #   }
    # }
    worker_state = {}

    def assign_next_task(worker):
        """Pull the next pending task and submit it to this worker. Returns True if assigned."""
        try:
            filename, task_data, task_file = next(pending)
        except StopIteration:
            return False   # no more tasks
        

        future, obstacles, obstacles_count = submit_task(
            worker, task_data, task_file, obs_config, birrt_timeout)

        worker_state[worker] = {
            "future":          future,
            "filename":        filename,
            "task_data":       task_data,
            "task_file":       task_file,
            "attempt":         1,
            "obstacles":       obstacles,
            "obstacles_count": obstacles_count,
        }
        return True

    def retry_same_worker(worker):
        """Resample obstacles and resubmit the same task on the same worker."""
        state = worker_state[worker]
        future, obstacles, obstacles_count = submit_task(
            worker, state["task_data"], state["task_file"], obs_config, birrt_timeout)

        state["future"]          = future
        state["attempt"]        += 1
        state["obstacles"]       = obstacles
        state["obstacles_count"] = obstacles_count

    # ── Seed all workers with their first task ─
    for worker in workers:
        if not assign_next_task(worker):
            break   # fewer tasks than workers


    # ── Main loop ─────────────────────────────
    ray_timeout = birrt_timeout + 30   # a little headroom over the internal timeout

    while worker_state:
        
        # Save every 100 tasks
        if done_count%100 == 0:
            dump_json(logs, logs_file, save_write=True)
        
        # All active futures
        all_futures  = [s["future"] for s in worker_state.values()]
        future_to_worker = {s["future"]: w for w, s in worker_state.items()}

        # Wait for whichever worker finishes first
        ready, _ = ray.wait(all_futures, num_returns=1, timeout=ray_timeout)

        # ── Timeout: nothing finished in time ─
        if not ready:
            # Pick the first worker, treat it as timed-out
            future = all_futures[0]
            worker = future_to_worker[future]
            state  = worker_state[worker]
            ray.cancel(future, force=True)
            waypoints = None
        else:
            future = ready[0]
            worker = future_to_worker[future]
            state  = worker_state[worker]
            try:
                waypoints = ray.get(future, timeout=5)   # already done, 5s is plenty
            except Exception as e:
                print(f"[ERROR] Cant get ray! {e}")
                logs["runs"][state["filename"]] = "error:0"
                logs["stats"]["processed"] += 1
                logs["stats"]["errors"] += 1
                done_count += 1
                del worker_state[worker]
                assign_next_task(worker)
                continue

        filename        = state["filename"]
        task_data       = state["task_data"]
        task_file       = state["task_file"]
        attempt         = state["attempt"]
        obstacles       = state["obstacles"]
        obstacles_count = state["obstacles_count"]

        # ── Success ───────────────────────────
        if waypoints is not None and len(waypoints) > 0:
            task_data['obstacles']       = obstacles
            task_data['obstacles_count'] = obstacles_count
            dump_json(task_data, target_dir + filename)

            task = build_task(task_data, task_file, obstacles)
            np.save(f'{experts_dir}/{task.id}.npy', waypoints)

            logs["runs"][filename]       = f"success:{attempt}"
            logs["stats"]["processed"] += 1
            logs["stats"]["success"]   += 1
            done_count += 1

            # Worker is free — give it a new task
            del worker_state[worker]
            assign_next_task(worker)

        # ── Failure: retry on the same worker ─
        else:
            if attempt < max_attempts:
                retry_same_worker(worker)
            else:
                logs["runs"][filename]      = f"failed:{attempt}"
                logs["stats"]["processed"] += 1
                logs["stats"]["failed"]    += 1
                done_count += 1

                # Worker is free — give it a new task
                del worker_state[worker]
                assign_next_task(worker)


    
    dump_json(logs, logs_file, save_write=True)

    # ── Summary ───────────────────────────────
    s = logs["stats"]
    print(f"\nDone!  processed={s['processed']}  "
          f"success={s['success']}  failed={s['failed']}  errors={s['errors']}")
    ray.shutdown()


if __name__ == "__main__":
    # Generate experts for all tasks
    

    generate_expert_demonstrations(
        task_name=sys.argv[1],
        target_name=sys.argv[2],
        reset="reset" in sys.argv,
        num_workers=int(sys.argv[3]),
        config_file='configs/RRTconfig.json',
        gui=False
    )