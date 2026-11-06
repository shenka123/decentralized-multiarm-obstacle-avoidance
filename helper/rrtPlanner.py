import ray
import numpy as np
from json import load
from pathlib import Path
from environment.rrt import RRTWrapper
from environment.tasks import Task
from os.path import exists
from os import makedirs
from tqdm import tqdm
import json



def generate_expert_demonstrations(tasks_dir='tasks/', output_dir='tasks_expert/', config_file='', gui=False):
    """
    Generate RRT expert waypoints for all tasks in a directory
    
    Args:
        tasks_dir: Directory containing task JSON files
        output_dir: Directory to save expert waypoints
        gui: Whether to visualize RRT (slow, use False for batch processing)
    """
    
    # Initialize Ray
    print("Initializing Ray...")
    ray.init()
    
    # Environment config (from configs/default.json)
    
    with open(config_file) as f:
        env_config = json.load(f)['environment']
    
    # Create RRT wrapper
    print("Creating RRT wrapper...")
    rrt_wrapper = RRTWrapper.remote(
        env_config=env_config,
        gui=gui
    )
    
    # Create output directory if needed
    if not exists(output_dir):
        makedirs(output_dir)
        print(f"Created output directory: {output_dir}")
    
    # Get all task JSON files
    task_files = list(Path(tasks_dir).rglob('*.json'))
    print(f"Found {len(task_files)} task files")
    
    # Track statistics
    successful = 0
    failed = 0
    failed_tasks = []
    
    # Process each task
    for task_file in tqdm(task_files, desc="Generating expert waypoints"):
        
        try:
            # Load task data
            task_data = load(open(task_file))
            #print(task_data)
            
            # Create Task object
            task = Task(
                target_eff_poses=task_data['target_eff_poses'],
                base_poses=task_data['base_poses'],
                start_config=task_data['start_config'],
                goal_config=task_data['goal_config'],
                start_goal_config=task_data.get('goal_config'),
                obstacles=task_data.get('obstacles'),
                difficulty=task_data.get('difficulty', 0.0),
                dynamic_speed=task_data.get('dynamic_speed'),
                task_path=str(task_file)
            )
            
            
            # Generate expert waypoints using RRT
            #print("=============================", task_data['goal_config'])
            waypoints = ray.get(rrt_wrapper.birrt_from_task.remote(task))

            print("")
            
            if waypoints is not None and len(waypoints) > 0:
                # Save waypoints
                output_path = f'{output_dir}/{task.id}.npy'
                np.save(output_path, waypoints)
                successful += 1
            else:
                failed += 1
                failed_tasks.append(task.id)
                print(f"\n✗ RRT failed for task {task.id}")
                
        except Exception as e:
            failed += 1
            task_id = task_file.stem
            failed_tasks.append(task_id)
            print(f"\n✗ Error processing {task_file}: {e}")
    
    # Print summary
    print("\n" + "="*60)
    print("Expert Generation Summary")
    print("="*60)
    print(f"Total tasks: {len(task_files)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/len(task_files)*100:.2f}%")
    
    if failed_tasks:
        print(f"\nFailed tasks: {failed_tasks[:10]}")  # Show first 10
        if len(failed_tasks) > 10:
            print(f"... and {len(failed_tasks) - 10} more")
    
    # Cleanup
    ray.shutdown()
    print("\nDone!")

if __name__ == "__main__":
    # Generate experts for all tasks
    
    generate_expert_demonstrations(
        tasks_dir='tasks/task_sphere_test/',
        output_dir='tasks/tasks_sphere_test_experts/',
        config_file='configs/RRTconfig.json',
        gui=True  # Set to True to visualize (much slower)
    )