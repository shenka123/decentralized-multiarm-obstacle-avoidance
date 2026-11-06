"""Task folder discovery + loading, built on environment.tasks.Task."""
from pathlib import Path

from environment.tasks import Task

REPO_ROOT = Path(__file__).resolve().parent.parent
TASKS_ROOT = REPO_ROOT / 'tasks'


def list_task_folders():
    if not TASKS_ROOT.exists():
        return []
    return sorted(p.name for p in TASKS_ROOT.iterdir() if p.is_dir())


def list_task_files(folder):
    folder_path = TASKS_ROOT / folder
    if not folder_path.exists():
        return []
    return sorted(p.name for p in folder_path.glob('*.json'))


def load_task(folder, filename):
    task_path = TASKS_ROOT / folder / filename
    return Task.from_file(str(task_path))
