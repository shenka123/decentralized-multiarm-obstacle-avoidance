"""Long-lived webapp/plan_worker.py process in WSL's multiarm-ompl env.

See plan_worker's docstring for why planning happens there rather than in
the webapp's own env, and for the line protocol spoken here. The defaults
match this machine's setup; override them with MULTIARM_WSL_PYTHON /
MULTIARM_WSL_DISTRO if it moves.
"""
import json
import os
import queue
import subprocess
import threading
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
WSL_PYTHON = os.environ.get(
    'MULTIARM_WSL_PYTHON', '/home/shenk/miniconda3/envs/multiarm-ompl/bin/python')
WSL_DISTRO = os.environ.get('MULTIARM_WSL_DISTRO')  # None -> default distro

# Must match plan_worker.
READY_MARKER = '@@READY'
RESULT_MARKER = '@@RESULT '

STARTUP_TIMEOUT = 300


def to_wsl_path(path):
    """C:\\Users\\x -> /mnt/c/Users/x"""
    path = Path(path).resolve()
    drive = path.drive.rstrip(':').lower()
    return f'/mnt/{drive}' + path.as_posix()[len(path.drive):]


class WorkerProcess:
    """One plan_worker process, started with `args` (which select the
    method). start() blocks until its setup is done; plan() then reuses it
    for every task."""

    def __init__(self, *args):
        self.args = args
        self.proc = None
        self._replies = queue.Queue()
        self._recent_output = deque(maxlen=20)  # for error messages
        self._lock = threading.Lock()  # one request at a time

    @property
    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        with self._lock:
            if self.alive:
                return
            cmd = ['wsl.exe']
            if WSL_DISTRO:
                cmd += ['-d', WSL_DISTRO]
            cmd += ['--cd', to_wsl_path(REPO_ROOT), '-e', WSL_PYTHON,
                    '-u', '-m', 'webapp.plan_worker', *self.args]
            self._replies = queue.Queue()
            self.proc = subprocess.Popen(
                cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, errors='replace',
                bufsize=1)
            threading.Thread(
                target=self._read_output, args=(self.proc, self._replies),
                daemon=True).start()
            self._wait_for(READY_MARKER, STARTUP_TIMEOUT)

    def plan(self, task, timeout):
        """Returns plan_worker's result dict for `task`."""
        self.start()
        with self._lock:
            request = json.dumps({'task': to_wsl_path(task.task_path)})
            self.proc.stdin.write(request + '\n')
            self.proc.stdin.flush()
            result = json.loads(self._wait_for(RESULT_MARKER, timeout))
        if 'error' in result:
            raise RuntimeError(result['error'])
        return result

    def close(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None

    def _read_output(self, proc, replies):
        for line in proc.stdout:
            line = line.rstrip('\n')
            if line.startswith((READY_MARKER, RESULT_MARKER)):
                replies.put(line)
            else:
                self._recent_output.append(line)
        replies.put(None)  # process exited

    def _wait_for(self, marker, timeout):
        try:
            line = self._replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise RuntimeError(f'plan_worker timed out after {timeout}s')
        if line is None:
            last = next((l for l in reversed(self._recent_output)
                         if l.strip()), 'no output')
            self.close()
            raise RuntimeError(f'plan_worker exited: {last}')
        return line[len(marker):]
