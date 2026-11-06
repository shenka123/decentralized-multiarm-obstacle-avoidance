"""Saves the replay view to webapp/downloads/: the current frame as a PNG,
every frame as numbered PNGs in a folder, or every frame as a GIF.

Images are rendered exactly like the live view -- camera, trajectories,
labels, caption, highlighting and collision colors as currently set -- and
the GIF plays at the current playback speed. Files are named by the time
the export started, e.g. 20261030162314005.png (to the millisecond).
"""
import threading
from datetime import datetime
from pathlib import Path

DOWNLOADS_DIR = Path(__file__).resolve().parent / 'downloads'


def timestamp():
    now = datetime.now()
    return now.strftime('%Y%m%d%H%M%S') + f'{now.microsecond // 1000:03d}'


class Exporter:
    def __init__(self, sim_manager):
        self.sim = sim_manager
        self._thread = None

    @property
    def busy(self):
        return self._thread is not None and self._thread.is_alive()

    def start(self, kind):
        """kind: 'png', 'frames' or 'gif'. Runs in the background; the
        result is logged. Returns False if an export is already running."""
        if self.busy:
            return False
        self._thread = threading.Thread(
            target=self._run, args=(kind,), daemon=True)
        self._thread.start()
        return True

    def _run(self, kind):
        try:
            path = {'png': self._png, 'frames': self._frames,
                    'gif': self._gif}[kind](timestamp())
        except Exception as exc:
            self.sim.log(f'Export failed: {type(exc).__name__}: {exc}')
            return
        if path is not None:
            self.sim.log(
                f'Saved {path.relative_to(DOWNLOADS_DIR.parent).as_posix()}')

    def _png(self, name):
        DOWNLOADS_DIR.mkdir(exist_ok=True)
        path = DOWNLOADS_DIR / f'{name}.png'
        self.sim.render_image().save(path)
        return path

    def _frames(self, name):
        images = self.sim.render_all_frames()
        if images is None:
            self.sim.log('Nothing to export: no trajectory loaded.')
            return None
        folder = DOWNLOADS_DIR / name
        folder.mkdir(parents=True)
        for number, image in enumerate(images, start=1):
            image.save(folder / f'{number:03d}.png')
        return folder

    def _gif(self, name):
        frame_seconds = self.sim.frame_seconds()
        images = self.sim.render_all_frames()
        if images is None:
            self.sim.log('Nothing to export: no trajectory loaded.')
            return None
        DOWNLOADS_DIR.mkdir(exist_ok=True)
        path = DOWNLOADS_DIR / f'{name}.gif'
        images[0].save(path, save_all=True, append_images=images[1:],
                       duration=round(frame_seconds * 1000), loop=0,
                       optimize=False)
        return path
