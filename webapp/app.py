"""Dash app entry point.

Run from the repo root with the webapp conda env (see webapp/requirements.txt):
    python -m webapp.app
"""
from dash import Dash

from .callbacks.run_callbacks import register_run_callbacks
from .layout.run_tab import build_run_tab
from .video_feed import register_video_feed

app = Dash(__name__)
app.title = 'UR5 Multi-arm Planner Simulation'  # browser tab
register_video_feed(app.server)

# Passed as a function so Dash rebuilds the layout on every page load,
# picking up new checkpoints in runs/webapp/ and new task folders.
app.layout = build_run_tab
register_run_callbacks(app)


if __name__ == '__main__':
    # threaded=True is required: the MJPEG stream on /video_feed is a
    # long-lived connection, and Flask's dev server defaults to handling one
    # request at a time, which would otherwise starve Dash's own callbacks.
    # use_reloader=False avoids Werkzeug's reloader importing this module
    # (and re-connecting PyBullet) twice.
    app.run(debug=True, threaded=True, use_reloader=False)
