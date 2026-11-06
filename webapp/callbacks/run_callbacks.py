import time

from dash import ctx, html, no_update, Input, Output, State

from ..layout.run_tab import CAMERA_RANGES, PAUSE_SYMBOL, PLAY_SYMBOL
from ..render import DEFAULT_CAMERA
from ..sim_manager import object_label, sim_manager, STEP_SIZE
from ..tasks_io import list_task_files, load_task as load_task_file

# Camera center boxes are clamped to +/- this, which they're sized for.
CENTER_LIMIT = 9.9

# Disabled while sim_manager is busy (loading a planner or planning).
SELECTION_CONTROLS = ['algorithm-dropdown', 'task-folder-dropdown',
                      'task-file-dropdown', 'prev-task-button',
                      'next-task-button']


def register_run_callbacks(app):
    @app.callback(
        Output('task-file-dropdown', 'options'),
        Output('task-file-dropdown', 'value'),
        Input('task-folder-dropdown', 'value'),
    )
    def update_task_files(folder):
        if not folder:
            return [], None
        files = list_task_files(folder)
        options = [{'label': f, 'value': f} for f in files]
        return options, (files[0] if files else None)

    @app.callback(
        Output('task-file-dropdown', 'value', allow_duplicate=True),
        Input('prev-task-button', 'n_clicks'),
        Input('next-task-button', 'n_clicks'),
        State('task-file-dropdown', 'options'),
        State('task-file-dropdown', 'value'),
        prevent_initial_call=True,
    )
    def step_task_file(_prev, _next, options, current):
        # Wraps around: Previous on the first task goes to the last, Next
        # on the last goes to the first. Changing the value runs the task
        # via handle_auto_run_task, as picking it from the dropdown would.
        files = [option['value'] for option in options or []]
        if not files or sim_manager.busy:
            return no_update
        step = -1 if ctx.triggered_id == 'prev-task-button' else 1
        if current not in files:
            return files[0] if step == 1 else files[-1]
        return files[(files.index(current) + step) % len(files)]

    @app.callback(
        Output('camera-yaw-value', 'children'),
        Output('camera-pitch-value', 'children'),
        Output('camera-distance-value', 'children'),
        Input('camera-yaw', 'value'),
        Input('camera-pitch', 'value'),
        Input('camera-distance', 'value'),
    )
    def update_camera(yaw, pitch, distance):
        sim_manager.set_camera(yaw=yaw, pitch=pitch, distance=distance)
        return f'{yaw:g}', f'{pitch:g}', f'{distance:.1f}'

    @app.callback(
        Output('camera-yaw', 'value'),
        Output('camera-pitch', 'value'),
        Output('camera-distance', 'value'),
        *[Input(f'camera-{key}-{direction}', 'n_clicks')
          for key in CAMERA_RANGES for direction in ('up', 'down')],
        Input('reset-camera-button', 'n_clicks'),
        State('camera-yaw', 'value'),
        State('camera-pitch', 'value'),
        State('camera-distance', 'value'),
        prevent_initial_call=True,
    )
    def step_camera(*args):
        # The arrows next to each slider, and "Reset default view"; moving
        # the sliders fires update_camera, which applies it.
        if ctx.triggered_id == 'reset-camera-button':
            return tuple(DEFAULT_CAMERA[key] for key in CAMERA_RANGES)
        values = dict(zip(CAMERA_RANGES, args[-3:]))
        _, key, direction = ctx.triggered_id.split('-')
        low, high, step = CAMERA_RANGES[key]
        value = values[key] + (step if direction == 'up' else -step)
        values[key] = round(min(max(value, low), high), 1)
        return tuple(values.values())

    @app.callback(
        Output('camera-center-x', 'value'),
        Output('camera-center-y', 'value'),
        Output('camera-center-z', 'value'),
        Input('camera-center-x', 'value'),
        Input('camera-center-y', 'value'),
        Input('camera-center-z', 'value'),
        *[Input(f'camera-center-{axis}-{direction}', 'n_clicks')
          for axis in 'xyz' for direction in ('up', 'down')],
        Input('reset-camera-button', 'n_clicks'),
        prevent_initial_call=True,
    )
    def update_camera_center(x, y, z, *_clicks):
        """Typing in a center box (applied on Enter or leaving it), its
        arrows, and "Reset default view". The boxes are text fields so they
        can always show one decimal; anything that isn't a number falls
        back to the current center."""
        if ctx.triggered_id == 'reset-camera-button':
            center = list(DEFAULT_CAMERA['target'])
        else:
            center = list(sim_manager.camera['target'])
            for i, text in enumerate((x, y, z)):
                try:
                    center[i] = float(text)
                except (TypeError, ValueError):
                    pass
            if ctx.triggered_id.endswith(('-up', '-down')):
                _, _, axis, direction = ctx.triggered_id.split('-')
                center['xyz'.index(axis)] += (
                    0.1 if direction == 'up' else -0.1)
        center = [round(min(max(value, -CENTER_LIMIT), CENTER_LIMIT), 1)
                  for value in center]
        sim_manager.set_camera(target=center)
        return tuple(f'{value:.1f}' for value in center)

    @app.callback(
        Output('algorithm-echo', 'data'),
        Input('algorithm-dropdown', 'value'),
    )
    def handle_select_algorithm(algorithm_key):
        # Starts loading the planner (e.g. the RL checkpoint) right away, in
        # the background. A loaded task is re-run with the new planner,
        # since there is no Run button; the run waits for the load.
        if sim_manager.select_algorithm(algorithm_key) and algorithm_key \
                and sim_manager.current_task is not None:
            sim_manager.run(algorithm_key)
        return ''

    @app.callback(
        Output('load-task-echo', 'data'),
        Input('task-file-dropdown', 'value'),
        State('task-folder-dropdown', 'value'),
        State('algorithm-dropdown', 'value'),
    )
    def handle_auto_run_task(filename, folder, algorithm_key):
        # Loads and runs as soon as a task is selected. Also fires on the
        # folder dropdown's own selection (which sets task-file-dropdown's
        # value to the first file), so picking a folder alone runs its
        # first task.
        # Returns a new token per loaded task, which refreshes the highlight
        # checkboxes (update_highlight_options).
        if not folder or not filename:
            return no_update  # nothing picked yet (e.g. on page load)
        if sim_manager.busy:
            # The controls are disabled while busy; this only catches a
            # click that slipped in before they greyed out.
            sim_manager.log(f'Busy, ignored {folder}/{filename}.')
            return no_update
        task = load_task_file(folder, filename)
        if task is None:
            sim_manager.log(f'Failed to load {folder}/{filename}.')
            return no_update
        sim_manager.load_task(task, name=f'{folder}/{filename}')
        if algorithm_key:
            sim_manager.run(algorithm_key)
        else:
            sim_manager.log('Select a planner to run this task.')
        return f'{folder}/{filename}@{time.monotonic()}'

    @app.callback(
        Output('highlight-arms', 'options'),
        Output('highlight-arms', 'value'),
        Output('highlight-obstacles', 'options'),
        Output('highlight-obstacles', 'value'),
        Input('load-task-echo', 'data'),
    )
    def update_highlight_options(_loaded):
        # One checkbox per arm/obstacle of the loaded task, none ticked
        # (sim_manager.load_task has already cleared the highlight).
        arms, obstacles = sim_manager.object_keys()
        return (_checkbox_options(arms), [],
                _checkbox_options(obstacles), [])

    @app.callback(
        Output('highlight-echo', 'data'),
        Input('highlight-arms', 'value'),
        Input('highlight-obstacles', 'value'),
    )
    def handle_highlight(arms, obstacles):
        sim_manager.set_highlight((arms or []) + (obstacles or []))
        return ''

    @app.callback(
        Output('highlight-arms', 'value', allow_duplicate=True),
        Output('highlight-obstacles', 'value', allow_duplicate=True),
        Input('highlight-clear-button', 'n_clicks'),
        Input('highlight-collided-button', 'n_clicks'),
        prevent_initial_call=True,
    )
    def handle_highlight_buttons(_clear, _collided):
        if ctx.triggered_id == 'highlight-collided-button':
            keys = sim_manager.collided
        else:
            keys = []
        return ([k for k in keys if k.startswith('arm:')],
                [k for k in keys if k.startswith('obs:')])

    # Grey the selection controls out the moment one is used, rather than
    # on the next status poll; poll_status re-enables them once not busy.
    app.clientside_callback(
        """
        function() {
            return [true, true, true, true, true];
        }
        """,
        [Output(component_id, 'disabled', allow_duplicate=True)
         for component_id in SELECTION_CONTROLS],
        Input('algorithm-dropdown', 'value'),
        Input('task-file-dropdown', 'value'),
        Input('prev-task-button', 'n_clicks'),
        Input('next-task-button', 'n_clicks'),
        prevent_initial_call=True,
    )

    @app.callback(
        Output('frame-slider', 'min'),
        Output('frame-slider', 'max'),
        Output('frame-slider', 'value'),
        Output('playback-status', 'children'),
        Output('play-button', 'children'),
        Output('speed-button', 'children'),
        *[Output(component_id, 'disabled')
          for component_id in SELECTION_CONTROLS],
        Output('highlight-collided-button', 'style'),
        Output('collision-checklist', 'style'),
        Output('export-png-button', 'disabled'),
        Output('export-frames-button', 'disabled'),
        Output('export-gif-button', 'disabled'),
        Input('status-poll', 'n_intervals'),
    )
    def poll_status(_n_intervals):
        n_frames = len(sim_manager.frames)
        max_index = max(n_frames - 1, 0)
        playback_text = (
            f'Frame {sim_manager.frame_index + 1} / {n_frames}'
            + (' (playing)' if sim_manager.playing else '')
            if n_frames else 'No path loaded.')
        busy = sim_manager.busy
        collided_style = {} if sim_manager.collided else {'display': 'none'}
        return (0, max_index, sim_manager.frame_index, playback_text,
                *_transport_labels(), *[busy] * len(SELECTION_CONTROLS),
                collided_style, collided_style,
                *_export_disabled(n_frames))

    @app.callback(
        Output('message-log', 'children'),
        Output('message-log-count', 'data'),
        Input('status-poll', 'n_intervals'),
        State('message-log-count', 'data'),
    )
    def poll_message_log(_n_intervals, shown_count):
        total, entries = sim_manager.log_entries()
        # Only re-render (and re-scroll) when something new was logged.
        # Compare the running total, not len(entries): that stops changing
        # once the log is capped at MAX_LOG_ENTRIES.
        if total == shown_count:
            return no_update, no_update
        rows = [html.Div([html.Span(stamp, className='log-time'), text],
                         className='log-entry')
                for stamp, text in reversed(entries)]  # newest first
        return rows, total

    # Jump back to the newest entry (the top) whenever the log changes.
    app.clientside_callback(
        """
        function(_children) {
            const log = document.getElementById('message-log');
            if (log) {
                setTimeout(() => { log.scrollTop = 0; }, 0);
            }
            return '';
        }
        """,
        Output('message-log-scroll', 'data'),
        Input('message-log', 'children'),
    )

    @app.callback(
        Output('seek-echo', 'data'),
        Input('frame-slider', 'value'),
        prevent_initial_call=True,
    )
    def handle_seek(value):
        if value is not None:
            sim_manager.seek(value)
        return ''

    @app.callback(
        Output('display-echo', 'data'),
        Input('display-checklist', 'value'),
    )
    def handle_display(values):
        values = values or []
        sim_manager.set_show_path('trajectories' in values)
        sim_manager.set_show_labels('labels' in values)
        sim_manager.set_show_caption('caption' in values)
        return ''

    @app.callback(
        Output('export-echo', 'data'),
        Input('export-png-button', 'n_clicks'),
        Input('export-frames-button', 'n_clicks'),
        Input('export-gif-button', 'n_clicks'),
        prevent_initial_call=True,
    )
    def handle_export(_png, _frames, _gif):
        kind = {'export-png-button': 'png',
                'export-frames-button': 'frames',
                'export-gif-button': 'gif'}[ctx.triggered_id]
        if not sim_manager.exporter.start(kind):
            sim_manager.log('An export is already running.')
        return ''

    @app.callback(
        Output('collision-echo', 'data'),
        Input('collision-checklist', 'value'),
    )
    def handle_show_collision(values):
        sim_manager.set_show_collision('collision' in (values or []))
        return ''

    @app.callback(
        Output('play-button', 'children', allow_duplicate=True),
        Output('speed-button', 'children', allow_duplicate=True),
        Input('rewind-button', 'n_clicks'),
        Input('play-button', 'n_clicks'),
        Input('skip-button', 'n_clicks'),
        Input('replay-button', 'n_clicks'),
        Input('speed-button', 'n_clicks'),
        prevent_initial_call=True,
    )
    def handle_transport(_rewind, _play, _skip, _replay, _speed):
        # Returns the new labels right away; poll_status also keeps them
        # in sync, e.g. when playback reaches the last frame by itself.
        triggered = ctx.triggered_id
        if triggered == 'rewind-button':
            sim_manager.step(-STEP_SIZE)
        elif triggered == 'play-button':
            sim_manager.toggle_play()
        elif triggered == 'skip-button':
            sim_manager.step(STEP_SIZE)
        elif triggered == 'replay-button':
            sim_manager.replay()
        elif triggered == 'speed-button':
            sim_manager.cycle_speed()
        return _transport_labels()


def _checkbox_options(keys):
    return [{'label': f' {object_label(key)}', 'value': key} for key in keys]


def _transport_labels():
    """(play/pause button label, speed button label)"""
    speed = sim_manager.playback_speed
    return (PAUSE_SYMBOL if sim_manager.playing else PLAY_SYMBOL,
            f'{speed:g}x')


def _export_disabled(n_frames):
    """(png, frames, gif) disabled flags: the current view can always be
    saved once a task is loaded; the others need a trajectory. All are off
    while an export runs."""
    exporting = sim_manager.exporter.busy
    has_task = sim_manager.current_task is not None
    return (exporting or not has_task,
            exporting or not n_frames,
            exporting or not n_frames)
