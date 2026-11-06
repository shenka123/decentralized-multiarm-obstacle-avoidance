"""Page layout. Styling lives in webapp/assets/style.css."""
from dash import dcc, html

from ..algorithms.registry import list_adapters
from ..render import DEFAULT_CAMERA
from ..tasks_io import list_task_folders

# One triangle glyph throughout so the transport buttons match; \ufe0e
# asks for the text (not emoji) presentation.
PLAY_SYMBOL = '\u25b6\ufe0e'
PAUSE_SYMBOL = '\u275a\u275a'
REWIND_SYMBOL = '\u25c0\ufe0e\u25c0\ufe0e'
SKIP_SYMBOL = '\u25b6\ufe0e\u25b6\ufe0e'

# Camera sliders: (min, max, step). The arrows next to each value move it
# by one step, within the same range.
CAMERA_RANGES = {
    'yaw': (-180, 180, 1),
    'pitch': (-90, 0, 1),
    'distance': (0.5, 3, 0.1),
}


def build_selection_column(adapters, current_algorithm):
    """Left: what to run (planner + task) and a log of every message.

    Picking a task runs it straight away; there is no Run button.
    """
    return html.Div(
        [
            html.H3('Planner and Task Selection', className='card-title'),
            html.Label('Planner', className='field-label'),
            dcc.Dropdown(
                id='algorithm-dropdown',
                options=[{'label': label, 'value': key}
                         for key, label in adapters],
                # Shows the planner already loaded, so refreshing the
                # page doesn't unload it; on a fresh start that's none,
                # and nothing loads until one is picked.
                value=current_algorithm, placeholder='(select planner)',
                clearable=False),
            html.Label('Task folder', className='field-label'),
            dcc.Dropdown(
                id='task-folder-dropdown',
                options=[{'label': f, 'value': f}
                         for f in list_task_folders()]),
            html.Label('Task file', className='field-label'),
            dcc.Dropdown(id='task-file-dropdown'),
            html.Div(
                [
                    html.Button('Previous', id='prev-task-button',
                                n_clicks=0),
                    html.Button('Next', id='next-task-button', n_clicks=0),
                ],
                className='nav-row'),
            html.Label('Logs', className='field-label log-label'),
            html.Div(id='message-log', className='message-log'),
        ],
        className='card side-column selection-column')


def build_simulation_column():
    """Middle: the animation and its playback controls."""
    return html.Div(
        [
            html.Div(
                [
                    html.H3('Trajectory Replay', className='card-title'),
                    # Save to webapp/downloads/ (see exporter.py).
                    html.Div(
                        [
                            _export_button('export-png-button', 'image',
                                           'Save current frame (PNG)'),
                            _export_button('export-frames-button', 'images',
                                           'Save all frames (PNGs)'),
                            _export_button('export-gif-button', 'gif',
                                           'Save as GIF'),
                        ],
                        className='export-buttons'),
                ],
                className='card-header'),
            html.Img(id='sim-view', src='/video_feed', className='sim-view'),
            html.Div(
                [
                    html.Button(REWIND_SYMBOL, id='rewind-button',
                                n_clicks=0, title='Back 10 frames',
                                className='glyph'),
                    # Shows play or pause; the label follows the state.
                    html.Button(PLAY_SYMBOL, id='play-button', n_clicks=0,
                                title='Play / pause', className='glyph'),
                    html.Button(SKIP_SYMBOL, id='skip-button', n_clicks=0,
                                title='Forward 10 frames', className='glyph'),
                    html.Button('\u21bb', id='replay-button', n_clicks=0,
                                title='Replay from the start'),
                    html.Button('1x', id='speed-button', n_clicks=0,
                                title='Playback speed',
                                className='speed-button'),
                    html.Div(
                        dcc.Slider(id='frame-slider', min=0, max=0, step=1,
                                   value=0, allow_direct_input=False,
                                   tooltip={'placement': 'bottom'}),
                        className='frame-slider'),
                ],
                className='transport'),
            html.Div(id='playback-status', className='message'),
        ],
        className='card simulation-column')


def build_settings_column():
    """Right: display preferences that don't change what is simulated."""
    return html.Div(
        [
            html.H3('View', className='card-title'),
            html.Div(
                html.Span('Annotations', className='section-title'),
                className='section-header'),
            dcc.Checklist(
                id='display-checklist',
                options=[
                    {'label': ' Show trajectories', 'value': 'trajectories'},
                    {'label': ' Show labels', 'value': 'labels'},
                    {'label': ' Show caption', 'value': 'caption'},
                ],
                value=[],
                className='checklist'),
            html.Div(
                [
                    html.Span('Camera View', className='section-title'),
                    html.Button('Reset default view', id='reset-camera-button',
                                n_clicks=0, className='link-button'),
                ],
                className='section-header'),
            html.Details(
                [
                    # style.css shows whichever text matches open/closed.
                    html.Summary([
                        html.Span('Show center point', className='when-closed'),
                        html.Span('Hide center point', className='when-open'),
                    ]),
                    html.Div(
                        [_center_input(axis, value) for axis, value in
                         zip('xyz', DEFAULT_CAMERA['target'])],
                        className='camera-row center-row'),
                ],
                className='center-settings'),
            _camera_row('Yaw', 'yaw'),
            _camera_row('Pitch', 'pitch'),
            _camera_row('Distance', 'distance'),
            html.Div(
                html.Span('Highlight Objects', className='section-title'),
                className='section-header'),
            # Only shown after a run that ended in a collision.
            dcc.Checklist(
                id='collision-checklist',
                options=[{'label': ' Highlight collided links',
                          'value': 'collision'}],
                value=['collision'],
                className='checklist highlight-checklist collision-checklist',
                style={'display': 'none'}),
            html.Div(
                [
                    html.Button('Deselect all', id='highlight-clear-button',
                                n_clicks=0),
                    # Also only shown after a run that ended in a collision.
                    html.Button('Highlight collided objects',
                                id='highlight-collided-button', n_clicks=0,
                                style={'display': 'none'}),
                ],
                className='highlight-actions'),
            html.Div(
                [
                    _highlight_column('highlight-arms'),
                    _highlight_column('highlight-obstacles'),
                ],
                className='highlight-columns'),
        ],
        className='card side-column')


def _export_button(button_id, icon, title):
    return html.Button(html.Img(src=f'/assets/icons/{icon}.svg', alt=title),
                       id=button_id, n_clicks=0, title=title,
                       className='icon-button')


def _highlight_column(checklist_id):
    """Arms on the left, obstacles on the right; options are filled in per
    task by the highlight callbacks."""
    return html.Div(
        dcc.Checklist(id=checklist_id, options=[], value=[],
                      className='checklist highlight-checklist'),
        className='highlight-column')


def _center_input(axis, value):
    """'x [  ]' -- one coordinate of the point the camera looks at, with
    up/down buttons that step it by 0.1. A text field rather than a number
    one, so update_camera_center can always show one decimal."""
    return html.Div(
        [
            html.Span(axis, className='center-axis'),
            html.Div(
                [
                    dcc.Input(id=f'camera-center-{axis}', type='text',
                              debounce=True,
                              value=f'{value:.1f}',
                              className='center-input'),
                    _step_buttons(f'camera-center-{axis}', 0.1),
                ],
                className='center-box'),
        ],
        className='center-field')


def _step_buttons(id_prefix, step):
    """Stacked up/down chevron buttons, '<id_prefix>-up' and '-down'."""
    return html.Div(
        [
            html.Button(id=f'{id_prefix}-up', n_clicks=0, title=f'+{step:g}',
                        className='step-button up'),
            html.Button(id=f'{id_prefix}-down', n_clicks=0,
                        title=f'-{step:g}', className='step-button down'),
        ],
        className='step-buttons')


def _camera_row(label, key):
    """Label | slider | current value with up/down arrows, on one line.
    The value text is filled in by the update_camera callback."""
    low, high, step = CAMERA_RANGES[key]
    return html.Div(
        [
            html.Span(label, className='camera-label'),
            html.Div(
                dcc.Slider(id=f'camera-{key}', min=low, max=high, step=step,
                           value=DEFAULT_CAMERA[key], marks=None,
                           allow_direct_input=False),
                className='camera-slider'),
            html.Span(id=f'camera-{key}-value', className='camera-value'),
            _step_buttons(f'camera-{key}', step),
        ],
        className='camera-row')


def build_run_tab():
    # Imported here: sim_manager connects PyBullet on import.
    from ..sim_manager import sim_manager

    adapters = list_adapters()
    return html.Div([
        html.H2('UR5 Multi-arm Planner Simulation', className='app-title'),
        html.Div(
            [
                # Selection and Trajectory Replay share a row so they
                # stretch to the same height (the log fills the extra).
                html.Div(
                    [
                        build_selection_column(adapters,
                                               sim_manager.algorithm_key),
                        build_simulation_column(),
                    ],
                    className='paired-columns'),
                build_settings_column(),
            ],
            className='columns'),
        # Callback plumbing, not shown.
        dcc.Store(id='algorithm-echo'),
        dcc.Store(id='load-task-echo'),
        dcc.Store(id='message-log-count', data=0),
        dcc.Store(id='message-log-scroll'),
        dcc.Store(id='seek-echo'),
        dcc.Store(id='display-echo'),
        dcc.Store(id='collision-echo'),
        dcc.Store(id='export-echo'),
        dcc.Store(id='highlight-echo'),
        dcc.Interval(id='status-poll', interval=500, n_intervals=0),
    ], className='app')
