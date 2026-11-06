"""Algorithm adapter interface: turn a Task into planned joint-config waypoints."""


class AlgorithmAdapter:
    # False when plan() runs entirely outside the webapp's PyBullet client
    # (e.g. in a subprocess), so SimulationManager can plan without holding
    # its lock and the video feed keeps streaming meanwhile.
    uses_sim = True

    def __init__(self):
        # Set by plan() once it finishes, read by SimulationManager.
        # obstacle_frames: per-frame {obs_id: position} for moving obstacles,
        #     or None when obstacles stay where the task put them.
        # success: whether the goal markers should turn green at the end.
        # result_message: status-bar summary, or None for the default one.
        # collided: sim_manager object keys ('arm:0', 'obs:s1', ...) of the
        #     objects in the episode's collision, [] if there was none.
        # collided_links: [(arm index, link index), ...] in contact.
        # collision_frame: frame index the collision was detected at, or None.
        self.obstacle_frames = None
        self.success = True
        self.result_message = None
        self.collided = []
        self.collided_links = []
        self.collision_frame = None

    def prepare(self):
        """Do any slow one-off setup (e.g. loading a checkpoint). Called in
        the background as soon as the algorithm is picked, so the first
        Run doesn't pay for it. Adapters are reused across runs."""

    def close(self):
        """Release whatever prepare() set up."""

    def check_replay(self, task, frames, ur5_group):
        """Optional pass over the planned frames in the webapp's own scene,
        before playback (e.g. to find collisions and set collided,
        collided_links, collision_frame). It may shorten `frames` in place
        to end the replay early. Called with the scene locked; whatever it
        leaves posed there is reset by the first frame."""

    def plan(self, task, ur5_group):
        """Yield successive joint-config waypoints (flattened across all arms).

        Implementations call `ur5_group.setup(...)` against `task`
        themselves -- `ur5_group` already shares the caller's PyBullet
        client, so this needs no client/connection handling of its own.
        """
        raise NotImplementedError
