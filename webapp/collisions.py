"""Which arm links and objects are in a collision, and how to describe it.

Shared by the RL rollout (plan_worker, in WSL) and the OMPL replay check
(algorithms/ompl.py, in the webapp), so both report collisions the same
way. Works on whichever PyBullet client is the implicit one in the calling
process.

A contact is (arm index, its link, what it hit, that link). What it hit is
'arm:<i>' / 'obs:<id>' for another arm or an obstacle -- the sim_manager
object keys -- else 'the ground' or 'itself'; the second link is set for
another arm and for itself.
"""
import pybullet


def find_contacts(arms, obstacles, colliding, margin):
    """Contacts of the arms in `colliding` (indices into `arms`, the
    active UR5s), against each other and `obstacles` ({obs_id: obstacle}).
    Queried directly, mirroring UR5.check_collision, since its
    prev_collided_with can be stale (its ground check returns without
    updating it)."""
    bodies = {ur5.body_id: f'arm:{i}' for i, ur5 in enumerate(arms)}
    bodies.update({obs.body_id: f'obs:{obs_id}'
                   for obs_id, obs in obstacles.items()})
    contacts = []
    for i in colliding:
        ur5 = arms[i]
        found = []
        for body, key in bodies.items():
            if body == ur5.body_id:
                continue
            for point in pybullet.getClosestPoints(
                    ur5.body_id, body, distance=margin):
                if point[8] < margin:
                    found.append((i, point[3], key,
                                  point[4] if key.startswith('arm:')
                                  else None))
        if not found:  # body 0 is the plane, as in UR5
            found = [(i, point[3], 'the ground', None)
                     for point in pybullet.getClosestPoints(
                         ur5.body_id, 0, distance=0.0)]
        if not found:
            for link_a, link_b in ur5.link_pairs:
                if pybullet.getClosestPoints(
                        ur5.body_id, ur5.body_id, distance=0,
                        linkIndexA=link_a, linkIndexB=link_b):
                    found.append((i, link_a, 'itself', link_b))
        contacts.extend(found or [(i, None, 'itself', None)])
    return contacts


def _pairs(contacts):
    """Unique (arm index, what it hit), in order."""
    pairs = []
    for i, _, other, _ in contacts:
        if (i, other) not in pairs:
            pairs.append((i, other))
    return pairs


def collided_objects(contacts):
    """Object keys of everything in the contacts, arms and obstacles."""
    keys = []
    for i, other in _pairs(contacts):
        for key in (f'arm:{i}', other):
            if ':' in key and key not in keys:
                keys.append(key)
    return keys


def collided_links(contacts):
    """[[arm index, link index], ...] of every arm link in contact, on
    both sides of the contact -- what the webapp colors red."""
    links = set()
    for i, link_a, other, link_b in contacts:
        if link_a is not None:
            links.add((i, link_a))
        if other.startswith('arm:') and link_b is not None:
            links.add((int(other.split(':')[1]), link_b))
        elif other == 'itself' and link_b is not None:
            links.add((i, link_b))
    return [list(link) for link in sorted(links)]


def describe(contacts):
    """E.g. 'Arm 1 with Obstacle s2; Arm 2 with the ground'."""
    def name(key):
        if ':' not in key:
            return key
        kind, ident = key.split(':', 1)
        return (f'Arm {int(ident) + 1}' if kind == 'arm'
                else f'Obstacle {ident}')

    parts, seen = [], set()
    for i, other in _pairs(contacts):
        pair = frozenset((f'arm:{i}', other))
        if pair in seen:  # A-B and B-A are the same collision
            continue
        seen.add(pair)
        parts.append(f'{name(f"arm:{i}")} with {name(other)}')
    return '; '.join(parts) or 'unknown'
