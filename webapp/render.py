"""Offscreen PyBullet rendering: camera math + getCameraImage -> JPEG bytes.

The rest of this codebase (UR5, UR5Group, birrt, ...) never passes a
physicsClientId around -- it always relies on there being exactly one
implicit pybullet connection. This module follows the same convention on
purpose, since SimulationManager only ever opens a single client.

Renders with ER_BULLET_HARDWARE_OPENGL rather than the default
ER_TINY_RENDERER: SimulationManager connects via p.GUI (window immediately
hidden -- see sim_manager._hide_own_windows) specifically so a real OpenGL
context backs getCameraImage. Measured ~7x faster on this machine (~0.03s
vs ~0.18s per 640x480 frame) -- TinyRenderer's software rasterization was
the actual bottleneck behind slow-feeling playback.
"""
import io

import numpy as np
import pybullet as p
from PIL import Image, ImageDraw, ImageFont

DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480
DEFAULT_FOV = 60
DEFAULT_NEAR = 0.05
DEFAULT_FAR = 10.0
DEFAULT_JPEG_QUALITY = 80
# Starting view, also what "Reset default view" in the settings restores.
DEFAULT_CAMERA = {
    'target': [0.0, 0.0, 0.3],
    'distance': 2.0,
    'yaw': 50.0,
    'pitch': -35.0,
}


def compute_view_matrix(target, distance, yaw, pitch, roll=0):
    return p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=distance,
        yaw=yaw,
        pitch=pitch,
        roll=roll,
        upAxisIndex=2,
    )


def compute_projection_matrix(width, height,
                              fov=DEFAULT_FOV,
                              near=DEFAULT_NEAR,
                              far=DEFAULT_FAR):
    aspect = width / float(height)
    return p.computeProjectionMatrixFOV(fov, aspect, near, far)


def render_jpeg(camera, quality=DEFAULT_JPEG_QUALITY, **kwargs):
    """render_image, as JPEG bytes for the live stream."""
    buf = io.BytesIO()
    render_image(camera, **kwargs).save(buf, format='JPEG', quality=quality)
    return buf.getvalue()


def render_image(camera,
                 width=DEFAULT_WIDTH,
                 height=DEFAULT_HEIGHT,
                 labels=(),
                 occluders=(),
                 caption=(),
                 caption_note=None):
    """The view as an RGB PIL image.
    camera: dict with 'target' (x, y, z), 'distance', 'yaw', 'pitch'.
    labels, occluders: see draw_labels -- drawn on the image afterwards,
    since PyBullet's own debug text (like its debug lines) never shows up
    in getCameraImage. caption: lines of text for a box in the top right;
    caption_note: an extra last line in it, smaller and grey."""
    view_matrix = compute_view_matrix(
        target=camera['target'],
        distance=camera['distance'],
        yaw=camera['yaw'],
        pitch=camera['pitch'],
    )
    proj_matrix = compute_projection_matrix(width, height)
    _, _, rgb, _, segmentation = p.getCameraImage(
        width, height, view_matrix, proj_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
    )
    rgb_array = np.reshape(rgb, (height, width, 4))[:, :, :3].astype(np.uint8)
    image = Image.fromarray(rgb_array, 'RGB')
    if labels:
        image = draw_labels(
            image, labels, view_matrix, proj_matrix,
            np.reshape(segmentation, (height, width)), occluders)
    if caption:
        image = draw_caption(image, caption, caption_note)
    return image


def project_to_pixel(position, view_matrix, proj_matrix, width, height):
    """World (x, y, z) -> image (x, y), or None if behind the camera.
    PyBullet's matrices are flat, column-major 4x4s. Plain Python rather
    than numpy's @: the webapp env's MKL build crashes on matmul (see
    webapp/requirements.txt), and this is only a 4x4 anyway."""
    def apply(matrix, vector):
        return [sum(matrix[col * 4 + row] * vector[col] for col in range(4))
                for row in range(4)]

    clip = apply(proj_matrix, apply(view_matrix, [*position, 1.0]))
    if clip[3] <= 0:
        return None
    ndc_x, ndc_y = clip[0] / clip[3], clip[1] / clip[3]
    return ((ndc_x + 1) / 2 * width, (1 - ndc_y) / 2 * height)


def _label_font(size=12):
    for name in ('segoeuisb.ttf', 'segoeui.ttf', 'arial.ttf'):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


LABEL_FONT = _label_font()
CAPTION_NOTE_FONT = _label_font(10)


def _is_hidden(label, segmentation, occluders, view_matrix, proj_matrix):
    """Whether one of occluders is what the camera sees at the label's
    'probe' point, i.e. its object is behind another one there."""
    height, width = segmentation.shape
    pixel = project_to_pixel(label['probe'], view_matrix, proj_matrix,
                             width, height)
    if pixel is None:
        return False
    x, y = int(pixel[0]), int(pixel[1])
    if not (0 <= x < width and 0 <= y < height):
        return False
    seen = segmentation[y, x]
    return seen != label['body'] and seen in occluders


def draw_labels(image, labels, view_matrix, proj_matrix,
                segmentation=None, occluders=()):
    """labels: dicts with 'text', 'position' (world x, y, z), optional
    'color' (rgb in 0..1, shown as a dot before the text), and 'body' and
    'probe' (its body id and a world point on or inside it). Each is a
    small rounded tag centred just above its projected position.

    A label is drawn faint when its object is hidden behind another at
    'probe' -- the tag is drawn on top of everything, so at full strength
    it would seem to name the object in front. segmentation (body id per
    pixel, from getCameraImage) tells what the camera sees there;
    occluders are the body ids that count as hiding something (solid
    objects, not see-through markers or faded objects)."""
    width, height = image.size
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    for label in labels:
        pixel = project_to_pixel(label['position'], view_matrix,
                                 proj_matrix, width, height)
        if pixel is None:
            continue
        hidden = (segmentation is not None and 'probe' in label
                  and _is_hidden(label, segmentation, occluders,
                                 view_matrix, proj_matrix))
        alpha = 0.35 if hidden else 1.0
        text = label['text']
        left, top, right, bottom = draw.textbbox((0, 0), text, font=LABEL_FONT)
        dot = 10 if label.get('color') is not None else 0
        tag_w = right - left + dot + 12
        tag_h = bottom - top + 8
        x0 = pixel[0] - tag_w / 2
        y0 = pixel[1] - tag_h - 6
        draw.rounded_rectangle(
            (x0, y0, x0 + tag_w, y0 + tag_h), radius=tag_h / 2,
            fill=(255, 255, 255, int(215 * alpha)))
        if dot:
            r, g, b = (int(c * 255) for c in label['color'][:3])
            cy = y0 + tag_h / 2
            draw.ellipse((x0 + 6, cy - 3.5, x0 + 13, cy + 3.5),
                         fill=(r, g, b, int(255 * alpha)))
        draw.text((x0 + 6 + dot - left, y0 + 4 - top), text, font=LABEL_FONT,
                  fill=(43, 47, 54, int(255 * alpha)))
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')


def draw_caption(image, lines, note=None, margin=10, padding=8, line_gap=4):
    """A rounded box in the top-right corner with one line of text each,
    styled like the labels; note, if given, is a last line in a smaller grey
    font."""
    overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    rows = [(line, LABEL_FONT, (43, 47, 54, 255)) for line in lines]
    if note:
        rows.append((note, CAPTION_NOTE_FONT, (107, 114, 128, 255)))
    # Line heights from a fixed sample, so they don't vary with the text.
    heights = [font.getbbox('Ag')[3] - font.getbbox('Ag')[1]
               for _, font, _ in rows]
    boxes = [draw.textbbox((0, 0), text, font=font) for text, font, _ in rows]
    box_w = max(right - left for left, _, right, _ in boxes) + 2 * padding
    box_h = sum(heights) + (len(rows) - 1) * line_gap + 2 * padding
    x0 = image.size[0] - margin - box_w
    draw.rounded_rectangle(
        (x0, margin, x0 + box_w, margin + box_h), radius=6,
        fill=(255, 255, 255, 215))
    y = margin + padding
    for (text, font, fill), height in zip(rows, heights):
        draw.text((x0 + padding - font.getbbox('Ag')[0],
                   y - font.getbbox('Ag')[1]), text, font=font, fill=fill)
        y += height + line_gap
    return Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
