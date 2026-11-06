"""Flask route serving the live simulation view as an MJPEG stream.

Embedded in the Dash layout as a plain <img src="/video_feed">, which the
browser refreshes continuously on its own -- no Dash callback round-trip
needed for the video itself.
"""
import time

from .sim_manager import sim_manager

FRAME_INTERVAL_SECONDS = 1.0 / 30  # cap; actual rate is usually render-bound
RESEND_SECONDS = 1.0  # resend an unchanged frame this often, as a keepalive


def _mjpeg_generator():
    last_frame, last_sent = None, 0.0
    while True:
        frame = sim_manager.render_jpeg()  # cached while nothing changes
        now = time.monotonic()
        if frame is not last_frame or now - last_sent >= RESEND_SECONDS:
            yield (b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            last_frame, last_sent = frame, now
        time.sleep(FRAME_INTERVAL_SECONDS)


def register_video_feed(flask_app):
    from flask import Response

    @flask_app.route('/video_feed')
    def video_feed():
        return Response(
            _mjpeg_generator(),
            mimetype='multipart/x-mixed-replace; boundary=frame')
