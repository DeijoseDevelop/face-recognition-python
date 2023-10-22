import cv2
from .frame import Frame
import interfaces


class CameraController(interfaces.Controller):

    def __init__(self) -> None:
        # Get a reference to webcam #0 (the default one)
        self.video_capture = cv2.VideoCapture(0)
        self._frame = Frame()

    def get_frame(self):
        ret, frame = self.video_capture.read()
        self._frame.set_value(frame)
        return self._frame

    def draw_rectangle(self, positions: tuple, thickness: int) -> None:
        (top, right, bottom, left) = positions
        cv2.rectangle(
            self._frame.get_value(),
            (left, top), (right, bottom),
            (0, 0, 255), thickness
        )

    def draw_text(self, name: str, positions: tuple, thickness: int) -> None:
        (top, right, bottom, left) = positions
        cv2.putText(
            self._frame.get_value(), name,
            (left + 6, bottom - 6),
            cv2.FONT_HERSHEY_DUPLEX,
            1.0, (255, 255, 255), thickness
        )

    def show_image(self) -> None:
        cv2.imshow('Webcam', self._frame.get_value())

    def close_videoCapture(self):
        # Release handle to the webcam
        self.video_capture.release()
        cv2.destroyAllWindows()