import cv2
import numpy as np

from models import (
    DataManager,
    FaceDetector,
    CameraController,
)


class Main(object):

    @staticmethod
    def init():
        camera_controller = CameraController()
        face_detector = FaceDetector()
        data_manager = DataManager(face_detector)

        known_face_encodings = data_manager.get_encodings()
        known_face_names = data_manager.get_names()

        while True:
            # Grab a single frame of video
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_frame = camera_controller\
                .get_frame()\
                .convert_to_rgb()\
                .get_rgb_value()

            # Find all the faces and face enqcodings in the frame of video
            face_locations = face_detector.get_face_locations(rgb_frame)
            face_encodings = face_detector.get_face_encodings(rgb_frame, face_locations)

            # Loop through each face in this frame of video
            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                # See if the face is a match for the known face(s)
                matches = face_detector.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"

                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_detector.get_face_distances(
                    known_face_encodings,
                    face_encoding
                )

                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                # Draw a box around the face
                camera_controller.draw_rectangle((top, right, bottom, left), 2)

                # Draw a label with a name below the face
                camera_controller.draw_rectangle((bottom, right, bottom - 35, left), cv2.FILLED)
                camera_controller.draw_text(name, (top, right, bottom, left), 1)

            camera_controller.show_image()

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        camera_controller.close_videoCapture()


if __name__ == "__main__":
    Main.init()