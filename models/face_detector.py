import face_recognition

import interfaces


class FaceDetector(interfaces.Detector):

    def load_image(self, frame):
        return face_recognition.load_image_file(frame)

    def get_face_locations(self, frame) -> list[tuple]:
        return face_recognition.face_locations(frame)

    def get_face_encodings(self, frame, locations = None) -> list:
        return face_recognition.face_encodings(frame, locations)

    def compare_faces(self, know_encodings: list, unknow_encoding) -> list:
        return face_recognition.compare_faces(know_encodings, unknow_encoding)

    def get_face_distances(self, know_encodings: list, unknow_encoding):
        return face_recognition.face_distance(know_encodings, unknow_encoding)