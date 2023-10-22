import os

from .face_detector import FaceDetector
import interfaces


DATA_PATH = 'C:/Users/Deiver Vasquez/Documents/persona_projects/python/face-recognition/data'
IMAGES_PATH = f'{DATA_PATH}/images'

class DataManager(interfaces.Manager):

    def __init__(self, face_detector: interfaces.Detector, path: str = IMAGES_PATH):
        self._path = path
        self.face_detector = face_detector

        self._image_names = os.listdir(path)
        self._data = []
        self._names = []
        self._encodings = []

        self.set_data()

    def get_image_names(self) -> list[str]:
        return self._image_names

    def get_names(self) -> list[str]:
        return self._names

    def get_encodings(self) -> list:
        return self._encodings

    def get_data(self) -> list:
        return self._data

    def set_data(self) -> None:
        for name, encoding in zip(self.set_names(), self.set_encodings()):
            self._data.append({
                "name": name,
                "encoding": encoding,
            })

    def set_encodings(self) -> list:
        for image_name in self.get_image_names():
            image = self.face_detector.load_image(os.path.join(self._path, image_name))
            self._encodings.append(self.face_detector.get_face_encodings(image)[0])
        return self._encodings

    def set_names(self) -> list[str]:
        self._names = [
            image_name.split('.')[0]
            for image_name in self.get_image_names()
        ]

        # with open(f'{DATA_PATH}/names.txt', 'r') as file:
        #     lines = file.readlines()
        # self._names = [line.strip() for line in lines]

        return self._names
