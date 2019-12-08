import os
import sys

from PIL import Image
import numpy as np

import time

from library.exceptions import ResultNotExist, NameNotPassed


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            print('%r  %2.2f ms' % \
                  (method.__name__, (te - ts) * 1000))
        return result
    return timed


class LabImage:
    """
    Базовый класс для работы с изображением.

    Может быть инициализирован следующими способами:
        - передачей параметра path
        - передачей существующего экземпляра класса
        - иниализация пустыми параметрами с дальнейшим вызовом функции read
    """
    def __init__(self, path=None, image=None):
        """
        Инициализация объекта класса LabImage

        :param path: путь до изображения
        :type path: str or None
        :param image: экземпляр класса LabImage
        :type image: LabImage or None
        """
        self.path = path
        self.result = None
        self.grayscale_matrix = None

        if path is not None:

            if not os.path.isabs(path):
                self.path = os.path.normpath(os.path.join(sys.path[0], path))

            self.orig = Image.open(self.path).convert("RGB")
            self.size = self.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)

            self.calc_grayscale_matrix()

        elif image is not None:
            for k, v in image.__dict__.items():
                setattr(self, k, v)

    def read(self, path: str):
        """
        Считывает изображение, расположенное по пути path, и заполняет внутренние переменные класса

        :param path: путь до изображения
        :type path: str
        """
        self.path = path

        self.orig = Image.open(path).convert("RGB")
        self.size = self.orig.size
        self.height, self.width = self.size
        self.rgb_matrix = np.array(self.orig)

        self.calc_grayscale_matrix()

    def show(self):
        """
        Отображает изображение, сохранённое во внутренней переменной result;
        при отсутствии такового отображает исходное изображение
        """
        if self.result is None:
            self.orig.show()
        else:
            self.result.show()

    def to_grayscale(self, image):
        new_img = image.convert('L')
        return new_img

    def calc_grayscale_matrix(self):
        """
        Производит расчёт полутоновой матрицы исходного изображения и сохраняет её во внутреннюю переменную
        """
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        self.grayscale_matrix = gray_matrix

    def save(self, name: str):
        """
        Сохраняет изображение из внутренней переменной result под заданным в name именем

        :param name: имя файла, под которым следует сохранить изображение
        :type name: str
        """
        if name != '':
            if self.result is not None:
                self.result.save(name)
            else:
                raise ResultNotExist("No such results for saving it to {}".format(name))
        else:
            raise NameNotPassed("Name of file must contain some symbols")

