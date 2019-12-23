import os
import sys

from PIL import Image,ImageOps
import numpy as np

import time

from exceptions import ResultNotExist, NameNotPassed


class LabImage:
    """
    Базовый класс для работы с изображением.

    Может быть инициализирован следующими способами:
        - передачей параметра **path**
        - передачей существующего экземпляра класса :class:`~core.LabImage`
        - передачей существующего экземпляра класса :class:`~PIL.Image`
        - иниализация пустыми параметрами с дальнейшим вызовом функции :meth:`~LabImage.read`
    """
    def __init__(self, path=None, image=None, pilImage=None):
        """
        Инициализация объекта класса LabImage

        :param path: путь до изображения
        :type path: str or None
        :param image: экземпляр класса :class:`~core.LabImage`
        :type image: :class:`~core.LabImage` or None
        :param pilImage: экземпляр класса :class:`~core.LabImage`
        :type pilImage: :class:`~core.LabImage`
        """
        self.path = path
        self.result = None
        self.grayscale_matrix = None

        if path is not None:

            if not os.path.isabs(path):
                self.path = os.path.normpath(os.path.join(sys.path[0], path))

            self.orig = Image.open(path).convert("RGB")
            self.size = self.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)
            self.gray_image = self.orig.convert('L')
            self.calc_grayscale_matrix()

        elif image is not None:
            for k, v in image.__dict__.items():
                setattr(self, k, v)

        elif pilImage is not None:
            self.orig = pilImage
            self.size = self.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)
            self.gray_image = self.orig.convert('L')
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


    def calc_grayscale_matrix(self):
        """
        Производит расчёт полутоновой матрицы исходного изображения и сохраняет её во внутреннюю переменную
        """
        if(len(self.rgb_matrix.shape)>2):
            gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        else:
             gray_matrix = self.rgb_matrix 
        self.grayscale_matrix = gray_matrix

    def save(self, name: str):
        """
        Сохраняет изображение из внутренней переменной result в виде BMP изображения под заданным в name именем

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

