import os
import sys

from PIL import Image
import numpy as np

import time

from exceptions import ResultNotExist


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
    def __init__(self, path=None, image=None):
        self.path = path
        self.result = None

        if path is not None:

            if not os.path.isabs(path):
                self.path = os.path.normpath(os.path.join(sys.path[0], path))

            self.orig = Image.open(self.path).convert("RGB")
            self.size = self.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)

        elif image is not None:
            self.orig = image.orig
            self.size = image.size
            self.height, self.width = image.size
            self.rgb_matrix = image.rgb_matrix

    def read(self, path: str):
        self.path = path

        self.orig = Image.open(path).convert("RGB")
        self.size = self.orig.size
        self.height, self.width = self.size
        self.rgb_matrix = np.array(self.orig)

    def show(self):
        if self.result is None:
            self.orig.show()
        else:
            self.result.show()

    def to_grayscale(self, image):
        new_img = image.convert('L')
        return new_img

    def calc_grayscale_matrix(self):
        gray_matrix = np.sum(self.rgb_matrix, axis=2) // 3
        self.grayscale_matrix = gray_matrix

    def save(self, name: str):
        if self.result is not None:
            self.result.save(name)
        else:
            raise ResultNotExist("No such results for saving it to {}".format(name))

