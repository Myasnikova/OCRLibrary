from core import LabImage
from BinaryImage import BinaryImage
from ContouredImage import ContouredImage
from TextProfiler import TextProfiler
from FilteredImage import FilteredImage
import numpy as np
import PIL as pl


class OCRImage:
    image = None

    def __init__(self, path):
        self.image = pl.Image.open(path)
        self.lab_image = LabImage(path)
        self.contoured_image_object = ContouredImage(image=self.lab_image)
        self.binary_image_object = BinaryImage(image=self.lab_image)
        self.filtered_image_object = FilteredImage(image=self.lab_image)
        self.text_profiler_object = TextProfiler(image=self.lab_image)

    def show_result(self):
        self.lab_image.show()

    def save(self, path):
        self.lab_image.save(path)

    def get_grayscale_image(self):
        return self.lab_image.gray_image

    def get_binary_image(self, method=None, _rsize=3, _Rsize=15, _eps=15, _w_size=15, _k=0.5):
        if method is None:
            ValueError('Undefined method: {}'.format(method))
            return None
        if method == 1:  # метод эйквила
            return self.binary_image_object.eikvil_binarization(rsize=_rsize, Rsize=_Rsize, eps=_eps)
        if method == 2:  # метод кристиана
            return self.binary_image_object.cristian_binarisation(w_size=_w_size, k=_k)

    def get_contoured_image(self, method=None, t=None):
        if t is None and method == 1:
            raise ValueError('Undefined threshold: {}'.format(t))
        if method == 1:  # оператор собеля
            return self.contoured_image_object.sobel_operator(t)
        if method == 2:  # оператор прюита
            return self.contoured_image_object.prewitt_operator()
        raise ValueError('Undefined method: {}'.format(method))

    def get_text_profiled_image(self):
        return self.text_profiler_object.get_text_segmentation()

    def get_filtered_image(self):
        return self.filtered_image_object.median_filter()


def main():
    img = OCRImage("pictures_for_test/text.bmp")
    #img.filtered_image_object.median_filter()
    img.binary_image_object.cristian_binarisation()
    img.show_result()


main()

