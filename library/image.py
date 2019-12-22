from ContouredImage import ContouredImage
from FilteredImage import FilteredImage
from CharsRecognizer import *
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
        self.chars_recognizer_object = None

    def show_result(self):
        self.lab_image.show()

    def save(self, path):
        self.lab_image.save(path)

    def get_binary_image(self, method=None, _rsize=3, _Rsize=15, _eps=15, _w_size=15, _k=0.5):
        if method is None:
            ValueError('Undefined method: {}'.format(method))
            return None
        if method == 1:  # метод эйквила
            self.lab_image = self.binary_image_object.eikvil_binarization(rsize=_rsize, Rsize=_Rsize, eps=_eps)
            return self.lab_image.result
        if method == 2:  # метод кристиана
            self.lab_image = self.binary_image_object.cristian_binarisation(w_size=_w_size, k=_k)
            return self.lab_image.result

    def get_contoured_image(self, method=None, t=None):
        if method is None or(t is None and method == 1):
            raise ValueError('Undefined threshold or method')
        if method == 1:  # оператор собеля
            self.lab_image = self.contoured_image_object.sobel_operator(t)
            return self.lab_image.result
        if method == 2:  # оператор прюита
            self.lab_image = self.contoured_image_object.prewitt_operator()
            return self.lab_image.result
        raise ValueError('Undefined method: {}'.format(method))

    def get_text_profiled_image(self, text="Привет мир", font_size=36, font='TNR.ttf', image_size=(600, 600),
                                filename="text"):
        createText(text, font_size, font, image_size, filename)
        self.lab_image = LabImage("pictures_for_test/" + filename + ".bmp")
        self.text_profiler_object = TextProfiler(image=self.lab_image)
        self.lab_image = self.text_profiler_object.get_text_segmentation()
        return self.lab_image.result

    def get_text_recognized_image(self, text="Привет мир", font_size=36, font='TNR.ttf', image_size=(600, 600), filename="text"):
        createText(text, font_size, font, image_size, filename)
        self.lab_image = LabImage("pictures_for_test/"+filename+".bmp")
        self.chars_recognizer_object = CharsRecognizer(image=self.lab_image, font=font, font_size=font_size)
        self.lab_image = self.chars_recognizer_object
        return self.lab_image.recognized_string

    def get_filtered_image(self, method=None, rank=None, wsize=3):
        if method is None or (rank is None and method != 1):
            ValueError('Undefined method or rank')
            return None
        if method == 1:#медианный фильтр
            self.lab_image = self.filtered_image_object.median_filter(wsize=wsize)
            return self.lab_image.result
        if method == 2:#взвешенный фильтр
            self.lab_image = self.filtered_image_object.weighted_rank_filter(rank=rank, wsize=wsize)
            return self.lab_image.result
        if method == 3:#ранговый фильтр
            self.lab_image = self.filtered_image_object.rank_filter(rank=rank, wsize=wsize)
            return self.lab_image.result
        raise ValueError('Undefined method: {}'.format(method))


