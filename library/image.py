from ContouredImage import ContouredImage
from FilteredImage import FilteredImage
from CharsRecognizer import *
import PIL as pl


class OCRImage:
    """
    Класс предоставляющий функционал для обработки изображения следующими способами:
        - :class:`~image.get_grayscale_image` -- получение монохромного изображения
        - :class:`~image.get_binary_image` -- получение бинаризованного изображения
        - :class:`~image.get_filtered_image` -- получение отфильтрованного изображения
        - :class:`~image.get_contoured_image` -- получение контурного изображения
        - :class:`~image.get_text_profiled_image` -- выделение символов в текстовом изображения
        - :class:`~image.get_text_recognized_image` -- распознавание текста на изображении
    """
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
        """
        Отображает результат последнего преобразования изображения,
        при отсутствии такового отображает исходное изображение
        """
        self.lab_image.show()

    def save(self, path: str):
        """
        Сохраняет изображение обработанное в виде BMP изображения под заданным в path путём к файлу

        :param path: путь к файлу для сохранения
        :type path: str
        """
        self.lab_image.save(path)

    def get_grayscale_image(self):
        """
        Возвращает изображение в оттенках серого
        :return: :class:`~PIL.Image` -- изображение в оттенках серого

        :raises: ValueError
        """
        self.result = self.lab_image.gray_image
        return self.result


    def get_binary_image(self, method=None, _rsize=3, _Rsize=15, _eps=15, _w_size=15, _k=0.5):
        """
        Возвращает изображение бинаризованное выбранным методом:
            - 1 - для метода Эйквила
            - 2 - для метода Кристиана

        :param method: метод бинаризации
        :type method: int or None
        :param _rsize: размер малого окна
        :type _rsize: int
        :param _Rsize: размер большего окна
        :type _Rsize: int
        :param _eps: величина отклонения для математических ожиданий чёрного и белого, в пределах которого можно считать \
        , что они отличются несущественно
        :type _eps: int
        :param _w_size: размер окна
        :type _w_size: inr
        :param _k: коэффициент, отвечающий за чувствительность бинаризатора
        :type _k: float

        :return: :class:`~PIL.Image` -- бинаризованное изображение

        :raises: ValueError
        """
        if method is None:
            ValueError('Undefined method: {}'.format(method))
            return None
        if method == 1:  # метод эйквила
            self.lab_image = self.binary_image_object.eikvil_binarisation(rsize=_rsize, Rsize=_Rsize, eps=_eps)
            return self.lab_image.result
        if method == 2:  # метод кристиана
            self.lab_image = self.binary_image_object.cristian_binarisation(w_size=_w_size, k=_k)
            return self.lab_image.result

    def get_filtered_image(self, method=None, rank=None, wsize=3):
        """
        Возвращает изображение отфильтрованное выбранным методом:
            - 1 - для медианного фильтра
            - 2 - для взвешенного рангового фильтра
            - 3 - для раногового фильтра

        :param method: метод фильтрации
        :type method: int or None
        :param rank: ранг фильтра
        :type rank: int
        :param wsize: размер окна фильтрации
        :type wsize: int

        :return: :class:`~PIL.Image` -- бинаризованное изображение

        :raises: ValueError
        """
        if method is None or (rank is None and method != 1):
            ValueError('Undefined method or rank')
            return None
        if method == 1:  # медианный фильтр
            self.lab_image = self.filtered_image_object.median_filter(wsize=wsize)
            return self.lab_image.result
        if method == 2:  # взвешенный фильтр
            self.lab_image = self.filtered_image_object.weighted_rank_filter(rank=rank, wsize=wsize)
            return self.lab_image.result
        if method == 3:  # ранговый фильтр
            self.lab_image = self.filtered_image_object.rank_filter(rank=rank, wsize=wsize)
            return self.lab_image.result
        raise ValueError('Undefined method: {}'.format(method))

    def get_contoured_image(self, method=None, t=None):
        """
        Возвращает контурное изображение вычисленное выбранным методом:
            - 1 - для оператора Собеля
            - 2 - для оператора Прюита

        :param method: метод бинаризации
        :type method: int or None
        :param t: порог
        :type t: int

        :return: :class:`~PIL.Image` -- бинаризованное изображение

        :raises: ValueError
        """
        if method is None or(t is None and method == 1):
            raise ValueError('Undefined threshold or method')
        if method == 1:  # оператор собеля
            self.lab_image = self.contoured_image_object.sobel_operator(t)
            return self.lab_image.result
        if method == 2:  # оператор прюита
            self.lab_image = self.contoured_image_object.prewitt_operator()
            return self.lab_image.result
        raise ValueError('Undefined method: {}'.format(method))


    def get_text_profiled_image(self, text="Привет мир", font_size=36, font='TNR.ttf', image_size=(600, 600), filename="text"):
        """
        Возвращает изображение с выделенными сегментами символов на сгенерированном изображении теста

        :param text: текст
        :type text: str or None
        :param font_size: размер шрифта
        :type font_size: int
        :param font: путь до файла шрифта
        :type font: str or None
        :param image_size: размер символа
        :type image_size: tuple
        :param filename: путь до файла сгенерированного текста
        :type filename: str or None

        :return: :class:`~PIL.Image` -- бинаризованное изображение

        :raises: ValueError
        """

        createText(text, font_size, font, image_size, filename)
        self.lab_image = LabImage("pictures_for_test/" + filename + ".bmp")
        self.text_profiler_object = TextProfiler(image=self.lab_image)
        self.lab_image = self.text_profiler_object.get_text_segmentation()
        return self.lab_image.result

    def get_text_recognized_image(self, text="Привет мир", font_size=36, font='TNR.ttf', image_size=(600, 600), filename="text"):
        """
        Возвращает распознаный на сгенерированном изображении текст

        :param text: текст
        :type text: str or None
        :param font_size: размер шрифта
        :type font_size: int
        :param font: путь до файла шрифта
        :type font: str or None
        :param image_size: размер символа
        :type image_size: tuple
        :param filename: путь до файла сгенерированного текста
        :type filename: str or None

        :return: str -- распознаный текст

        :raises: ValueError
        """
        createText(text, font_size, font, image_size, filename)
        self.lab_image = LabImage("pictures_for_test/"+filename+".bmp")
        self.chars_recognizer_object = CharsRecognizer(image=self.lab_image, font=font, font_size=font_size)
        self.lab_image = self.chars_recognizer_object
        return self.lab_image.recognized_string

    def get_filtered_image(self, method=None, rank=None, wsize=3):
        """
        Возвращает отфильтрованное изображение вычисленное выбранным методом:
            - 1 - медианный фильтр
            - 2 - взвешенный ранговый фильтр
            - 3 - ранговый фильтр

        :param method: метод бинаризации
        :type method: int or None
        :param rank: ранг фильтрации
        :type rank: int
        :param wsize: размер окна фильтрации
        :type wsize: int

        :return: :class:`~PIL.Image` -- фильтрованное изображение

        :raises: ValueError
        """
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

