from core import LabImage
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math

def get_y_profile(img):
    """
    Получение вертикального профиля изображения

    :param img: изображение
    :type img: PIL.Image

    :return: numpy.ndarray -- массив с профилем изображения

    """
    h = img.size[1]
    arr = np.asarray(img, dtype=np.uint8)
    prof = []
    for y in range(h):
        prof.append(np.sum(arr[y]))
    return prof

def get_x_profile(img):
    """
    Получение горизонтального профиля изображения

    :param img: изображение
    :type img: PIL.Image

    :return: numpy.ndarray -- массив с профилем изображения

    """
    w = img.size[0]
    arr = np.asarray(img, dtype=np.uint8).transpose()
    prof = []
    for x in range(w):
        prof.append(np.sum(arr[x]))
    return prof


def find_zero(arr, t):
    """
    Подсчет нулей в профиле изображения

    :param arr: профиль
    :type arr: numpy.ndarray

    :param t: порог
    :type t: int

    :return: int -- количество нулей

    """
    count = 0
    for i in arr:
        if i <= t:  # порог, для Times New Roman 12 подходит 250
            count += 1
    return count


def get_zones(prof, r, t):
    """
    Опрделение координат зон текста: для вертикального профиля - строки, для горизонтального - буквы

    :param prof: профиль
    :type prof: numpy.ndarray

    :param t: порог
    :type t: int

    :param r: размер окна
    :type r: размер окна

    :return: int -- количество нулей

    """
    w = len(prof)
    zone_coords = []
    flag = False
    zone_start = -1
    zone_finish = -1
    for i in range(w - r - 1):
        count = find_zero(prof[i:i + r],t)
        if count == 0 and not flag:
            flag = True
            zone_start = i
        if count == r and flag:
            flag = False
            zone_finish = i
            zone_coords.append((zone_start - 1, zone_finish))
            zone_start = -1
            zone_finish = -1
    return zone_coords


def get_letters_in_row(prof, y_start, y_end,t):
    """
    Опрделение координат букв

    :param prof: профиль
    :type prof: numpy.ndarray

    :param y_start: координаты начала строки
    :type y_start: int

    :param y_end: координаты конца строки
    :type y_end: int

    :param t: порог
    :type t: int

    :return: numpy.ndarray -- координаты букв

    """
    r = 1
    zones = get_zones(prof, r,t)

    letters = [[(i[0], y_start), (i[1], y_start), (i[0], y_end), (i[1], y_end)] for i in zones]
    return letters

def get_rows_in_text(prof, t):
    """
    Опрделение координат строк

    :param prof: горизональный профиль
    :type prof: numpy.ndarray

    :param t: порог
    :type t: int

    :return: numpy.ndarray -- координаты строк

    """
    zones = get_zones(prof, 3, t)
    return zones


def draw_segmented_row(img, zones):
    """
    Отрисовка сегментации текста на буквы

    :param img: изображение
    :type img: PIL.Image

    :param zones: координаты букв
    :type zones: numpy.ndarray

    :return: PIL.Image - размеченное изображение

    """
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)

    for x in zones:
        draw.line((x[0][0], x[0][1], x[1][0], x[1][1]), fill=128, width=1)
        draw.line((x[2][0], x[2][1], x[3][0], x[3][1]), fill=128, width=1)

        draw.line((x[0][0], x[0][1], x[2][0], x[2][1]), fill=128, width=1)
        draw.line((x[1][0], x[1][1], x[3][0], x[3][1]), fill=128, width=1)

    #new_img.show()
    return new_img


# класс для сегментации текстовых изображений
class TextProfiler(LabImage):
    """
    Класс осуществляющий выделение букв в тексте
    """
    def __init__(self, path=None, image=None):
        """
        Инициализация объекта класса FilteredImage

        :param path: путь до изображения
        :type path: str or None
        :param image: экземпляр класса LabImage
        :type image: LabImage or None
        """
        super(TextProfiler, self).__init__(path=path, image=image)

        self.work_image = self.orig.convert('L')
        self.letters_coords = []  # координаты букв на изображении
        self.result = self.orig  # после вызова метода get_text_segmentation записывается сегментарованное изображение

    def get_text_segmentation(self, t=0):
        """
        Получение координат символов на изображении

        :param t: порог
        :type t: int

        :return: LabImage -- объект изображения

        """

        image = self.work_image
        y_profile = get_y_profile(image)
        rows = get_rows_in_text(y_profile, 250)
        r = 0

        for i in rows:
            r_size = i[1] - i[0]
            row_img = image.crop((0, i[0], image.size[0], i[1]))
            x_profile = get_x_profile(row_img)
            letters_in_row = get_letters_in_row(x_profile, i[0], i[1], t)

            k = 0
            l_size_prev = 0
            letter_part = []

            #для буквы ы

            for let in letters_in_row:
                l_size = let[1][0] - let[0][0]
                if (l_size_prev / 2.5) > l_size:
                    letter_part.append(k)
                    letters_in_row[k - 1][1] = letters_in_row[k][1]
                    letters_in_row[k - 1][3] = letters_in_row[k][3]
                l_size_prev = l_size
                k += 1
            letters_in_row = np.delete(letters_in_row, letter_part, axis=0)
            self.result = draw_segmented_row(self.result, letters_in_row)
            self.letters_coords.append(letters_in_row)


        return self


def test():
    lab_img = LabImage("pictures_for_test/B.bmp")
    text_profile = TextProfiler(lab_img)
    text_profile.get_text_segmentation()

#test()
