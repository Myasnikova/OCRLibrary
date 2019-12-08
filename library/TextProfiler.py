from core import LabImage
from PIL import Image, ImageDraw, ImageOps
import numpy as np
import math


# возвращает вертикальный профиль изображения (проекция на ось y)
def get_y_profile(img):
    h = img.size[1]
    arr = np.asarray(img)
    prof = []
    for y in range(h):
        prof.append(np.sum(arr[y]))
    return prof


# возвращает горизонтальный профиль изображения (проекция на ось x)
def get_x_profile(img):
    w = img.size[0]
    arr = np.asarray(img).transpose()
    prof = []
    for x in range(w):
        prof.append(np.sum(arr[x]))
    return prof


# считает нули в профиле (для если не разделяются бквы увеличить порог)
def find_zero(arr):
    count = 0
    for i in arr:
        if i == 0:  # порог, для Times New Roman 12 подходит 250
            count += 1
    return count


# определяет координаты зон текста: для вертикального профиля - строки, для горизонтального - буквы
def get_zones(prof, r):
    w = len(prof)
    zone_coords = []
    flag = False
    zone_start = -1
    zone_finish = -1
    for i in range(w - r - 1):
        count = find_zero(prof[i:i + r])
        if count == 0 and not flag:
            flag = True
            zone_start = i
        if count == r and flag:
            flag = False
            zone_finish = i - 1
            zone_coords.append((zone_start - 1, zone_finish + 1))
            zone_start = -1
            zone_finish = -1
    return zone_coords


# находит координаты букв в строке, на вход профиль строки и координаты начала и конца строки
def get_letters_in_row(prof, y_start, y_end):
    r = 1
    zones = get_zones(prof, r)

    letters = [[(i[0], y_start), (i[1], y_start), (i[0], y_end), (i[1], y_end)] for i in zones]
    return letters


# находит координаты строк в тексте, на входгоризонтальный профиль текста
def get_rows_in_text(prof):
    zones = get_zones(prof, 1)
    return zones


# рисует сегментацию текста на изображении
def draw_segmented_row(img, zones):
    draw = ImageDraw.Draw(img)

    for x in zones:
        draw.line((x[0][0], x[0][1], x[1][0], x[1][1]), fill=128, width=1)
        draw.line((x[2][0], x[2][1], x[3][0], x[3][1]), fill=128, width=1)

        draw.line((x[0][0], x[0][1], x[2][0], x[2][1]), fill=128, width=1)
        draw.line((x[1][0], x[1][1], x[3][0], x[3][1]), fill=128, width=1)

    # img.show()


# класс для сегментации текстовых изображений
class TextProfiler(LabImage):
    def __init__(self, image=None):
        super().__init__()

        if image is not None:
            self.orig = image.orig
            self.gray_image = image.gray_image
            self.size = image.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)
            self.path = image.path

        new_img = self.orig.convert('L')
        self.work_image = ImageOps.invert(new_img)
        self.letters_coords = []  # координаты букв на изображении
        self.result = self.orig  # после вызова метода get_text_segmentation записывается сегментарованное изображение

    # находит координаты символов на изображении
    def get_text_segmentation(self):
        image = self.work_image

        y_profile = get_y_profile(image)
        rows = get_rows_in_text(y_profile)
        for i in rows:
            row_img = image.crop((0, i[0], image.size[0], i[1]))
            x_profile = get_x_profile(row_img)
            letters_in_row = get_letters_in_row(x_profile, i[0], i[1])
            draw_segmented_row(self.result, letters_in_row)
            self.letters_coords.append(letters_in_row)
        # self.result.show()
        return self.letters_coords


def test():
    lab_img = LabImage("pictures_for_test/text.bmp")
    text_profile = TextProfiler(lab_img)
    text_profile.get_text_segmentation()

#test()
