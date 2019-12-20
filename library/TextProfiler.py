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
def find_zero(arr, t):
    count = 0
    for i in arr:
        if i <= t:  # порог, для Times New Roman 12 подходит 250
            count += 1
    return count


# определяет координаты зон текста: для вертикального профиля - строки, для горизонтального - буквы
def get_zones(prof, r, t):
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


# находит координаты букв в строке, на вход профиль строки и координаты начала и конца строки
def get_letters_in_row(prof, y_start, y_end,t):
    r = 1
    zones = get_zones(prof, r,t)

    letters = [[(i[0], y_start), (i[1], y_start), (i[0], y_end), (i[1], y_end)] for i in zones]
    return letters


# находит координаты строк в тексте, на входгоризонтальный профиль текста
def get_rows_in_text(prof, t):
    zones = get_zones(prof, 3, t)
    return zones


# рисует сегментацию текста на изображении
def draw_segmented_row(img, zones):
    new_img = img.copy()
    draw = ImageDraw.Draw(new_img)

    for x in zones:
        draw.line((x[0][0], x[0][1], x[1][0], x[1][1]), fill=128, width=1)
        draw.line((x[2][0], x[2][1], x[3][0], x[3][1]), fill=128, width=1)

        draw.line((x[0][0], x[0][1], x[2][0], x[2][1]), fill=128, width=1)
        draw.line((x[1][0], x[1][1], x[3][0], x[3][1]), fill=128, width=1)

    new_img.show()


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

        #new_img = self.orig.convert('L')
        #self.work_image = ImageOps.invert(new_img)
        self.work_image = self.orig
        self.letters_coords = []  # координаты букв на изображении
        self.result = self.orig  # после вызова метода get_text_segmentation записывается сегментарованное изображение

    # находит координаты символов на изображении
    def get_text_segmentation(self, t=250):

        image = self.work_image
        y_profile = get_y_profile(image)
        rows = get_rows_in_text(y_profile, t)
        r = 0

        for i in rows:
            r_size = i[1] - i[0]
            row_img = image.crop((0, i[0], image.size[0], i[1]))
            x_profile = get_x_profile(row_img)
            letters_in_row = get_letters_in_row(x_profile, i[0], i[1], t)
            '''
            k = 0
            l_size_prev = 0
            letter_part=[]
            for let in letters_in_row:
                l_size = let[1][0]-let[0][0]
                dx = let[0][0]
                dy = let[0][1]
                let_img = image.crop((let[0][0], i[0], let[1][0], i[1]))
                pix = np.array(let_img)

                idx_col = np.argwhere(np.all(pix[..., :] == 0, axis=0))
                f_x = 0
                f_y = 0

                for j in range(1, idx_col.size):
                    d = idx_col[j][0]-idx_col[j-1][0]
                    if d > 1:
                        st_x = idx_col[j-1][0]
                        break

                for j in range(idx_col.size-1, 0, -1):
                    d = idx_col[j][0]-idx_col[j-1][0]
                    if d > 1:
                        f_x = idx_col[j][0]
                        break

                idx_row = np.argwhere(np.all(pix[..., :] == 0, axis=1))

                for j in range(1, idx_row.size):
                    d = idx_row[j][0]-idx_row[j-1][0]
                    if d > 1:
                        st_y = idx_row[j-1][0]
                        break
                for j in range(idx_row.size-1, 0, -1):
                    d = idx_row[j][0] - idx_row[j - 1][0]
                    if d > 1:
                        f_y = idx_row[j][0]
                        break
                letters_in_row[k] = [(st_x+dx, st_y+dy), (f_x+dx,st_y+dy), (st_x+dx, f_y+dy), (f_x+dx,f_y+dy)]
                if l_size_prev/2 > l_size:
                    letter_part.append([k])
                    letters_in_row[k - 1][1] = letters_in_row[k][1]
                    letters_in_row[k - 1][3] = letters_in_row[k][3]
                l_size_prev = l_size
                k += 1
            np.delete(letters_in_row, letter_part, axis=0)
            '''
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
            #draw_segmented_row(self.result, letters_in_row)
            #self.letters_coords.append(letters_in_row)

            # для близко стоящих букв
            '''
            for k in range(0, len(letters_in_row)-1):
                let = letters_in_row[k]
                let_next = letters_in_row[k+1]

                l_size = let[1][0] - let[0][0]
                l_size_next = let_next[1][0] - let_next[0][0]

                if (l_size / 1.5) > l_size_next:
                    m = min(x_profile[let[0][0]:let[0][0]+l_size])
                    ind_m = x_profile.index(m)
                    new_let = let
                    new_let[0] = m
                    new_let[0] = m
                    letters_in_row[k][1][0] = m
                    letters_in_row[k][2][0] = m
                    letters_in_row.insert(k+1, new_let)
            '''
            draw_segmented_row(self.result, letters_in_row)
            self.letters_coords.append(letters_in_row)


        return self


def test():
    lab_img = LabImage("pictures_for_test/B.bmp")
    text_profile = TextProfiler(lab_img)
    text_profile.get_text_segmentation()

#test()
