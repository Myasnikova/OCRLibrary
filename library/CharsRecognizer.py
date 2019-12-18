﻿from library.core import LabImage
from PIL import Image, ImageDraw, ImageOps
from library.TextProfiler import *
from library.SymbolImage import *
import math


class CharsRecognizer(LabImage):
    """
    Класс распознования символов на изображении с указанием параметров шрифта
    """
    def __init__(self, path=None, image=None):
        super(CharsRecognizer, self).__init__(image=image, path=path)

        if getattr(self, 'letters_coords', None) is None:
            self.letters_coords = TextProfiler(image=image).get_text_segmentation()
        if getattr(self, 'bin_matrix', None) is None:
            self.bin_matrix = BinaryImage(path=path, image=image).cristian_binarisation().bin_matrix
        self.tryToRecognizeWithFont(fontSize=32)


    def tryToRecognizeWithFont(self, font=None, fontSize=None):
        symbols = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.symbols_features= FontCharacteristics(symbols, font, fontSize).calc_characteristics()
        rec_chars = []
        for row in self.letters_coords:
            for i in row:
                char = self.bin_matrix[i[0][1]:i[2][1],i[0][0]:i[1][0]]
                img = Image.fromarray(np.uint8(char), 'L')
                sym_characteristics = SymbolImage(image=LabImage(pilImage=img)).calc_characteristics()
                dist_array = {}
                for c in self.symbols_features.symbol_list:
                     c_features = self.symbols_features.symbol_characteristics[c]
                     c_features = np.array(([c_features['norm_weight'], c_features['norm_center'][0], c_features['norm_center'][1], c_features['norm_moment'][0], c_features['norm_moment'][1]]))
                     cur_features = np.array(([sym_characteristics['norm_weight'], sym_characteristics['norm_center'][0], sym_characteristics['norm_center'][1], sym_characteristics['norm_moment'][0], sym_characteristics['norm_moment'][1]]))
                     dist = np.linalg.norm(c_features-cur_features)
                     dist_array[c] = 1 - dist
                sorted_dist = sorted(dist_array.items(), key=lambda kv: kv[1], reverse=True)
                rec_chars.append(sorted_dist[0])
                dist_array.clear()
        print(rec_chars)
        self.recognized_chars = rec_chars
        return self

def test():
    lab_img = LabImage("pictures_for_test/text.bmp")
    lab_img = TextProfiler(lab_img)
    lab_img.get_text_segmentation()
    img = CharsRecognizer(image=lab_img)
    img.show()
    print(img.symbol_characteristics)
test()

