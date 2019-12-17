from PIL import Image, ImageFont, ImageDraw
import numpy as np

import csv

from library.core import LabImage
from library.BinaryImage import BinaryImage
from library.exceptions import ResultNotExist, NameNotPassed


class SymbolImage(LabImage):
    """
    Класс осуществляющий выделение символьных признаков для заданного изображения
    """
    def __init__(self, path=None, image=None):
        super(SymbolImage, self).__init__(path=path, image=image)

        if getattr(self, 'bin_matrix', None) is None:
            # TODO надо бы выбрать способ бинаризации по умолчанию
            self.bin_matrix = BinaryImage(path=path, image=image).eikvil_binarization().bin_matrix
            # self.bin_matrix = self.grayscale_matrix

    def calc_characteristics(self):
        m, n = self.bin_matrix.shape

        weight = np.sum(self.bin_matrix) // 255
        norm_weight = weight / (self.height * self.width)

        x_center = np.sum([x * f for (x, y), f in np.ndenumerate(self.bin_matrix)]) // (weight * 255)
        y_center = np.sum([y * f for (x, y), f in np.ndenumerate(self.bin_matrix)]) // (weight * 255)

        norm_x_center = (x_center - 1) / (m - 1)
        norm_y_center = (y_center - 1) / (n - 1)

        x_moment = np.sum([f * (x - x_center) ** 2 for (x, y), f in np.ndenumerate(self.bin_matrix)]) // 255
        y_moment = np.sum([f * (y - y_center) ** 2 for (x, y), f in np.ndenumerate(self.bin_matrix)]) // 255

        norm_x_moment = x_moment / (m ** 2 + n ** 2)
        norm_y_moment = y_moment / (m ** 2 + n ** 2)

        return {'weight': weight, 'norm_weight': norm_weight,
                'center': (x_center, y_center),
                'norm_center': (norm_x_center, norm_y_center),
                'moment': (x_moment, y_moment),
                'norm_moment': (norm_x_moment, norm_y_moment)}


class FontCharacteristics:
    def __init__(self, symbols: list or tuple or str, font=None, font_size=None, symbol_size=None):
        self.font = font or 'TNR.ttf'
        self.font_size = font_size or 52
        self.symbol_size = symbol_size or (50, 50)
        self.symbol_list = symbols

        self.symbol_characteristics = {}

        self.create_symbol_images()

    def create_symbol_images(self) -> None:
        for sym in self.symbol_list:
            im = Image.new('L', self.symbol_size, color='white')
            d = ImageDraw.Draw(im)
            f = ImageFont.truetype(self.font, self.font_size)
            mw, mh = self.symbol_size
            w, h = d.textsize(sym, font=f)
            d.text((((mw - w) // 2), (mh - h) // 2), sym, font=f)
            if sym.isupper() and not sym.islower():
                im.save(sym + '_upper.bmp')
            elif sym.islower() and not sym.isupper():
                im.save(sym.upper() + '_lower.bmp')
            else:
                im.save(sym + '.bmp')

    def calc_characteristics(self):
        for sym in self.symbol_list:
            if sym.isupper() and not sym.islower():
                im = SymbolImage(sym + '_upper.bmp')
            elif sym.islower() and not sym.isupper():
                im = SymbolImage(sym.upper() + '_lower.bmp')
            else:
                im = SymbolImage(sym + '.bmp')
            self.symbol_characteristics[sym] = im.calc_characteristics()

        return self

    def to_csv(self, name: str):
        if name != '':
            if self.symbol_characteristics != {}:
                with open(name, "w", newline='') as csv_file:
                    writer = csv.writer(csv_file, delimiter=',')
                    writer.writerow(('symbol',
                                     'weight', 'norm_weight',
                                     'x', 'y',
                                     'norm_x', 'norm_y',
                                     'hor_ax_moment', 'ver_ax_moment',
                                     'norm_hor_ax_moment', 'norm_ver_ax_moment'))
                    for k, v in self.symbol_characteristics.items():
                        line = ()
                        for sub_el in v.values():
                            if type(sub_el) is tuple:
                                line += sub_el
                            else:
                                line += (sub_el,)
                        writer.writerow((k,) + line)
            else:
                raise ResultNotExist("No such results for saving it to {}".format(name))
        else:
            raise NameNotPassed("Name of file must contain some symbols")


FontCharacteristics("ABCabc+=jJ").calc_characteristics().to_csv('result.csv')
