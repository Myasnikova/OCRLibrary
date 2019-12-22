from core import LabImage
from PIL import Image, ImageDraw, ImageOps
from TextProfiler import *
from SymbolImage import *
import math
from BinaryImage import *

class CharsRecognizer(LabImage):
    """
    Класс распознования символов на изображении с указанием параметров шрифта
    """
    def __init__(self, path=None, image=None, font='TNR.ttf', font_size=36):
        """
        Инициализация объекта класса CharsRecognizer

        :param path: путь до изображения
        :type path: str or None

        :param image: экземпляр класса LabImage
        :type image: LabImage or None

        :param font: путь до файла шрифта
        :type font: str or None

        :param font_size: размер шрифта
        :type font_size: int
        """
        super(CharsRecognizer, self).__init__(image=image, path=path)

        invert_img = ImageOps.invert(image.orig)
        self.bin_matrix = np.asarray(invert_img.convert('L'), dtype=np.uint8)
        self.recognized_string = ""

        if getattr(self, 'letters_coords', None) is None:
            p=TextProfiler(LabImage(pilImage=self.orig)).get_text_segmentation()
            #self.result=p.result
            self.letters_coords = p.letters_coords
        self.tryToRecognizeWithFont(font=font, fontSize=font_size)


    def tryToRecognizeWithFont(self, font=None, fontSize=None,symbol_size=(50,50)):
        """
        Функция разпознавания символов в тексте

        :param path: путь до изображения
        :type path: str or None

        :param image: экземпляр класса LabImage
        :type image: LabImage or None

        :param font: путь до файла шрифта
        :type font: str or None

        :param fontSize: размер шрифта
        :type fontSize: int

        :param symbol_size: размер символа
        :type symbol_size: tuple

        :return: LabImage -- объект изображения

        """
        symbols = "АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯабвгдеёжзийклмнопрстуфхцчшщъыьэюя"
        self.recognized_string = ""
        self.symbols_features= FontCharacteristics(symbols, font, fontSize,symbol_size=symbol_size).calc_characteristics()
        rec_chars = []
        g=0
        for row in self.letters_coords:
            for i in row:
                char = self.bin_matrix[i[0][1]:i[2][1],i[0][0]:i[1][0]]
                idx = np.argwhere(np.all(char[..., :] == 0, axis=0))
                char = np.delete(char, idx, axis=1)
                idx = np.argwhere(np.all(char[..., :] == 0, axis=1))
                char = np.delete(char, idx, axis=0)

                img = Image.fromarray(np.uint8(char), 'L')
                (bw,bh) = symbol_size
                new_img = Image.new('L',(bw,bh), color='black')
                offset = ((bw - img.size[0]) // 2, (bh - img.size[1]) // 2)
                new_img.paste(img,offset)

                new_img.save("pictures_for_test/liza_letters/"+str(g)+".bmp")
                g=g+1
                sym_characteristics = SymbolImage(image=LabImage(pilImage=new_img)).calc_characteristics()
                dist_array = {}
                for c in self.symbols_features.symbol_list:
                     c_features = self.symbols_features.symbol_characteristics[c]
                     c_features = np.array(([c_features['norm_weight'], c_features['norm_center'][0], c_features['norm_center'][1], c_features['norm_moment'][0], c_features['norm_moment'][1]]))
                     cur_features = np.array(([sym_characteristics['norm_weight'], sym_characteristics['norm_center'][0], sym_characteristics['norm_center'][1], sym_characteristics['norm_moment'][0], sym_characteristics['norm_moment'][1]]))
                     dist = np.linalg.norm(c_features-cur_features)
                     dist_array[c] = 1-dist
                sorted_dist = sorted(dist_array.items(), key=lambda kv: kv[1], reverse=True)
                ch = sorted_dist[0]
                rec_chars.append(ch)
                self.recognized_string += ch[0]
                dist_array.clear()
        print(rec_chars)
        self.recognized_chars = rec_chars
        return self

def createText(text, font_size=36, font = 'TNR.ttf',image_size=(600,600),filename="text"):
    """
    Функция генерации текста

    :param text: текст
    :type text: str or None

    :param fontSize: размер шрифта
    :type fontSize: int

    :param font: путь до файла шрифта
    :type font: str or None

    :param image_size: размер символа
    :type image_size: tuple

    :param filename: путь до файла сгенерированного текста
    :type filename: str or None
    """
    im = Image.new('L', image_size, color='white')
    d = ImageDraw.Draw(im)
    f = ImageFont.truetype(font, font_size)
    w, h = d.textsize(text[0], font=f)
    row_arr = text.split("\n")
    max=0
    text_h=0
    for i in row_arr:
        w, h = d.textsize(text, font=f)
        text_h+=h
        if max<w:
            max=w
    mw, mh = image_size
    w, h = max, text_h
    x_start,y_start = (((mw - w) // 2), (mh - h) // 2)
    y=y_start
    for row in row_arr:
        x= x_start
        dx,dy = d.textsize(row[0], font=f)
        dx=3
        dy=5
        max=0
        for let in row:
            d.text((x, y), let, font=f, fill=(0))
            (xl,yl)=d.textsize(let, font=f)
            if max<yl:
                max=yl
            x+=dx+xl
        y += dy+max
    im.save("pictures_for_test/"+filename+".bmp")


