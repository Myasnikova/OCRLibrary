from core import LabImage
from PIL import Image, ImageDraw
import numpy as np
import math

# Класс контурированных изображений
    # Аргументы конструктора:
    # path: путь до изображения, string

class ContouredImage(LabImage):
    def __init__(self, path=None):
        super().__init__(path)

#Оператор Собеля
    # Аргументы:
    # img: изображение, класс Image
    # t: порог, integer

    def sobelOperator(self,img, t):
        img_arr = np.asarray(img)
        new_img = img.copy()
        w = new_img.size[0]
        h = new_img.size[1]
        gx = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        gy = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
        r = 1
        grad_matrix = np.zeros((h, w))
        draw = ImageDraw.Draw(new_img)
        for y in range(r, h - r):
            for x in range(r, w - r):
                tmp_img = new_img.crop((x - r, y - r, x + r + 1, y + r + 1))
                arr_pix = np.asarray(tmp_img)
                grad_matrix[y][x] = math.sqrt((np.sum(gx * arr_pix)) ** 2 + (np.sum(gy * arr_pix)) ** 2)
        grad_matrix_norm = grad_matrix * 255 / np.max(grad_matrix)
        for y in range(h):
            for x in range(w):
                draw.point((x, y), 255 if grad_matrix_norm[y][x] > t else 0)
        return new_img

#Оператор Прюита
    # Аргументы:
    # image: изображение, класс Image

    def prewittOperator(self,image):
        w = image.size[0]
        h = image.size[1]
        pixels = np.array(image, dtype=np.float)
        horizontal = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
        vertical = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

        newgradientImage = np.zeros((h, w))

        for i in range(1, h - 1):
            for j in range(1, w - 1):
                horizontalGrad = (horizontal[0, 0] * pixels[i - 1, j - 1]) + \
                                 (horizontal[0, 2] * pixels[i - 1, j + 1]) + \
                                 (horizontal[1, 0] * pixels[i, j - 1]) + \
                                 (horizontal[1, 2] * pixels[i, j + 1]) + \
                                 (horizontal[2, 0] * pixels[i + 1, j - 1]) + \
                                 (horizontal[2, 2] * pixels[i + 1, j + 1])

                verticalGrad = (vertical[0, 0] * pixels[i - 1, j - 1]) + \
                               (vertical[0, 1] * pixels[i - 1, j]) + \
                               (vertical[0, 2] * pixels[i - 1, j + 1]) + \
                               (vertical[2, 0] * pixels[i + 1, j - 1]) + \
                               (vertical[2, 1] * pixels[i + 1, j]) + \
                               (vertical[2, 2] * pixels[i + 1, j + 1])

                mag = np.sqrt(pow(horizontalGrad, 2.0) + pow(verticalGrad, 2.0))
                newgradientImage[i, j] = mag
        newgradientImage = newgradientImage / np.max(newgradientImage) * 255
        return Image.fromarray(np.uint8(newgradientImage), 'L')

# Функция выделения контуров на изображении
# Аргументы:
    # image: изображение, класс Image
    # method: метод выделения контуров,1 - оператор Собеля/ 2 - оператор Прюита
    # t: порог, integer

    def toContouredImage(self,image, method,t=None):
        if t is None and method == 1:
            raise ValueError('Undefined threshold: {}'.format(t))
        gray_image = self.to_grayscale(image)
        if method == 1:
            return self.sobelOperator(gray_image,t)
        if method == 2:
            return self.prewittOperator(gray_image)
        raise ValueError('Undefined method: {}'.format(method))
# Тест
def test():
    img = ContouredImage("pictures/cat.bmp")
    img.show()
    img.toContouredImage(img.orig,2).show()
    #img.toContouredImage(img.orig,1,25).show()
test()