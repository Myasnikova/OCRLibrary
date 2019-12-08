from core import LabImage
from PIL import Image, ImageDraw
import numpy as np
import math


# Класс контурированных изображений

class ContouredImage(LabImage):
    def __init__(self,  image=None):
        super().__init__()
        if image is not None:
            self.orig = image.orig
            self.gray_image = image.gray_image
            self.size = image.orig.size
            self.height, self.width = self.size
            self.rgb_matrix = np.array(self.orig)
            self.path = image.path

    # Оператор Собеля
    # Аргументы:
    # img: изображение, класс Image
    # t: порог, integer

    def sobel_operator(self, t):
        img = self.gray_image
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
        self.result = new_img
        return new_img

    # Оператор Прюита
    # Аргументы:
    # image: изображение, класс Image

    def prewitt_operator(self):
        image = self.gray_image
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
        self.result = Image.fromarray(np.uint8(newgradientImage), 'L')
        return self.result


# Тест
def test():
    lab_img = LabImage("pictures_for_test/cat.bmp")
    img = ContouredImage(lab_img)
    img.show()
    #img.prewitt_operator().show()
    img.sobel_operator(25).show()

#test()
