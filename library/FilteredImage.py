from tqdm import tqdm

from library.core import *
from library.exceptions import WrongRank, WrongWindowSize

from library.BinaryImage import BinaryImage


class FilteredImage(LabImage):
    """
    Класс осуществляющий фильтрацию переданного на вход изображения следующими методами:
        - медианная фильтрация
        - ранговая фильтрация
        - взвешанная ранговая фильтрация
    """
    def __init__(self, path=None, image=None):
        """
        Инициализация объекта класса FilteredImage

        :param path: путь до изображения
        :type path: str or None
        :param image: экземпляр класса LabImage
        :type image: LabImage or None
        """
        super(FilteredImage, self).__init__(path=path, image=image)

        self.filtered_matrix = None
        if getattr(self, 'bin_matrix', None) is None:
            # TODO надо бы выбрать способ бинаризации по умолчанию
            self.bin_matrix = BinaryImage(path=path, image=image).eikvil_binarization().bin_matrix
            # self.bin_matrix = self.grayscale_matrix

    def median_filter(self, wsize=3):
        """
        Медианная фильтрация изображения

        :param wsize: размер окна фильтрации
        :type wsize: int

        :return: LabImage -- объект изображения

        :raises: WrongWindowSize
        """
        if (not wsize % 2) or (wsize < 0):
            raise WrongWindowSize("wsize must be odd, positive and integer")

        bias = wsize // 2
        pixels = self.bin_matrix / 255

        right, left, bottom, upper = [np.zeros(self.bin_matrix.shape)] * 4
        for b in range(1, bias + 1):
            right_ = np.roll(pixels, -bias, axis=1)
            right_[:, -bias:] = pixels[:, -bias:][:, ::-1]
            right += right_
            left_ = np.roll(pixels, bias, axis=1)
            left_[:, :bias] = pixels[:, :bias][:, ::-1]
            left += left_
            bottom_ = np.roll(pixels, -bias, axis=0)
            bottom_[-bias:, :] = pixels[-bias:, :][::-1]
            bottom += bottom_
            upper_ = np.roll(pixels, bias, axis=0)
            upper_[:bias, :] = pixels[:bias, :][::-1]
            upper += upper_

        filtered_matr = pixels + right + left + bottom + upper
        filtered_matr = np.where(filtered_matr < wsize, 0, 255)

        self.filtered_matrix = filtered_matr
        self.result = Image.fromarray(np.uint8(filtered_matr), 'L')

        return self

    def weighted_rank_filter(self, rank: int, wsize=3):
        """
        Взвешенная ранговая фильтрация изображения

        :param rank: ранг фильтра
        :type rank: int
        :param wsize: размер окна фильтрации (поддерживаются только окна размера 3 или 5)
        :type wsize: int

        :return: LabImage -- объект изображения

        :raises: WrongWindowSize
        """
        def prepare_matrix(matrix: np.ndarray):
            bias = wsize // 2
            new_matrix = np.vstack((matrix[1:(bias + 1)][::-1], matrix, matrix[-(bias + 1):-1][::-1]))
            new_matrix = np.hstack((new_matrix[:, 1:(bias + 1)][:, ::-1], new_matrix, new_matrix[:, -(bias + 1):-1][:, ::-1]))

            return new_matrix

        def custom_multiply(orig_matr: np.ndarray):
            res = []
            for k in range(orig_matr.size):
                res = res + [orig_matr.flatten()[k]] * mask.flatten()[k]

            return res

        w_3 = np.array([[1, 2, 1],
                        [2, 4, 2],
                        [1, 2, 1]])
        w_5 = np.array([[0, 0, 1, 0, 0],
                        [0, 2, 4, 2, 0],
                        [1, 4, 8, 4, 1],
                        [0, 2, 4, 2, 0],
                        [0, 0, 1, 0, 0]])

        if wsize not in (3, 5):
            raise WrongWindowSize("wsize must be only 3 or 5")
        elif wsize == 3:
            mask = w_3
        else:
            mask = w_5

        prepared_matrix = prepare_matrix(self.bin_matrix) // 255
        filtered_matrix = np.ndarray(self.bin_matrix.shape)
        for (x, y), _ in tqdm(np.ndenumerate(self.bin_matrix),
                              total=self.bin_matrix.size,
                              desc='rank filter: '):
            filtered_matrix[x, y] = sorted(custom_multiply(prepared_matrix[x: x + wsize, y: y + wsize]))[rank]

        self.filtered_matrix = np.uint8(filtered_matrix) * 255
        self.result = Image.fromarray(self.filtered_matrix, 'L')

        return self

    def rank_filter(self, rank: int, wsize=3):
        """
        Невзвешенная ранговая фильтрация изображения

        :param rank: ранг фильтра
        :type rank: int
        :param wsize: размер окна фильтрации (поддерживаются только окна размера 3 или 5)
        :type wsize: int

        :return: LabImage -- объект изображения

        :raises: WrongWindowSize, WrongRank
        """
        def prepare_matrix(matrix: np.ndarray):
            bias = wsize // 2
            new_matrix = np.vstack((matrix[1:(bias + 1)][::-1], matrix, matrix[-(bias + 1):-1][::-1]))
            new_matrix = np.hstack((new_matrix[:, 1:(bias + 1)][:, ::-1], new_matrix, new_matrix[:, -(bias + 1):-1][:, ::-1]))

            return new_matrix

        if (not wsize % 2) or (wsize < 0):
            raise WrongWindowSize("wsize must be odd, positive and integer")

        if rank >= wsize**2 or rank < 0:
            raise WrongRank("rank must be positive and less than wsize*wsize")

        prepared_matrix = prepare_matrix(self.bin_matrix)
        filtered_matrix = np.ndarray(self.bin_matrix.shape)
        for (x, y), _ in tqdm(np.ndenumerate(self.bin_matrix),
                              total=self.bin_matrix.size,
                              desc='rank filter: '):
            filtered_matrix[x, y] = sorted(prepared_matrix[x: x+wsize, y: y+wsize].flatten())[rank]

        self.filtered_matrix = np.uint8(filtered_matrix)
        self.result = Image.fromarray(self.filtered_matrix, 'L')

        return self


im = LabImage("../sample_2.bmp")
im = FilteredImage(image=im)
im.median_filter(wsize=7)
# im.weighted_rank_filter(3, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], 10)
im.show()
