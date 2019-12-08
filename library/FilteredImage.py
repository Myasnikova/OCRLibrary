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
            self.bin_matrix = BinaryImage(path=path, image=image).eikvil_binarization()

    def median_filter(self, wsize=3):
        bias = wsize // 2
        pixels = self.grayscale_matrix / 255

        right = np.roll(pixels, -bias, axis=1)
        right[:, -bias:] = pixels[:, -bias:][:, ::-1]
        left = np.roll(pixels, bias, axis=1)
        left[:, :bias] = pixels[:, :bias][:, ::-1]
        bottom = np.roll(pixels, -bias, axis=0)
        bottom[-bias:, :] = pixels[-bias:, :][::-1]
        upper = np.roll(pixels, bias, axis=0)
        upper[:bias, :] = pixels[:bias, :][::-1]

        filtered_matr = pixels + right + left + bottom + upper
        filtered_matr = np.where(filtered_matr < wsize, 0, 255)

        self.filtered_matrix = filtered_matr
        self.result = Image.fromarray(np.uint8(filtered_matr), 'L')

        return self

    def weighted_rank_filter(self, wsize=3):
        r = R // 2
        # draw = ImageDraw.Draw(new_img)
        for y in range(r, self.height - r):
            for x in range(r, self.width - r):
                arr_pix = self.rgb_matrix[x - r: x + r + 1, y - r: y + r + 1]
                # black_pix_amount = np.sum(arr_pix)
                black_pix_amount = np.sum(f * arr_pix)
                if black_pix_amount >= k:
                    draw.point((x, y), True)
                else:
                    draw.point((x, y), False)
        return new_img

    def rank_filter(self, rank: int, wsize=3):
        def prepare_matrix(matrix: np.ndarray):
            bias = wsize // 2
            new_matrix = np.vstack((matrix[1:(bias + 1)][::-1], matrix, matrix[-(bias + 1):-1][::-1]))
            new_matrix = np.hstack((new_matrix[:, 1:(bias + 1)][:, ::-1], new_matrix, new_matrix[:, -(bias + 1):-1][:, ::-1]))

            return new_matrix

        if not wsize % 2:
            raise WrongWindowSize("wsize must be odd, positive and integer")

        if rank >= wsize**2 or rank < 0:
            raise WrongRank("rank must be positive and less than wsize*wsize")

        prepared_matrix = prepare_matrix(self.grayscale_matrix)
        filtered_matrix = np.ndarray(self.grayscale_matrix.shape)
        for (x, y), _ in tqdm(np.ndenumerate(self.grayscale_matrix),
                              total=self.grayscale_matrix.size,
                              desc='rank filter: '):
            filtered_matrix[x, y] = sorted(prepared_matrix[x: x+wsize, y: y+wsize].flatten())[rank]

        self.filtered_matrix = np.uint8(filtered_matrix)
        self.result = Image.fromarray(self.filtered_matrix, 'L')


im = LabImage("../sample_2.bmp")
im = FilteredImage(image=im)
im.median_filter(wsize=3)
# im.weighted_rank_filter(3, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], 10)
im.show()
