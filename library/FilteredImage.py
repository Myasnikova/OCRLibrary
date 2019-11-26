from PIL import Image

import numpy as np
from tqdm import tqdm

from core import LabImage, timeit
from exceptions import WrongWindowSize, WrongRank


class FilteredImage(LabImage):
    def __init__(self, Image=None):
        super(FilteredImage, self).__init__(image=Image)

        self.filtered_matrix = None
        self.calc_grayscale_matrix()

    def median_filter(self):
        aprt = np.array(([0, 1, 0], [1, 1, 1], [0, 1, 0]))
        # TODO я в душе не ебу что написано в комментах, поэтому реализую по наитию
        pixels = np.array(self.orig, dtype=np.float)
        pixels = pixels / 255

        right = np.roll(pixels, -1, axis=1)
        right[:, self.height - 1] = 1
        left = np.roll(pixels, +1, axis=1)
        right[:, 0] = 1
        bottom = np.roll(pixels, -1, axis=0)
        bottom[self.width - 1, :] = 1
        upper = np.roll(pixels, +1, axis=0)
        upper[0, :] = 1

        sum = pixels + right + left + bottom + upper
        new_pixels = (sum >= 3) * 255

        self.filtered_matrix = new_pixels
        self.result = Image.fromarray(np.uint8(new_pixels), 'L')

        return self

    def weighted_rank_filter(self, R, f, k):
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

        if self.grayscale_matrix is None:
            self.to_grayscale()

        prepared_matrix = prepare_matrix(self.grayscale_matrix)
        filtered_matrix = np.ndarray(self.grayscale_matrix.shape)
        for (x, y), _ in tqdm(np.ndenumerate(self.grayscale_matrix), total=self.grayscale_matrix.size):
            filtered_matrix[x, y] = sorted(prepared_matrix[x: x+wsize, y: y+wsize].flatten())[rank]

        self.filtered_matrix = np.uint8(filtered_matrix)
        self.result = Image.fromarray(self.filtered_matrix, 'L')


im = LabImage("../sample_1.bmp")
im = FilteredImage(im)
im.rank_filter(6, wsize=3)
# im.weighted_rank_filter(3, [[1, 2, 1], [2, 4, 2], [1, 2, 1]], 10)
im.show()
