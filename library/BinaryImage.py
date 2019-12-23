from tqdm import tqdm
from PIL import ImageDraw
from math import ceil

from core import *
from exceptions import WrongWindowSize


class BinaryImage(LabImage):
    """
    Класс осуществляющий бинаризацию переданного на вход изображения следующими методами:
        - :meth:`~BinaryImage.eikvil_binarisation` -- метод Эйквила
        - :meth:`~BinaryImage.cristian_binarisation` -- метод Кристиана
    """
    def __init__(self, path=None, image=None, pilImage=None):
        super(BinaryImage, self).__init__(path=path, image=image, pilImage=pilImage)

    def eikvil_binarisation(self, rsize=3, Rsize=15, eps=15):
        """
        Бинаризация изображения методом Эйквила

        :param rsize: размер малого окна
        :type rsize: int
        :param Rsize: размер большего окна
        :type Rsize: int
        :param eps: величина отклонения для математических ожиданий чёрного и белого, в пределах которого можно считать \
        , что они отличются несущественно
        :type eps: int

        :return: :class:`~core.LabImage` -- объект изображения

        :raises: WrongWindowSize
        """
        def otsu_global(matrix: np.ndarray) -> tuple:
            """
            Глобальная бинаризация методом Отсу

            :param matrix: матрица изображения
            :type numpy.ndarray

            :return: tuple -- image threshold, black mean, white mean
            """
            n_curr = 0
            T_res = 0
            M0_res = 0
            M1_res = 0

            p_tmp = np.unique(matrix, return_counts=True)
            p = p_tmp[1] / matrix.size

            for t in range(matrix.min(), matrix.max()):
                w0 = p[p_tmp[0] <= t].sum() if p[p_tmp[0] <= t].sum() > 0.00001 else 0.00001
                w1 = 1 - w0 if 1 - w0 > 0.00001 else 0.00001
                M0 = (p_tmp[0][p_tmp[0] <= t] * p[p_tmp[0] <= t]).sum() / w0
                M1 = (p_tmp[0][p_tmp[0] > t] * p[p_tmp[0] > t]).sum() / w1
                D0 = (p[p_tmp[0] <= t] * np.square(p_tmp[0][p_tmp[0] <= t] - M0)).sum()
                D1 = (p[p_tmp[0] > t] * np.square(p_tmp[0][p_tmp[0] > t] - M1)).sum()

                n = (w0 * w1 * (M0 - M1)**2) // (w0 * D0 + w1 * D1)
                if n >= n_curr:
                    n_curr = n
                    T_res = t
                    M0_res = M0
                    M1_res = M1

            return T_res, M0_res, M1_res

        # @timeit
        def split_submatrix(matrix: np.ndarray, submat1_shape: tuple, submat2_shape: tuple):
            """
            Генерирует кортеж из начальных и конечных координат для малого и большего окна

            :param matrix: матрица изображения
            :type numpy.ndarray
            :param submat1_shape: размер малого окна
            :type submat1_shape: tuple
            :param submat2_shape: размер большего окна
            :type submat2_shape: tuple
            """
            p, q = submat1_shape
            P, Q = submat2_shape
            m, n = matrix.shape

            bias_p = (P - p) // 2
            bias_q = (Q - q) // 2
            for x in range(0, m, p):
                for y in range(0, n, q):
                    yield (
                              (
                                  (x, (x + p) if (x + p - m) < 0 else m),
                                  (y, (y + q) if (y + q - n) < 0 else n)
                              ),
                              (
                                  ((x - bias_p) if (x - bias_p) > 0 else 0, (x + P - bias_p) if (x + P - bias_p) < m else m),
                                  ((y - bias_q) if (y - bias_q) > 0 else 0, (y + Q - bias_q) if (y + Q - bias_q) < n else n)
                              )
                    )

        def binarization_processor(matrix_ind: tuple):
            """
            Бинаризация малого окна по Эйквилу

            :param matrix_ind: кортеж из начальных и конечных координат для малого и большего окна
            :type matrix_ind: tuple
            """
            matrix_k_ind, matrix_K_ind = matrix_ind
            matrix_k = self.grayscale_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1],
                                             matrix_k_ind[1][0]: matrix_k_ind[1][1]]
            matrix_K = self.grayscale_matrix[matrix_K_ind[0][0]: matrix_K_ind[0][1],
                                             matrix_K_ind[1][0]: matrix_K_ind[1][1]]
            T, M0, M1 = otsu_global(matrix_K)

            if abs(M1 - M0) >= eps:
                self.bin_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]] = \
                    np.where(matrix_k < T, 0, 255)
            else:
                k_mean = matrix_k.mean()
                new_T = (M0 + M1) / 2
                self.bin_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]]\
                    .fill(0 if k_mean <= new_T else 255)

        if (not (rsize % 2) and not (Rsize % 2)) or ((rsize % 2) and (Rsize % 2)):
            self.bin_matrix = self.grayscale_matrix.astype(np.uint8)
            for x in tqdm(split_submatrix(self.bin_matrix, (rsize, rsize), (Rsize, Rsize)),
                          total=(ceil(self.bin_matrix.shape[0] / rsize) * ceil(self.bin_matrix.shape[1] / rsize)),
                          desc='eikvil binarization: '):
                binarization_processor(x)
            self.result = Image.fromarray(self.bin_matrix, 'L')

            return self

        else:
            raise WrongWindowSize("Rsize={} and rsize={} must be even or odd both together".format(Rsize, rsize))

    def calc_integ(self, img: np.ndarray):
         """
         Расчет интегрального изображения из исходного

         :param img: матрица изображения
         :type img: numpy.ndarray
         """
         h, w = img.shape
         integr = np.zeros((h, w),  np.float)
         #сначала посчитаем для первого столбца и первой строки, чтоб не выходить за границы в рекурсивной формуле
         integr[0, :] = np.cumsum(img[0, :])
         integr[:, 0] = np.cumsum(img[:, 0])
         #рекурсивная формула, 20 слайд
         for y in range(1,h):
             for x in range(1,w):
                 integr[y,x] = img[y,x] - integr[y-1,x-1] + integr[y-1,x] + integr[y,x-1]
         return integr
         

    def cristian_binarisation(self, w_size=15, k=0.5):
        """
        Бинаризация изображения методом Кристиана

        :param w_size: размер окна
        :type w_size: int
        :param k: коэффициент, отвечающий за чувствительность бинаризатора
        :type k: float

        :return: :class:`~core.LabImage` -- объект изображения
        """
        if self.grayscale_matrix is None:
                self.calc_grayscale_matrix()
        rows, cols = self.grayscale_matrix.shape
        pix = self.grayscale_matrix 

        integr = self.calc_integ(pix) 
        npix = pix.astype(int)
        n = np.square(npix)
        sqr_integr = self.calc_integ(np.square(npix)) 

        half_w = w_size // 2
 
        #сейчас будут магические вещи: сделаем массив, в котором для каждого пикселя будут индексы границы окна
        #я сам немного удивлен, что это работает, но это работает!        
        x, y = np.meshgrid(np.arange(0, cols), np.arange(0, rows))
    
        x1 = (x - half_w).clip(0, cols-1) #левая граница по x
        x2 = (x + half_w).clip(0, cols-1) #правая граница по х
        y1 = (y - half_w).clip(0, rows-1) #левая граница по y
        y2 = (y + half_w).clip(0, rows-1) #правая граница по y

        # площадь окна для каждого пикселя. Будет другая у граничных, поэтому это тут
        s = (y2 - y1 + 1) * (x2 - x1 + 1)

        
        sums = np.zeros((rows, cols),  np.float)
        for y in range(0,rows):
            for x in range(0,cols):
                sums[y,x] = integr[y2[y,x], x2[y,x]] - integr[y2[y,x], x1[y,x]] - integr[y1[y,x], x2[y,x]] + integr[y1[y,x] , x1[y,x]] # 0_O

        means = sums / s

       
        dev_sums = np.zeros((rows, cols),  np.float)
        for y in range(0,rows):
            for x in range(0,cols):
                dev_sums[y,x] = sqr_integr[y2[y,x], x2[y,x]] - sqr_integr[y2[y,x], x1[y,x]] - sqr_integr[y1[y,x], x2[y,x]] + sqr_integr[y1[y,x] , x1[y,x]] # 0_O [2]
        devs = np.sqrt(dev_sums / s - np.square(means))

        
        R = np.max(devs)
        M = np.min(self.grayscale_matrix)

        
        thresholds = ((1.0 - k) * means + k * M + k * devs / R * (means - M))
        img = ((self.grayscale_matrix >= thresholds) * 255).astype(np.uint8) #255, если больше, 0 если меньше
        self.result = Image.fromarray(np.uint8(img) , 'L')
        self.bin_matrix = img
        return self


def otsu(image):
    """
    Метод Отсу

    :param image: изображение
    :type PIL.Image

    :return: int -- порог бинаризации
    """
    hist = (np.histogram(image, bins=256)[0]) / image.size[0] * image.size[1]
    #hist = cv2.calcHist([np.asarray(image)], [0], None, [256], [0, 256])
    bins = np.arange(256)
    hist_norm = hist.ravel() / hist.max()
    Q = np.cumsum(hist_norm)
    fn_min = np.inf
    thresh = -1
    for i in list(range(1, 256)):
        p1, p2 = np.hsplit(hist_norm, [i])  # probabilities
        q1, q2 = Q[i], Q[255] - Q[i]  # cum sum of classes
        if q1 == 0:
            q1 = 0.00000001
        if q2 == 0:
            q2 = 0.00000001
        b1, b2 = np.hsplit(bins, [i])  # weights
        # finding means and variances
        m1, m2 = np.sum(p1 * b1) / q1, np.sum(p2 * b2) / q2
        v1, v2 = np.sum(((b1 - m1) ** 2) * p1) / q1, np.sum(((b2 - m2) ** 2) * p2) / q2
        # calculates the minimization function
        fn = v1 * q1 + v2 * q2
        if fn < fn_min:
            fn_min = fn
            thresh = i
            res_m1, res_m2 = m1, m2
    return thresh


def get_bin_by_tresh(image):
    """
    Глобальная бинаризация методом Отсу

    :param image: изображение
    :type PIL.Image

    :return: :class:`~PIL.Image` -- бинаризованное изображение
    """
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    treshold = otsu(image)
    new_img = Image.new('L',(width,height))
    draw = ImageDraw.Draw(new_img)
    for x in range(width):
        for y in range(height):
            if pix[x, y] > treshold:
                draw.point((x, y), 255)
            else:
                draw.point((x, y), 0)
    return new_img

