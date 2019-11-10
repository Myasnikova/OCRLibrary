from core import *

class BinaryImage(LabImage):
    def __init__(self, Image=None):
        self.orig =  Image.orig
        self.grayscale_matrix = Image.grayscale_matrix
        self.size = Image.orig.size
        self.height, self.width = self.size
        self.rgb_matrix = np.array(self.orig)

    def kir_binarization(self, rsize=3, Rsize=15, eps=15):
        def otsu_global(matrix: np.ndarray):
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

        def binarization_processor(matrix_ind: tuple, epsilon=eps):
            matrix_k_ind, matrix_K_ind = matrix_ind
            matrix_k = self.grayscale_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1],
                                             matrix_k_ind[1][0]: matrix_k_ind[1][1]]
            matrix_K = self.grayscale_matrix[matrix_K_ind[0][0]: matrix_K_ind[0][1],
                                             matrix_K_ind[1][0]: matrix_K_ind[1][1]]
            T, M0, M1 = otsu_global(matrix_K)

            if abs(M1 - M0) >= epsilon:
                self.bin_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]] = \
                    np.where(matrix_k < T, 0, 255)
            else:
                k_mean = matrix_k.mean()
                new_T = (M0 + M1) / 2
                self.bin_matrix[matrix_k_ind[0][0]: matrix_k_ind[0][1], matrix_k_ind[1][0]: matrix_k_ind[1][1]]\
                    .fill(0 if k_mean <= new_T else 255)

        if (not (rsize % 2) and not (Rsize % 2)) or ((rsize % 2) and (Rsize % 2)):
            if self.grayscale_matrix is None:
                self.to_grayscale()
            self.bin_matrix = self.grayscale_matrix.astype(np.uint8)

            for x in split_submatrix(self.bin_matrix, (rsize, rsize), (Rsize, Rsize)):
                binarization_processor(x)

            self.result = Image.fromarray(self.bin_matrix, 'L')

        else:
            raise WrongWindowSize("Rsize={} and rsize={} must be even or odd both together".format(Rsize, rsize))


        #принимает np.array, возвращает интегральный np.array. Можно сделать средствами нумпая в одну стоку, но пусть будет ручной
    def calc_integ(self, img):
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
        if self.grayscale_matrix is None:
                self.to_grayscale()
        rows, cols = self.grayscale_matrix.shape
        pix = self.grayscale_matrix #выгружаем в np.array

        integr = self.calc_integ(pix) #считаем интегральное изображение
        sqr_integr = self.calc_integ(np.square(pix)) #то же самое, но уже квадраты

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

        #считаем среднее значение
        sums = np.zeros((rows, cols),  np.float)
        for y in range(0,rows):
            for x in range(0,cols):
                sums[y,x] = integr[y2[y,x], x2[y,x]] - integr[y2[y,x], x1[y,x]] - integr[y1[y,x], x2[y,x]] + integr[y1[y,x] , x1[y,x]] # 0_O

        means = sums / s

        #считаем отклонение
        dev_sums = np.zeros((rows, cols),  np.float)
        for y in range(0,rows):
            for x in range(0,cols):
                dev_sums[y,x] = sqr_integr[y2[y,x], x2[y,x]] - sqr_integr[y2[y,x], x1[y,x]] - sqr_integr[y1[y,x], x2[y,x]] + sqr_integr[y1[y,x] , x1[y,x]] # 0_O [2]
        devs = np.sqrt(dev_sums / s - np.square(means))

        # минимальные и максимальные (????)
        R = np.max(devs)
        M = np.min(self.grayscale_matrix)

        # считаем порог
        thresholds = ((1.0 - k) * means + k * M + k * devs / R * (means - M))
        img = ((self.grayscale_matrix >= thresholds) * 255).astype(np.uint8) #255, если больше, 0 если меньше
        self.result = Image.fromarray(np.uint8(img) , 'L')
