from PIL import Image, ImageDraw
import cv2
import numpy as np
import time
from threading import Thread
from multiprocessing import Pool, freeze_support
from functools import partial

def timeit(f):
    def wrap(*args):
        time_start = time.time()
        ret = f(*args)
        time_end = time.time()
        print((time_end-time_start)*1000.0)
        return ret
    return wrap

def eikvil(img, res_path="pictures/res_res.bmp",r_lit=3,r_big=15):
    bin_img=img.copy()
    drw = ImageDraw.Draw(bin_img)
    pix = img.load()
    w = img.size[0]
    h = img.size[1]
    delta = (r_big - r_lit) // 2
    x_begin=y_begin=-delta
    x_end=y_end=r_big-delta
    tresh_list=[]
    for y in range(0,h,r_lit):
        for x in range(0,w,r_lit):
            otsu_img = img.crop((max(0,x_begin),max(0,y_begin),min(w,x_end),min(h,y_end)))

            tresh_list=otsu(otsu_img)
            t = tresh_list[0]
            m0_ideal = tresh_list[1]
            m1_ideal = tresh_list[2]
            epsilon = 15

            if abs(m0_ideal - m1_ideal) >= epsilon:
                for i in range(x, min(x + r_lit,w), 1):
                    for j in range(y, min(y + r_lit,h), 1):
                        br = pix[i, j]
                        if br> t:
                            drw.point((i, j), 255)
                        else:
                            drw.point((i, j), 0)
            else:
                for i in range(x, min(x + r_lit,w), 1):
                    for j in range(y, min(y + r_lit,h), 1):
                        br = pix[i, j]
                        if abs(br - m0_ideal) > abs(br - m1_ideal):
                            drw.point((i, j), 255)
                        else:
                            drw.point((i, j), 0)
            x_begin += r_lit
            x_end += r_lit
        y_begin+=r_lit
        y_end+=r_lit
        x_end = r_big - delta
        x_begin = -delta
    #bin_img.save(res_path, "BMP")
    return bin_img





def otsu(image):
    #hist = (np.histogram(image, bins=256)[0])/image.size[0]*image.size[1]
    hist = cv2.calcHist([np.asarray(image)], [0], None, [256], [0, 256])
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
            res_m1,res_m2 = m1,m2
    return [thresh,res_m1,res_m2]


def get_bin_by_tresh(image):
    result_path="pictures/result_otsu_2.bmp"
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    #hist_br = get_hist_br(image)
    treshold = otsu(image)[0]

    for x in range(width):
        for y in range(height):
            r = pix[x, y][0]
            g = pix[x, y][1]
            b = pix[x, y][2]
            av = (b + r + g) // 3
            if av > treshold:
                draw.point((x, y), (255, 255, 255))
            else:
                draw.point((x, y), (0, 0, 0))
    image.save(result_path, "BMP")

def to_grayscale(image):
    new_img = image.convert('L')
    draw = ImageDraw.Draw(new_img)
    width = new_img.size[0]
    height = new_img.size[1]
    pix = new_img.load()

    for x in range(width):
        for y in range(height):
            r = pix[x, y]
            g = pix[x, y]
            b = pix[x, y]
            av = (b + r + g) // 3
            draw.point((x, y), av)
    new_img.show()
    return new_img
@timeit
def test():
    path_to_img = "pictures/ches_big.bmp"
    #res_path = "pictures/res_.bmp"
    image = Image.open(path_to_img)
    w = image.size[0]
    h = image.size[1]
    img_list = []
    N = 4
    x_begin = 0
    delta = w // N
    r_lit = 3
    r_big = 15
    img_list=[]
    path_list=[]
    for i in range(N):
        image_part = image.copy().crop((x_begin, 0, min(x_begin + delta+r_big,w), h)).convert('L')
        res_path = "pictures/res_" + str(i) + ".bmp"
        img_list.append(image_part)
        path_list.append(res_path)
        # image_part.show()
        x_begin += delta
    pool = Pool(processes=N)
    results=pool.map(eikvil,img_list)
    x_begin = 0
    for i in results:
        image.paste(i,(x_begin,0))
        x_begin+=delta
    image.save("pictures/ches_big_bin_1.bmp", "BMP")
'''
def main():
    path_to_img = "pictures/panda.bmp"
    # res_path = "pictures/res_.bmp"
    image = Image.open(path_to_img)
    gray_img = to_grayscale(image)
    bin_img = eikvil(gray_img)

    bin_img.save("pictures/panda_bin.bmp", "BMP")
'''
if __name__ == '__main__':
  freeze_support()
  test()
#main()
