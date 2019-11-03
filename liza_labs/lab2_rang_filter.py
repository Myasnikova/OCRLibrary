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
    return thresh

def get_bin_by_tresh(image):
    width = image.size[0]
    height = image.size[1]
    pix = image.load()
    #hist_br = get_hist_br(image)
    treshold = otsu(image)
    new_img = Image.new('1',(width,height))
    draw = ImageDraw.Draw(new_img)
    for x in range(width):
        for y in range(height):
            if pix[x, y] > treshold:
                draw.point((x, y), True)
            else:
                draw.point((x, y), False)
    return new_img
@timeit
def rang_filtr(img, R,f,k):
    #img = image.convert('1')
    new_img = img.copy()
    w = new_img.size[0]
    h = new_img.size[1]
    r=R//2
    draw = ImageDraw.Draw(new_img)
    for y in range(r,h-r):
        for x in range(r,w-r):
            tmp_img = new_img.crop((x-r,y-r,x+r+1,y+r+1))
            arr_pix = np.asarray(tmp_img)
            #black_pix_amount = np.sum(arr_pix)
            black_pix_amount=np.sum(f*arr_pix)
            if black_pix_amount>=k:
                draw.point((x, y), True)
            else:
                draw.point((x, y), False)
    return new_img
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
    return new_img

@timeit
def main():
    path_to_img = "pictures/filtering/earth.bmp"

    image = Image.open(path_to_img)
    gray_img = to_grayscale(image)
    bin_img = get_bin_by_tresh(gray_img)
    bin_path = path_to_img.split('.')[0] + "_bin.bmp"
    bin_img.save(bin_path,"bmp")
    bin_img.show()
    img_arr = np.asarray(bin_img)
    #print(img_arr)
    r=3
    #f=[[1,0,1],[1,0,1],[1,2,1]]
    f = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    k=10
    res_path = path_to_img.split('.')[0] + "_res_" + str(k) +str(np.sum(f))+'.bmp'
    new_image = rang_filtr(bin_img,r,f,k)
    new_image.save(res_path, "BMP")

main()
