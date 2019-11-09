from PIL import Image, ImageOps
import numpy as np
image_name = 'text1'
type = '.png'

#переводит изображение в оттенки серого. На вход - экземпляр Image, на выходе другой(!) экземпляр Image, уже серый
def ImgToGrayscale(image):
    image = image.convert('RGB')
    width = image.size[0]  # Определяем ширину
    height = image.size[1]  # Определяем высоту
    pixels = image.load() #грузим изображение в матрицу яркостей по разным каналам
    new_pixels = np.zeros((height, width), np.float)
    for x in range(width):
        for y in range(height):
           tmp = pixels[x, y]
           r, g, b =  pixels[x, y][:3] #отдельно берем яркости по разным цветовым каналам
           sr = (r + g + b) / 3 #среднее значение
           new_pixels[y,x] = sr #в numpy все наоборот, поэтому тут сначала y, потом x
    return Image.fromarray(np.uint8(new_pixels) , 'L') #сохраняем в новое изображение, перед этим руками переводим все в массиве в int8, а то fromarray руагется

#добавление рамки к изображению заданного цвета, уже не используется
def add_border(image, top, bottom, left, right, value):
    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image
    h, w = image.shape
    new_shape = list(image.shape)
    new_shape[0] += top + bottom
    new_shape[1] += left + right
    new_image = np.empty(new_shape, dtype=image.dtype)
    new_image.fill(value)
    new_image[top:top+h, left:left+w] = image
    return new_image


#принимает np.array, возвращает интегральный np.array. Можно сделать средствами нумпая в одну стоку, но пусть будет ручной
def calc_integ(img):
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

def calc_cristian_thresholds(img, w_size=15, k=0.5):
    cols, rows = img.size
    pix = np.array(img, dtype=np.float) #выгружаем в np.array

    integr = calc_integ(pix) #считаем интегральное изображение
    sqr_integr = calc_integ(np.square(pix)) #то же самое, но уже квадраты

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
    M = np.min(img)

    # считаем порог
    thresholds = ((1.0 - k) * means + k * M + k * devs / R * (means - M))
    return thresholds


def main():
    print('Start...')
    with open("images.txt") as file:
        for image_name in file:
            image = Image.open("images/" + image_name[:-1])
            new_image = ImgToGrayscale(image)
            new_image.save('results/' + image_name[:-5]+'_grayscale.bmp', "BMP")
            print('Grayscale image done!')
            for k in np.arange(-0.2, 0.8, 0.2):
                thresholds = calc_cristian_thresholds(new_image, 15, k)
                img = ((new_image >= thresholds) * 255).astype(np.uint8) #255, если больше, 0 если меньше
                Image.fromarray(np.uint8(thresholds) , 'L').save('results/' + image_name[:-5]+'_thresholds_%.2f.bmp' %k, "BMP")
            print('Done!')

if __name__ == "__main__":
    main()