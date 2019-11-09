from PIL import Image, ImageOps
import numpy as np
import time

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

def prewitt(image):
    w = image.size[0]  # ���������� ������
    h = image.size[1]  # ���������� ������
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
    return Image.fromarray(np.uint8(newgradientImage) , 'L') #��������� � ����� �����������, ����� ���� ������ ��������� ��� � ������� � int8, � �� fromarray ��������
    
    

    return thresholds

def main():
    print('Start...')
    with open("images.txt") as file:
        for image_name in file:
            image = Image.open("images/" + image_name[:-1])
            start_time = time.time()
            new_image = ImgToGrayscale(image)
            print("--- %s seconds ---" % (time.time() - start_time))
            new_image.save('results/' + image_name[:-5]+'_grayscale.bmp', "BMP")
            start_time = time.time()
            prewitt_image = prewitt(new_image)
            print("--- %s seconds ---" % (time.time() - start_time))
            prewitt_image.save('results/' + image_name[:-5]+'_prewitt.bmp', "BMP")
            prewitt_array = np.array(prewitt_image, dtype=np.uint8)
            for i in range(0, 255, 15):
                img = ((prewitt_array >= i) * 255).astype(np.uint8)      
                Image.fromarray(np.uint8(img) , 'L').save('results/' + image_name[:-5]+'_thresholds_%.2f.bmp' %i, "BMP")
            print('Done!')

if __name__ == "__main__":
    main()