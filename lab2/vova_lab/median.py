from PIL import Image, ImageOps
import numpy as np

def median_filter(image):
    aprt = np.array(([0,1,0],[1,1,1],[0,1,0]))
    width = image.size[0]  # ���������� ������
    height = image.size[1]  # ���������� ������
    pixels = np.array(image, dtype=np.float) #��������� � np.array 
    pixels = pixels /255
    new_pixels = np.ones((height, width), np.float)
    #for x in range(1,width-1):
    #    for y in range(1,height-1):
    #       tmp = pixels[y-1:y+2,x-1:x+2]
    #       sum_res =  np.sum(tmp*aprt)
    #       if(sum_res >=3):
    #            new_pixels[y,x] = 1 
    #       else:
    #            new_pixels[y,x] = 0
    right = np.roll(pixels, -1, axis=1)
    right[:, width-1] = 1
    left = np.roll(pixels, +1, axis=1)
    right[:, 0] = 1
    bottom = np.roll(pixels, -1, axis=0)
    bottom[height-1, :] = 1
    upper = np.roll(pixels, +1, axis=0)
    upper[0, :] = 1

    sum = pixels + right + left + bottom + upper
    new_pixels = (sum >=3) * 255
    return Image.fromarray(np.uint8(new_pixels) , 'L') #��������� � ����� �����������, ����� ���� ������ ��������� ��� � ������� � int8, � �� fromarray ��������
    
    

    return thresholds

def main():
    print('Start...')
    with open("images.txt") as file:
        for image_name in file:
            image = Image.open("images/" + image_name[:-1])
            new_image = median_filter(image)
            new_image.save('results/' + image_name[:-5]+'_median.bmp', "BMP")
            dif_image = np.abs(np.array(image, dtype=np.uint8)-new_image)
            dif_image = (dif_image == 0) * 255
            dif_image = Image.fromarray(np.uint8(dif_image) , 'L')
            dif_image.save('results/' + image_name[:-5]+'_dif.bmp', "BMP")
            print('Done!')


if __name__ == "__main__":
    main()