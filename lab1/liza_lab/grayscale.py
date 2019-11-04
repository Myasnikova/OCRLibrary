from PIL import Image, ImageDraw

def to_grayscale(path_to_img, result_path):
    image = Image.open(path_to_img)
    draw = ImageDraw.Draw(image)
    width = image.size[0]
    height = image.size[1]
    pix = image.load()

    for x in range(width):
        for y in range(height):
            r = pix[x, y][0]
            g = pix[x, y][1]
            b = pix[x, y][2]
            av = (b + r + g) // 3
            draw.point((x, y), (av, av, av))
    image.save(result_path, "BMP")

result_path="pictures/binarization/ches_big_gray.bmp"
path_to_img="pictures/binarization/ches_big.bmp"

to_grayscale(path_to_img, result_path)



