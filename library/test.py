import coverage as cov
from image import OCRImage

def test_binarization(img):
    img.get_binary_image(1)
    img.show_result()
    img.save("test_result/binrization_eikvil.bmp")
    img.get_binary_image(2)
    img.show_result()
    img.save("test_result/binrization_cristian.bmp")

def test_filteration(img):
    img.get_filtered_image(1)
    img.show_result()
    img.save("test_result/filteration_median.bmp")
    img.get_filtered_image(2, 5)
    img.show_result()
    img.save("test_result/filteration_rank_weighted.bmp")
    img.get_filtered_image(3, 5)
    img.show_result()
    img.save("test_result/filteration_rank.bmp")

def test_countouring(img):
    img. get_contoured_image(1, 25)
    img.show_result()
    img.save("test_result/countoured_sobel.bmp")
    img. get_contoured_image(2)
    img.show_result()
    img.save("test_result/countoured_prewit.bmp")


def test_recognizing(img, text):
    assert img.get_text_recognized_image(text) == text.replace(' ', '')


def test_profiled(img, text):
    img.get_text_profiled_image(text)
    img.show_result()
    img.save("test_result/profiled_text.bmp")


def cov_test(text,path):
    coverage = cov.Coverage()
    coverage.erase()
    coverage.start()
    img = OCRImage(path)
    dir='coverage_reports'
    test = [test_filteration, test_countouring, test_binarization]
    for f in test:
        f(img)
    test_profiled(img, text)
    test_recognizing(img, text)
    coverage.html_report(directory=dir)
    coverage.stop()
    coverage.save()


path = "pictures_for_test/cat.bmp"
text = "Гори в адУ"

cov_test(text, path)
#img = OCRImage(path)
#test_filteration(img)


