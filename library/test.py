import coverage as cov
from image import OCRImage

def test_binarization(img):
    img.get_binary_image(1)
    #img.show_result()
    img.save("test_result/binrization_eikvil.bmp")
    img.get_binary_image(2)
    #img.show_result()
    img.save("test_result/binrization_cristian.bmp")

def test_filteration(img):
    img.get_filtered_image(1)
    #img.show_result()
    img.save("test_result/filteration_median.bmp")
    img.get_filtered_image(2, 5)
    #img.show_result()
    img.save("test_result/filteration_rank_weighted.bmp")
    img.get_filtered_image(3, 5)
    #img.show_result()
    img.save("test_result/filteration_rank.bmp")

def test_countouring(img):
    img. get_contoured_image(1, 25)
    #img.show_result()
    img.save("test_result/countoured_sobel.bmp")
    img. get_contoured_image(2)
    #img.show_result()
    img.save("test_result/countoured_prewit.bmp")

def test_recognizing(text):
    assert img.get_text_recognized_image(text) == text.replace(' ', '')

def cov_test(test, img, dir,text):
    coverage = cov.Coverage()
    coverage.erase()
    coverage.start()

    for f in test:
        f(img)
    test_recognizing(text)
    coverage.html_report(directory=dir)
    coverage.stop()
    coverage.save()


img = OCRImage("pictures_for_test/cat.bmp")
cov_test([test_filteration, test_countouring, test_binarization], img, 'coverage_reports', "Гори в адУ")


