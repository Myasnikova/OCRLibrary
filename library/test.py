import coverage as cov
from image import OCRImage
coverage = cov.Coverage()
coverage.erase()
coverage.start()

ocr = OCRImage("pictures_for_test/text.bmp") 

ocr.get_filtered_image(1)

coverage.html_report(directory = "htmlcov/filtred/median")

ocr.get_filtered_image(2)

coverage.html_report(directory = "htmlcov/filtred/weight")

ocr.get_filtered_image(3)

coverage.html_report(directory = "htmlcov/filtred/rank")

ocr.get_binary_image(2)

coverage.html_report(directory = "htmlcov/binary/cristian")

ocr.get_contoured_image(1,5)# нужно определить значение параметра t

coverage.html_report(directory = "htmlcov/contour/sobel")

ocr.get_contoured_image(2)

coverage.html_report(directory = "htmlcov/contour/pruit")

coverage.stop()
