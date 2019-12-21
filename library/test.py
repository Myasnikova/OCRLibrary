import coverage as cov
from image import OCRImage
coverage = cov.Coverage()
coverage.erase()
coverage.start()

ocr = OCRImage("pictures_for_test/cat.bmp") 
ocr.get_filtered_image(1)

coverage.html_report(directory = "htmlcov/filtred/median")

ocr.get_filtered_image(2)

coverage.html_report(directory = "htmlcov/filtred/weight")

ocr.get_filtered_image(3)

coverage.html_report(directory = "htmlcov/filtred/rank")

#ci.test()

#coverage.html_report(directory = "htmlcov/contoured")


coverage.stop()
