Example of using library
========================

Необходимо импортировать класс :class:`~image.OCRImage`, инициализировать его экземпляр и использовать необходимые вам функции.

Пример использования библиотеки:

.. code:: python

    from image import OCRImage

    img = OCRImage("path/to/image.bmp")
    img.binary_image_object.cristian_binarisation()
    img.show_result()

