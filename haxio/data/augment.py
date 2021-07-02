from random import choice

import numpy as np
from PIL import Image

__all__ = ["RandomShift", "DownSample"]


class RandomShift:
    def __init__(self, vertical_ratio: float, horizontal_ratio: float) -> None:
        self._vertical_ratio = vertical_ratio
        self._horizontal_ratio = horizontal_ratio

    @staticmethod
    def _draw_number(range_):
        try:
            h = choice(np.arange(-int(range_), int(range_), ))
        except IndexError:
            h = 0
        return h

    def _draw_affine_matrix(self, image_size):
        big_v, big_h = image_size.size
        h = self._draw_number(self._horizontal_ratio * big_h)
        v = self._draw_number(self._vertical_ratio * big_h)
        return 1, 0, v, 0, 1, h

    def __call__(self, img: Image.Image):
        affine_matrix = self._draw_affine_matrix(img)
        new_img = img.transform(
            img.size, Image.AFFINE, affine_matrix, fillcolor=(255, 255, 255)
        )
        return new_img


class DownSample:
    def __init__(self, down_sample_ratio: 1) -> None:
        self._r = down_sample_ratio

    def __call__(self, image: Image.Image) -> Image.Image:
        h, w = image.size
        h_, w_ = int(h / self._r), int(w / self._r)
        return image.copy().resize((h_, w_))
