# -*- coding: utf-8 -*-
"""
==============
Edge operators
==============
Edge operators are used in image processing within edge detection algorithms.
They are discrete differentiation operators, computing an approximation of the
gradient of the image intensity function.
"""
import numpy as np
import matplotlib.pyplot as plt

from skipp import filters
from skimage.data import camera
from skimage.util import compare_images


image = camera().astype(np.float32)
edge_prewitt = filters.prewitt(image)
edge_sobel = filters.sobel(image)

fig, axes = plt.subplots(ncols=2, sharex=True, sharey=True,
                         figsize=(8, 4))

axes[0].imshow(edge_prewitt, cmap=plt.cm.gray)
axes[0].set_title('Prewitt Edge Detection')

axes[1].imshow(edge_sobel, cmap=plt.cm.gray)
axes[1].set_title('Sobel Edge Detection')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()
