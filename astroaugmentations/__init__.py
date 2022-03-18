from .image_domain import *
from . import datasets
from . import utils
from . import composed

# cv2 changes laregly not a hinderence when using multithreaded dataloaders,
# but potentially essential for multi-gpu usages.
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

### Information ###
__credits__ = "The Manchester University; The Alan Turing Institute"
__version__ = "0.2.0"
__author__ = "Micah Bowles"
