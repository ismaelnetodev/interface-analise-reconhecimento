import cv2
import numpy as np
from skimage.feature import local_binary_pattern

def anti_spoofing(face_img, threshold=3):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    return np.var(lbp) > threshold
