import cv2
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
from skimage.morphology import skeletonize
from skimage import img_as_bool, io
import math
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline


def find_lines(cmi):
    # find median line using skeleton
    _, binary_image = cv2.threshold(cmi, 120, 255, cv2.THRESH_BINARY)
    binary_image = img_as_bool(binary_image)
    skeleton = skeletonize(binary_image)
    y_indices, x_indices = np.where(skeleton)
    _, unique_indices = np.unique(x_indices, return_index=True)
    x_indices = x_indices[unique_indices]
    y_indices = y_indices[unique_indices]
    mid_line = np.stack([x_indices, y_indices], axis=1)

    # find edge line
    edges = cv2.Canny(cmi, 200, 200)
    yy, xx = np.where(edges>0)
    upper_line = []
    lower_line = []
    for x, y in zip(xx, yy):
        mid_y = y_indices[np.where(x_indices == x)[0]]
        if mid_y.size > 0:
            if y < mid_y:
                upper_line.append((x, y))
            else:
                lower_line.append((x, y))
    upper_line = np.array(upper_line)
    idx = np.argsort(upper_line[:, 0])
    upper_line = upper_line[idx]
    lower_line = np.array(lower_line)
    idx = np.argsort(lower_line[:, 0])
    lower_line = lower_line[idx]

    return mid_line, upper_line, lower_line


def fitting(x, y, degree=4):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    polyreg = make_pipeline(PolynomialFeatures(degree),LinearRegression())
    polyreg.fit(x, y)
    return polyreg


def first_derivative(x, polyreg):
    coeffs = polyreg.named_steps['linearregression'].coef_[0]
    slope = sum([coeffs[i] * i * x**(i-1) for i in range(1, len(coeffs))])
    return slope


def draw(x1, y1, m):
    m_perpendicular = -1 / m
    x = np.linspace(x1-1, x1+1, 100)
    y = m_perpendicular * (x - x1) + y1
    plt.plot(x, y, 'w', linewidth=0.8)


def measure_one_point(x1, mid_polyreg, upper_polyreg, lower_polyreg, vertical_spacing, horizontal_spacing):
    x_range = np.linspace(x1-20, x1+20, 500).reshape(-1, 1)
    
    y1 = mid_polyreg.predict(np.array([[x1]]))[0, 0]
    k = -1 / first_derivative(x1, mid_polyreg)
    b = y1 - k*x1
    y_line = k*x_range + b

    y_poly = upper_polyreg.predict(x_range)
    diff = abs(y_poly - y_line)
    min_index = np.argmin(diff)
    inter_upper_x = x_range[min_index]
    inter_upper_y = y_poly[min_index]

    y_poly = lower_polyreg.predict(x_range)
    diff = abs(y_poly - y_line)
    min_index = np.argmin(diff)
    inter_lower_x = x_range[min_index]
    inter_lower_y = y_poly[min_index]

    # d = math.sqrt((inter_upper_x-inter_lower_x)**2 + (inter_lower_y - inter_upper_y)**2)
    thickness = math.sqrt(
        ((inter_upper_x-inter_lower_x)*horizontal_spacing)**2 + 
        ((inter_lower_y - inter_upper_y)*vertical_spacing)**2)
    return thickness


def moving_average(thickness, n):
    if n % 2 == 0:
        raise ValueError("n must be an odd number.")
    padding = n // 2
    extended_thickness = np.pad(thickness, (padding, padding), 'edge')
    averaged_thickness = np.zeros_like(thickness, dtype=float)
    for i in range(padding, len(extended_thickness) - padding):
        averaged_thickness[i - padding] = np.mean(extended_thickness[i - padding:i + padding + 1])
    return averaged_thickness


def measure_CMIT(cmi, vertical_spacing, horizontal_spacing, degree=4):
    cmi[:, 0:20] = 0
    cmi[:, -20:] = 0
    cmi = np.uint8(cmi * 255)
    mid_line, upper_line, lower_line = find_lines(cmi)
    mid_polyreg = fitting(mid_line[:, 0], mid_line[:, 1], degree)
    upper_polyreg = fitting(upper_line[:, 0], upper_line[:, 1], degree)
    lower_polyreg = fitting(lower_line[:, 0], lower_line[:, 1], degree)
    CMIT = []
    for x1 in range(mid_line[:, 0].min(), mid_line[:, 0].max(), 10):
        thickness = measure_one_point(
            x1, 
            mid_polyreg, 
            upper_polyreg, 
            lower_polyreg, 
            vertical_spacing, 
            horizontal_spacing
            )
        CMIT.append(thickness)
    CMIT = moving_average(CMIT, 3)
    CMIT = round(CMIT.mean(), 3)
    return CMIT, upper_polyreg, lower_polyreg
   