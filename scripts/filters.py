import numpy as np
import cv2


def grayscale(frame_bgr):
    return cv2.cvtColor(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def negative(frame_bgr):
    return cv2.bitwise_not(frame_bgr.copy())


def sepia(frame_bgr):
    sepia = np.array([[0.27, 0.53, 0.13],
                      [0.38, 0.72, 0.20],
                      [0.45, 0.81, 0.22]])
    return cv2.cvtColor(cv2.transform(cv2.cvtColor(frame_bgr.copy(),cv2.COLOR_BGR2RGB), sepia), cv2.COLOR_RGB2BGR)


def solarize(frame_bgr, threshold=128):
    return np.where(frame_bgr < threshold, frame_bgr.copy(), 255 - frame_bgr)


def thermal(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def spring(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SPRING)


def summer(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SUMMER)


def autumn(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_AUTUMN)


def winter(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_WINTER)


def hot(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)


def cool(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_COOL)


def bone(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_BONE)


def cartoon(frame_bgr):
    color = frame_bgr.copy()
    for i in range(3):
        color = cv2.bilateralFilter(color,d=9,sigmaColor=75,sigmaSpace=75)
    gray = cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(gray, 40, 80)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    edges = cv2.morphologyEx(edges,cv2.MORPH_CLOSE,np.ones((3, 3), np.uint8))
    edges = cv2.bitwise_not(edges)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    return cv2.bitwise_and(color, edges)


def pixelate(frame_bgr, pixel_size=10):
    h, w = frame_bgr.shape[:2]
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    temp = cv2.resize(frame_bgr.copy(),(small_w, small_h),interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp,(w, h),interpolation=cv2.INTER_NEAREST)
    return pixelated


def vignette(frame_bgr, strength=0.6):
    h, w = frame_bgr.shape[:2]
    kernel_x = cv2.getGaussianKernel(w, w * strength)
    kernel_y = cv2.getGaussianKernel(h, h * strength)
    kernel = kernel_y * kernel_x.T # (.T transposes the kernel from column kernel to row)
    mask = kernel / kernel.max()
    vignetted = np.empty_like(frame_bgr.copy(), dtype=np.float32)
    for i in range(3):
        vignetted[:, :, i] = frame_bgr[:, :, i] * mask
    return np.clip(vignetted, 0, 255).astype(np.uint8)
