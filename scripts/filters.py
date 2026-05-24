import numpy as np
import cv2


# Converts the image to grayscale while keeping 3 color channels (BGR format)
def grayscale(frame_bgr):
    return cv2.cvtColor(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


# Inverts all pixel colors to create a negative image effect
def negative(frame_bgr):
    return cv2.bitwise_not(frame_bgr.copy())


# Applies a sepia-tone filter
def sepia(frame_bgr):
    sepia = np.array([[0.27, 0.53, 0.13],
                      [0.38, 0.72, 0.20],
                      [0.45, 0.81, 0.22]])
    return cv2.cvtColor(cv2.transform(cv2.cvtColor(frame_bgr.copy(),cv2.COLOR_BGR2RGB), sepia), cv2.COLOR_RGB2BGR)


# Solarizes the image by inverting pixels above a brightness threshold
def solarize(frame_bgr, threshold=128):
    return np.where(frame_bgr < threshold, frame_bgr.copy(), 255 - frame_bgr)


# Applies a thermal heatmap-style color effect using the JET colormap
def thermal(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


# Applies the SPRING colormap for pink tones
def spring(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SPRING)


# Applies the SUMMER colormap for green tones
def summer(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SUMMER)


# Applies the AUTUMN colormap for orange tones
def autumn(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_AUTUMN)


# Applies the WINTER colormap for blue tones
def winter(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_WINTER)


# Applies the HOT colormap
def hot(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)


# Applies the COOL colormap
def cool(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_COOL)


# Applies the BONE colormap
def bone(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr.copy(), cv2.COLOR_BGR2GRAY), cv2.COLORMAP_BONE)


# Creates a cartoon effect by smoothing colors and emphasizing edges
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


# Pixelates the image by shrinking and enlarging it with nearest-neighbor scaling
def pixelate(frame_bgr, pixel_size=10):
    h, w = frame_bgr.shape[:2]
    small_w = max(1, w // pixel_size)
    small_h = max(1, h // pixel_size)
    temp = cv2.resize(frame_bgr.copy(),(small_w, small_h),interpolation=cv2.INTER_LINEAR)
    pixelated = cv2.resize(temp,(w, h),interpolation=cv2.INTER_NEAREST)
    return pixelated


# Applies a vignette effect that darkens the image toward the edges
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