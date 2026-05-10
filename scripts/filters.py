import numpy as np
import cv2


def grayscale(frame_bgr):
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)


def negative(frame_bgr):
    return cv2.bitwise_not(frame_bgr)


def sepia(frame_bgr):
    sepia = np.array([[0.27, 0.53, 0.13],
                      [0.38, 0.72, 0.20],
                      [0.45, 0.81, 0.22]])
    return cv2.transform(cv2.cvtColor(frame_bgr,cv2.COLOR_BGR2RGB), sepia)


def solarize(frame_bgr, threshold=128):
    return np.where(frame_bgr < threshold, frame_bgr, 255 - frame_bgr)


def thermal(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_JET)


def spring(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SPRING)


def summer(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_SUMMER)


def autumn(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_AUTUMN)


def winter(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_WINTER)


def hot(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_HOT)


def cool(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_COOL)


def bone(frame_bgr):
    return cv2.applyColorMap(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), cv2.COLORMAP_BONE)
