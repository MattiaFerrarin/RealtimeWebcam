import numpy as np
import cv2

def blur_background(frame_bgr, face):
    blurred = cv2.GaussianBlur(frame_bgr, (51, 51), 0)
    if face is not None:
        x, y, w, h = face
        cx, cy = x + w // 2, y + h // 2
        radius = int(min(w, h) * 0.6)

        mask = np.zeros(frame_bgr.shape[:2], dtype=np.float32)
        cv2.circle(mask, (cx, cy), radius, 1, -1)
        mask = cv2.GaussianBlur(mask, (51, 51), 0)
        mask = cv2.merge([mask, mask, mask])

        frame_bgr = (frame_bgr * mask + blurred * (1 - mask)).astype(np.uint8)
    else:
        frame_bgr = blurred
    return frame_bgr
