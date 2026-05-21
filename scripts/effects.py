from pathlib import Path
import numpy as np
import cv2

# Load globally to prevent multiple loading of the same assets
BASE_DIR = Path(__file__).resolve().parent.parent
hat_png = cv2.imread(str(BASE_DIR / "assets/pink_hat.png"), cv2.IMREAD_UNCHANGED)
hacker_png = cv2.imread(str(BASE_DIR / "assets/hacker.png"), cv2.IMREAD_UNCHANGED)
beard_png = cv2.imread(str(BASE_DIR / "assets/beard.png"), cv2.IMREAD_UNCHANGED)
glasses_png = cv2.imread(str(BASE_DIR / "assets/glasses.png"), cv2.IMREAD_UNCHANGED)


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

        frame_bgr = (frame_bgr.astype(np.float32) * mask + blurred.astype(np.float32) * (1 - mask)).astype(np.uint8)
    else:
        frame_bgr = blurred
    return frame_bgr


def overlay_image(frame, img, x, y, width, height):
    frame_h, frame_w = frame.shape[:2]

    resized = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    x1 = max(x, 0)
    y1 = max(y, 0)
    x2 = min(x + width, frame_w)
    y2 = min(y + height, frame_h)

    if x1 >= x2 or y1 >= y2:
        return frame

    crop = resized[y1 - y:y2 - y, x1 - x:x2 - x]

    if crop.shape[2] == 4: # RGBA
        overlay_rgb = crop[:, :, :3].astype(np.float32)
        overlay_alpha = crop[:, :, 3].astype(np.float32) / 255.0
    else: # RGB
        overlay_rgb = crop.astype(np.float32)
        overlay_alpha = np.ones((crop.shape[0], crop.shape[1]), dtype=np.float32)

    overlay_alpha = cv2.GaussianBlur(overlay_alpha, (3, 3), 0)
    overlay_alpha = overlay_alpha[..., np.newaxis]

    roi = frame[y1:y2, x1:x2].astype(np.float32)
    blended = (overlay_rgb * overlay_alpha + roi * (1.0 - overlay_alpha))
    frame[y1:y2, x1:x2] = blended.astype(np.uint8)
    return frame


def pink_hat(frame_bgr, face):
    frame_bgr = frame_bgr.copy()
    img = hat_png
    if face is None or img is None:
        return frame_bgr

    x, y, w, h = face

    item_width = int(w * 1.4)
    aspect_ratio = img.shape[0] / img.shape[1]
    item_height = int(item_width * aspect_ratio)

    item_x = x - int((item_width - w) / 2)
    item_y = y - int(item_height * 1)

    return overlay_image(frame_bgr, img, item_x, item_y, item_width, item_height)

def hacker(frame_bgr, face):
    frame_bgr = frame_bgr.copy()
    img = hacker_png
    if face is None or img is None:
        return frame_bgr

    x, y, w, h = face

    item_width = int(w * 1.2)
    aspect_ratio = img.shape[0] / img.shape[1]
    item_height = int(item_width * aspect_ratio)

    item_x = x - int((item_width - w) / 2)
    item_y = y - int(item_height * 0.2)

    return overlay_image(frame_bgr, img, item_x, item_y, item_width, item_height)

def beard(frame_bgr, face):
    frame_bgr = frame_bgr.copy()
    img = beard_png
    if face is None or img is None:
        return frame_bgr

    x, y, w, h = face

    item_width = int(w * 1.2)
    aspect_ratio = img.shape[0] / img.shape[1]
    item_height = int(item_width * aspect_ratio)

    item_x = x - int((item_width - w) / 2)
    item_y = y - int(item_height * -0.3)

    return overlay_image(frame_bgr, img, item_x, item_y, item_width, item_height)

def glasses(frame_bgr, face):
    frame_bgr = frame_bgr.copy()
    img = glasses_png
    if face is None or img is None:
        return frame_bgr

    x, y, w, h = face

    item_width = int(w * 1.0)
    aspect_ratio = img.shape[0] / img.shape[1]
    item_height = int(item_width * aspect_ratio)

    item_x = x - int((item_width - w) / 2)
    item_y = y - int(item_height * -0.3)

    return overlay_image(frame_bgr, img, item_x, item_y, item_width, item_height)
