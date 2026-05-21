import os
import sys
import math
from pathlib import Path
import cv2
import numpy as np
import time
import datetime
import mediapipe as mp
from fontTools.misc.cython import returns

from scripts import filters
from scripts import effects
from scripts.ui import UI


# -- Camera selection --
def make_grid(frames, grid_shape=None, cell_size=(320, 240)):
    n = len(frames)
    if n == 0:
        return None
    if grid_shape is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    else:
        cols, rows = grid_shape
    w, h = cell_size
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)

    for i, (idx, frame) in enumerate(frames):
        if frame is None:
            continue
        frame = cv2.resize(frame, (w, h))
        cv2.rectangle(frame,(0, 0),(w - 1, h - 1),(255, 255, 255),3)
        cv2.putText(frame, f"Cam {idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        row = i // cols
        col = i % cols
        y1 = row * h
        y2 = y1 + h
        x1 = col * w
        x2 = x1 + w

        grid[y1:y2, x1:x2] = frame

    return grid

def find_available_cameras(max_index=10):
    available = []
    for i in range(max_index):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                available.append(i)
        cap.release()
    return available

def select_camera():
    cameras = find_available_cameras()
    if not cameras:
        raise RuntimeError("No cameras found")
    if len(cameras) == 1:
        print(f"[CAM] Only one camera found: {cameras[0]}")
        return cv2.VideoCapture(cameras[0])

    print(f"[CAM] Multiple cameras found: {cameras}")
    print("Press number key to select camera, Q to quit")
    caps = {i: cv2.VideoCapture(i) for i in cameras}
    selected = cameras[0]
    while True:
        frames = []
        for idx in cameras:
            cap = caps[idx]
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((240, 320, 3), dtype=np.uint8)
                cv2.putText(frame, "NO SIGNAL", (50, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            frames.append((idx, frame))
        grid = make_grid(frames)
        cv2.imshow("Camera Selection", grid)
        key = cv2.waitKey(1) & 0xFF
        # number selection
        if ord('0') <= key <= ord('9'):
            cam_id = int(chr(key))
            if cam_id in cameras:
                selected = cam_id
                break
        if key == ord('q') or key == ord("Q") or key == 27:
            selected = None
            break
    # cleanup
    for cap in caps.values():
        cap.release()
    cv2.destroyAllWindows()
    if selected is None:
        return None
    print(f"[CAM] Selected camera: {selected}")
    return cv2.VideoCapture(selected)
# -- End of Camera Selection --

# -- Helper classes --
class StateHandler:
    def __init__(self, filters, effects):
        self.filters = filters
        self.effects = effects
        self.currFilter = 0
        self.currEffect = 0
        self.flipped = False
    def scroll(self, name, q):
        if name == "filters":
            n = len(self.filters)
            if n == 0:
                return
            self.currFilter = (self.currFilter + q) % n
            return self.filters[self.currFilter]
        elif name == "effects":
            n = len(self.effects)
            if n == 0:
                return
            self.currEffect = (self.currEffect + q) % n
            return self.effects[self.currEffect]

    def flip(self):
        self.flipped = not self.flipped

class FaceHandler:
    def __init__(self, alpha=0.7):
        self.alpha = alpha
        self.prev = None

    def detectFaces(self, frame, detector):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        results = detector.process(rgb)
        rgb.flags.writeable = True
        faces = []
        if results.detections:
            h, w, _ = frame.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                bw = int(bbox.width * w)
                bh = int(bbox.height * h)
                x = max(0, x)
                y = max(0, y)
                bw = min(w - x, bw)
                bh = min(h - y, bh)
                faces.append((x, y, bw, bh))
        return faces

    def smooth(self, face):
        if face is None:
            self.prev = None
            return None

        x, y, w, h = face

        if self.prev is None:
            self.prev = (x, y, w, h)
            return face

        px, py, pw, ph = self.prev

        sx = int(self.alpha * px + (1 - self.alpha) * x)
        sy = int(self.alpha * py + (1 - self.alpha) * y)
        sw = int(self.alpha * pw + (1 - self.alpha) * w)
        sh = int(self.alpha * ph + (1 - self.alpha) * h)

        self.prev = (sx, sy, sw, sh)
        return (sx, sy, sw, sh)
# -- End of Helper Classes --

# -- Helper functions --
def applyFilter(filter, frame):
    if filter is None:
        return frame
    elif filter == "grayscale":
        return filters.grayscale(frame)
    elif filter == "negative":
        return filters.negative(frame)
    elif filter == "sepia":
        return filters.sepia(frame)
    elif filter == "bone":
        return filters.bone(frame)
    elif filter == "solarize":
        return filters.solarize(frame)
    elif filter == "thermal":
        return filters.thermal(frame)
    elif filter == "spring":
        return filters.spring(frame)
    elif filter == "summer":
        return filters.summer(frame)
    elif filter == "autumn":
        return filters.autumn(frame)
    elif filter == "winter":
        return filters.winter(frame)
    elif filter == "hot":
        return filters.hot(frame)
    elif filter == "cool":
        return filters.cool(frame)
    elif filter == "cartoon":
        return filters.cartoon(frame)
    elif filter == "pixelate":
        return filters.pixelate(frame)
    elif filter == "vignette":
        return filters.vignette(frame)
    else:
        return frame
def applyEffect(effect, frame, face):
    if effect is None:
        return frame
    elif effect == "blur back":
        return effects.blur_background(frame, face)
    elif effect == "pink hat":
        return effects.pink_hat(frame, face)
    elif effect == "hacker":
        return effects.hacker(frame, face)
    elif effect == "beard":
        return effects.beard(frame, face)
    elif effect == "glasses":
        return effects.glasses(frame, face)
    else:
        return frame
# -- End of Helper functions --


if __name__ == "__main__":
    PROGRAM_NAME = "RealtimeWebcam"

    filtersList = (None, "grayscale", "negative", "sepia", "bone", "solarize", "thermal", "spring", "summer",
                   "autumn", "winter", "hot", "cool", "cartoon", "pixelate", "vignette")
    effectsList = (None, "blur back", "pink hat", "hacker", "beard", "glasses")

    faceDetector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = select_camera()
    if cap is None:
        sys.exit(0)
    ui = UI()
    state = StateHandler(filtersList, effectsList)
    face_handler = FaceHandler(alpha=0.7)
    running = True

    try:
        # Create window
        cv2.namedWindow("Webcam", cv2.WINDOW_AUTOSIZE)
        # Active effect and filter
        filterName = None
        effectName = None
        # FPS
        prev_time = time.time()
        fps_avg = 0
        fps_alpha = 0.1
        # Recording
        recording = False
        writer = None
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        record_fps = 30  # fixed output FPS

        while running:
            ret, frame = cap.read()
            if not ret:
                print(
                    "Error while reading from webcam")  # Make this more stable, maybe show a visual hint and then close the app after a few errors or wait until it works
                continue

            # input
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q") or key == ord("Q") or key == 27:
                running = False

            elif key == ord("f") or key == ord("F"):
                state.flip()

            elif key == ord("z") or key == ord("Z"):
                filterName = state.scroll("filters",-1)
            elif key == ord("x") or key == ord("X"):
                filterName = state.scroll("filters",1)

            elif key == ord("c") or key == ord("C"):
                effectName = state.scroll("effects",-1)
            elif key == ord("v") or key == ord("V"):
                effectName = state.scroll("effects",1)

            faces = face_handler.detectFaces(frame, faceDetector)
            face = max(faces, key=lambda f: f[2] * f[3], default=None)
            face = face_handler.smooth(face)

            frame = applyEffect(effectName, frame, max(faces, key=lambda f: f[2] * f[3], default=None))
            frame = applyFilter(filterName, frame)
            if state.flipped:
                frame = cv2.flip(frame, 1)

            if key == ord("s") or key == ord("S"):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                home = Path.home()
                candidate_dirs = [home / "Pictures" / PROGRAM_NAME, home / "Downloads" / PROGRAM_NAME, Path.cwd() / "screenshots"]
                for directory in candidate_dirs:
                    try:
                        directory.mkdir(parents=True, exist_ok=True)
                        file_path = directory / f"img_{timestamp}.jpg"
                        if cv2.imwrite(str(file_path), frame):
                            break
                    except Exception as e:
                        print(f"Error occurred while saving screenshot: no viable path found for directory: {directory}, falling back to next choice")

                # Creates the screenshot animation
                frozen = frame.copy()
                h, w = frozen.shape[:2]
                for i in range(10):
                    temp = frozen.copy()
                    t = 1 - (i / 10)
                    thickness = int(5 + 25 * t)
                    cv2.rectangle(temp,(0, 0),(w - 1, h - 1),(255, 255, 255),thickness)
                    overlay = np.full_like(temp, 255)
                    alpha = 0.35 * t
                    temp = cv2.addWeighted(overlay,alpha,temp,1 - alpha,0)

                    cv2.imshow("Webcam", temp)
                    cv2.waitKey(20)

            elif key == ord("r") or key == ord("R"):
                recording = not recording
                if recording:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    home = Path.home()
                    candidate_dirs = [home / "Videos" / PROGRAM_NAME, home / "Downloads" / PROGRAM_NAME, Path.cwd() / "recordings"]
                    writer = None
                    filename = None

                    for directory in candidate_dirs:
                        try:
                            directory.mkdir(parents=True, exist_ok=True)
                            filename = directory / f"video_{timestamp}.mp4"
                            h, w = frame.shape[:2]
                            writer = cv2.VideoWriter(str(filename), fourcc, record_fps, (w, h))

                            if writer.isOpened():
                                print(f"[REC] Started recording: {filename}")
                                break
                            else:
                                writer.release()
                                writer = None

                        except Exception as e:
                            print(f"[REC] Failed directory {directory}, trying next...")

                    if writer is None:
                        print("[REC] ERROR: No valid path found for video recording")
                        recording = False

                else:
                    if writer is not None:
                        writer.release()
                        writer = None
                        print("[REC] Stopped recording")

            # If recording write frame
            if recording and writer is not None:
                writer.write(frame)

            # Calculate averaged FPS
            curr_time = time.time()
            instant_fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            fps_avg = (1 - fps_alpha) * fps_avg + fps_alpha * instant_fps

            # UI and HUD
            frame = ui.draw(
                frame,
                filter_name=filterName,
                effect_name=effectName,
                face_count=len(faces),
                fps=round(fps_avg, 1),
                flipped=state.flipped,
                recording=recording
            )

            cv2.imshow("Webcam", frame)
    except Exception as e:
        print(f"Exception captured:\n{e}")
    finally:
        if writer is not None:
            writer.release()
        cap.release()
        cv2.destroyAllWindows()
