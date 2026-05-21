import os
from pathlib import Path
import cv2
import numpy as np
import time
import datetime
import mediapipe as mp
from scripts import filters
from scripts import effects
from scripts.ui import UI

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


if __name__ == "__main__":

    filtersList = (None, "grayscale", "negative", "sepia", "bone", "solarize", "thermal", "spring", "summer",
                   "autumn", "winter", "hot", "cool", "cartoon", "pixelate", "vignette")
    effectsList = (None, "blur back", "pink hat", "hacker", "beard", "glasses")

    faceDetector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(0)
    ui = UI()
    state = StateHandler(filtersList, effectsList)
    running = True

    try:
        filterName = None
        effectName = None
        prev_time = time.time()
        fps_avg = 0
        fps_alpha = 0.1
        face_handler = FaceHandler(alpha=0.7)
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

            frame = applyFilter(filterName, frame)
            frame = applyEffect(effectName, frame, max(faces, key=lambda f: f[2] * f[3], default=None))
            if state.flipped:
                frame = cv2.flip(frame, 1)

            if key == ord("s") or key == ord("S"):
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                os.makedirs("screenshots", exist_ok=True)
                home = Path.home()
                candidate_dirs = [home / "Pictures", home / "Downloads", Path.cwd() / "screenshots"]
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
                flipped=state.flipped
            )

            cv2.imshow("Webcam", frame)
    except Exception as e:
        print(f"Exception captured:\n{e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()

