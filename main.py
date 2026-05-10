import cv2
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
    else:
        return frame
def applyEffect(effect, frame, face):
    if effect is None:
        return frame
    elif effect == "blur back":
        return effects.blur_background(frame, face)
    else:
        return frame
def detectFaces(frame, detector):
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


if __name__ == "__main__":
    filtersList = (None, "grayscale", "negative", "sepia", "bone", "solarize", "thermal", "spring", "summer", "autumn", "winter", "hot", "cool")
    effectsList = (None, "blur back", "ii", "iii", "iiii")

    faceDetector = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.6)

    cap = cv2.VideoCapture(0)
    ui = UI()
    state = StateHandler(filtersList, effectsList)
    running = True

    try:
        filterName = None
        effectName = None
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

            elif key == ord("w") or key == ord("W"):
                filterName = state.scroll("filters",1)
            elif key == ord("s") or key == ord("S"):
                filterName = state.scroll("filters",-1)

            elif key == ord("a") or key == ord("A"):
                effectName = state.scroll("effects",-1)
            elif key == ord("d") or key == ord("D"):
                effectName = state.scroll("effects",1)

            faces = detectFaces(frame, faceDetector)

            frame = applyFilter(filterName, frame)
            frame = applyEffect(effectName, frame, max(faces, key=lambda f: f[2] * f[3], default=None))

            if key == ord("s") or key == ord("S"):
                cv2.imwrite(f"img_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{}.jpg", frame)

            # UI and HUD
            frame = ui.draw(
                frame,
                filter_name=filterName,
                effect_name=effectName,
                face_count=len(faces),
                fps=cap.get(cv2.CAP_PROP_FPS)
            )

            cv2.imshow("Webcam", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        running = False
