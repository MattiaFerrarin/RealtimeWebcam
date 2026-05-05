import cv2
import scripts.filters
import scripts.effects
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
            return filters[self.currFilter]
        elif name == "effects":
            n = len(self.effects)
            if n == 0:
                return
            self.currEffect = (self.currEffect + q) % n
            return effects[self.currEffect]


def applyFilter(filters, filter, frame):
    return frame
def applyEffect(effects, effect, frame):
    return frame


if __name__ == "__main__":
    filters = (None, "idk", "idk2")
    effects = (None, "i", "ii", "iii", "iiii")

    cap = cv2.VideoCapture(0)
    ui = UI()
    state = StateHandler(filters, effects)
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

            elif key == ord("w"):
                filterName = state.scroll("filters",1)
            elif key == ord("s"):
                filterName = state.scroll("filters",-1)

            elif key == ord("a"):
                effectName = state.scroll("effects",-1)
            elif key == ord("d"):
                effectName = state.scroll("effects",1)

            frame = applyFilter(filters, filterName, frame)
            frame = applyEffect(effects, effectName, frame)

            # UI and HUD
            frame = ui.draw(
                frame,
                filter_name=filterName,
                effect_name=effectName,
                face_count="",
                fps=cap.get(cv2.CAP_PROP_FPS)
            )

            cv2.imshow("Webcam", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        running = False
