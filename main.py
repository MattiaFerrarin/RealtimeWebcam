import cv2
from scripts.ui import UI

cap = cv2.VideoCapture(0)
ui = UI()
running = True

while running:
    try:
        ret, frame = cap.read()
        if not ret:
            print("Error while reading from webcam") # Make this more stable, maybe show a visual hint and then close the app after a few errors or wait until it works
            continue

        # input
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == ord("Q") or key == 27:
            running = False

        elif key == ord(""): # Add other keys and structure
            pass

        # UI and HUD
        frame = ui.draw(
            frame,
            filter_name="",
            face_count="",
            fps=cap.get(cv2.CAP_PROP_FPS)
        )

        cv2.imshow("Webcam", frame)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        running = False
