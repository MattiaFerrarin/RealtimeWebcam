import numpy as np
import cv2


class UI:
    def __init__(self):
        pass

    def draw(self, frame, filter_name = None, effect_name = None, face_count = None, fps = None, flipped=False):
        frame = frame.copy()

        if filter_name is not None:
            cv2.putText(frame, f"Filter: {filter_name}", (10, np.shape(frame)[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if effect_name is not None:
            text = f"Effect: {effect_name}"
            (text_width, text_height), baseline = cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            x = frame.shape[1] - text_width - 10
            y = frame.shape[0] - 30
            cv2.putText(frame,text,(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 0),2)

        if face_count is not None:
            cv2.putText(frame, f"Faces: {face_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if fps is not None:
            cv2.putText(frame, f"FPS: {int(fps)}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)

        if flipped:
            (text_width, text_height), baseline = cv2.getTextSize("Flipped",cv2.FONT_HERSHEY_SIMPLEX,0.7,2)
            x = frame.shape[1] - text_width - 10
            y = 30
            cv2.putText(frame,"Flipped",(x, y),cv2.FONT_HERSHEY_SIMPLEX,0.7,(0, 255, 255),2)

        return frame