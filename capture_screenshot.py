
import mss
import numpy as np

def capture_screenshot():
    with mss.mss() as sct:
        # The screen part to capture
        monitor = {"top": 40, "left": 0, "width": 1280, "height": 720}
        # Grab the data
        img = np.array(sct.grab(monitor))
        return img