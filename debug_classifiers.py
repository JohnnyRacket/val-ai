import cv2
import time
from resize_and_grayscale import resize_img
from capture_screenshot import capture_screenshot

from direction_classifier.direction_classifier import classify_direction
from shot_classifier.shot_classifier import classify_shot

quit = False
while not quit:
    start_time = time.time()
    img = capture_screenshot()
    img = resize_img(img)
    #get classifications
    dir = classify_direction(img)
    # shot = classify_shot(img)


    cv2.putText(img, str(dir), (10, 20 ), cv2.FONT_HERSHEY_SIMPLEX, .8, (200,200,0), 2, cv2.LINE_AA )
    # if (shot == "kill"):
    #     cv2.putText(img, str(shot), (120, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,0), 2, cv2.LINE_AA )
    # if (shot == "hit"):
    #     cv2.putText(img, str(shot), (120, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,255,255), 2, cv2.LINE_AA )
    # if (shot == "miss"):
    #     cv2.putText(img, str(shot), (120, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2, cv2.LINE_AA )

    cv2.putText(img, str(1.0 / (time.time() - start_time)), (580, 20), cv2.FONT_HERSHEY_SIMPLEX, .8, (0,0,255), 2, cv2.LINE_AA )
     
    cv2.imshow("Debug Classifiers", img)
    if cv2.waitKey(25) & 0xFF == ord("q"):
        cv2.destroyAllWindows()
        quit = True