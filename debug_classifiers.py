import cv2
import time
import mss
import mss.tools
import queue
import multiprocessing
import numpy as np
from filter_images import resize_img, direction_classifier_image_filter
from capture_screenshot import capture_screenshot
import actions

from direction_classifier.direction_classifier import classify_direction
# from shot_classifier.shot_classifier import classify_shot

# def grab(queue):
#     # rect = {"top": 30, "left": 0, "width": 1280, "height": 720}
#     #     # Grab the data

#     # with mss.mss() as sct:
#     #     for _ in range(2_000):
#     #         queue.put(sct.grab(rect))
    

def save(queue):
    number = 0
    while "there are screenshots":
        img = queue.get()
        if img is None:
            print('none')
            break
        cv2.imwrite((r'screenshots\img' + str(number + 1) + '.png'), img)
        number += 1


if __name__ == '__main__':
    quit = False
    record = False
    queue = multiprocessing.Queue()

    if(record):
        # multiprocessing.Process(target=grab, args=(queue,)).start()
        multiprocessing.Process(target=save, args=(queue,)).start()

    while not quit:
        start_time = time.time()
        img = capture_screenshot()       
        if(record):
            queue.put(img)
            
        img = direction_classifier_image_filter(img)
        
        #get classifications
        dir = classify_direction(img)
        # shot = classify_shot(img)
        # if(dir == 'left'):
        #     actions.move_left()
        # if(dir == 'right'):
        #     actions.move_right()
        if(dir == 'center'):
            actions.click()



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
            queue.put(None)
            

