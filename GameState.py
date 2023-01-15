from actions import Action, click, move_left, move_right
from capture_screenshot import capture_screenshot
class GameState(object):
    def __init__(self):
        self.terminal = False
        self.kills = 0

    def frame_step(self, action):
        match action:
            case Action.CLICK:
                click()
            case Action.LEFT:
                move_left()
            case Action.RIGHT:
                move_right()
            case _:
                print('no action') 
        
        img = capture_screenshot()

        # need to get reward
        # check screenshot for kill animation
        # need to check for terminal

        if (self.kills >= 10):
            self.terminal = True

        return img, reward, self.terminal




