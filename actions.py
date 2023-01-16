
import win32api, win32con
from enum import Enum

#explore mouse pkg

class Action(Enum):
    CLICK = 1
    RIGHT = 2
    LEFT = 3

def click():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN,0,0,0,0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP,0,0,0,0)

def move_right():
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, 4, 0, 0, 0)

def move_left():
    win32api.mouse_event(win32con.MOUSEEVENTF_MOVE, -4, 0, 0, 0)