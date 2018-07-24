#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 16:04:54 2018

@author: eo
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes

class SimpleWindow:
    
    windowCount = 0
    
    def __init__(self, name=None, x=None, y=None, timelapse=1, enabled=True):
        
        # Store window name to use as a reference
        self._name = name if name is not None else "Frame - {}".format(self._updateCount())
        
        # Record timelapsing info
        self._timelapse = timelapse
        self._framecount = np.int32(-1)
        
        # Store initial x/y positions
        self._x = int(x) if x is not None else None
        self._y = int(y) if y is not None else None
        
        # Allocate storage for trackbar referencing
        self._trackbars = {}
        
        # Create and move the window, as long as this display is enabled
        if enabled:
            self._createWindow()
        
        else:
            # Blowout functions if this window is disabled
            def blankFunc(*args, **kwargs): return None
            
            # Delete functionality rather than including 'if(enabled):' in every function. Kind of hacky
            self.attachCallback = blankFunc
            self.move = blankFunc
            self.imshow = blankFunc
            self.addTrackbar = blankFunc
            self.readTrackbar = blankFunc
            self.reset = blankFunc
            self.restart = blankFunc
            self.close = blankFunc
            self.exists = blankFunc
        
    # ................................................................................................................. 
        
    def attachCallback(self, mouse_callback, callback_data):
        
        cv2.setMouseCallback(self._name, mouse_callback, callback_data)
       
    # ................................................................................................................. 
    
    def move(self, x, y):
        cv2.moveWindow(self._name, x, y)
        self._x = x
        self._y = y
        
    # ................................................................................................................. 
        
    def imshow(self, frame):
        
        # Update frame count, used for timelapsing
        self._framecount += 1
        
        # Check if the window exists (by looking for window properties)
        windowExists = self.exists()
        
        # Skip display update if timelapsing
        if (self._framecount % self._timelapse) != 0:
            return windowExists
        
        # Only update showing if the window exists
        if windowExists:
            cv2.imshow(self._name, frame)
        
        return windowExists
    
    # ................................................................................................................. 
    
    def addTrackbar(self, bar_name, start_value, max_value=100):
        
        maximum_value = int(max_value)
        starting_value = min(int(start_value), maximum_value)
        
        cv2.createTrackbar(bar_name, self._name, starting_value, maximum_value, lambda x: None)
        self._trackbars[bar_name] = starting_value
    
    # ................................................................................................................. 
    
    def readTrackbar(self, bar_name):
        
        # Get the trackbar value and check if it changed
        trackbar_value = cv2.getTrackbarPos(bar_name, self._name)
        trackbar_changed = self._trackbars[bar_name] != trackbar_value
        
        # Record the new (possible same) trackbar value
        self._trackbars[bar_name] = trackbar_value
        
        return trackbar_changed, trackbar_value
    
    # .................................................................................................................    
    
    def reset(self):
        self._createWindow()    # Similar to restart, but will re-position open windows as well
    
    # .................................................................................................................    
    
    def restart(self):
        if not self.exists(): self._createWindow()
    
    # .................................................................................................................    
    
    def close(self):
        cv2.destroyWindow(self._name)
        
    # .................................................................................................................  
    
    def exists(self):
        return cv2.getWindowProperty(self._name, 1) > 0
    
    # ................................................................................................................. 
    
    def _createWindow(self):
        
        # Create window
        cv2.namedWindow(self._name)
            
        # Re-position the window if x/y positions are specified
        if (self._x is not None) and (self._y is not None):
            self.move(self._x, self._y)
                
    # ................................................................................................................. 
    
    @classmethod
    def _updateCount(cls):
        cls.windowCount +=1
        return cls.windowCount
        
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions
    
def breakByKeypress(frame_delay=1):
    
    # Get keypress
    keyPress = cv2.waitKey(frame_delay) & 0xFF
    
    # Check if q or Esc where pressed, in which return a signal to break execution loop
    requestBreak = (keyPress == ord('q')) | (keyPress == 27)
    
    return requestBreak, keyPress 

# .....................................................................................................................

def arrowKeys(keypress):
    
    # Key reference (on linux at least)
    # left arrow  = 81
    # up arrow    = 82
    # right arrow = 83
    # down arrow  = 84
    
    arrowPressed = (80 < keypress < 85)
    if arrowPressed:
        return (arrowPressed, (int(keypress == 83) - int(keypress == 81), int(keypress == 84) - int(keypress == 82)))
    else:
        return (arrowPressed, (0, 0))

# .....................................................................................................................
    

# .....................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap



