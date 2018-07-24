#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:48:19 2018

@author: eo
"""


import os
import cv2
import datetime as dt

from statistics import median
from multiprocessing import Process, Pipe, Event
from functools import partial

from local.lib.video.windowing import SimpleWindow, breakByKeypress
from local.lib.video.io import setupVideoCapture, typeOfSource
from local.lib.utils.files import rtspString, rtspFromCommandLine, checkSavePath, saveHistoryFile, loadHistoryFile
from local.lib.utils.misc import hardQuit


# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes

class Timer:
    
    
    def __init__(self, source_type, vidFPS, global_start, global_finish, chunk_size_mins=5):
        
        
        self._source_type = source_type.copy()
        self._target_fps = vidFPS
        self._frame_count = 0
        
        self._current_time = dt.datetime.now()
        self._frame_time_update = dt.timedelta(seconds=1.0/vidFPS)
        
        self._global_start = global_start
        self._global_finish = global_finish
        
        self._chunk_delta = dt.timedelta(minutes=chunk_size_mins)
        self._backward_delta = self._startTimeDelta()
        
        self._curr_chunk = self._global_start
        self._next_chunk = self._global_start
        
        
    # .................................................................................................................
    
    def newChunkInfo(self):
        
        # Update the chunking times
        self._curr_chunk, self._next_chunk = self._next_chunk, (self._next_chunk + self._chunk_delta)
        
        # Get chunk timings
        chunk_start = self._curr_chunk
        chunk_end = self._next_chunk
        
        # Modify times for video files
        if self._source_type["video"]:
            chunk_start += self._backward_delta
            chunk_end += self._backward_delta

        return chunk_start, chunk_end
            
    # .................................................................................................................
    
    def getTime(self):
        
        # Keep track of the current (frame) timing differently based on using video files or live streams
        self._frame_count += 1
        if self._source_type["video"]:
            self._current_time += self._frame_time_update
        else:
            self._current_time = dt.datetime.now()
            
    # .................................................................................................................
            
    def frameCount(self):
        return self._frame_count
    
    # .................................................................................................................
    
    def waitToStart(self):
        return (self._current_time < self._global_start)
    
    # .................................................................................................................
    
    def isFinished(self):
        return (self._current_time > self._global_finish)
    
    # .................................................................................................................
    
    def needNewChunk(self):
        return (self._current_time >= self._next_chunk)
    
    # .................................................................................................................
    
    def timestamp(self):
        
        if self._source_type["video"]:
            return self._current_time + self._backward_delta
        else:
            return self._current_time
    
    # .................................................................................................................
    
    def _startTimeDelta(self):
        
        # Create a timedelta to get zero time offsets
        timeless_start = dt.datetime(self._global_start.year, self._global_start.month, self._global_start.day)
        backwards_delta = timeless_start - self._global_start
        return backwards_delta
    
    # .................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions

closeall = cv2.destroyAllWindows

# .....................................................................................................................

def numericalUserInput(user_input, return_type=int, default_value=1):
    
    output_as_number = None
    conversion_error = False
    if not user_input:
        output_as_number = default_value
    else:
        try:
            output_as_number = return_type(user_input)
        except ValueError:
            output_as_number = default_value
            conversion_error = True
            
    return conversion_error, output_as_number 

# .....................................................................................................................

def chunkWriter(out_path, global_start_time, global_finish_time, timelapse_factor, file_ext, 
                fcc_code, fps, frameSize, isColor):
    
    # Some convenient terms to record
    date_format = "%Y-%m-%d"
    time_format = "%H_%M_%S"
    datetime_format = " ".join([date_format, time_format])
    
    # Figure out whether the recording takes place over a single day
    same_day = (global_start_time.date() == global_finish_time.date())
    
    # Determine string used to name folder for recording files
    start_string = global_start_time.strftime(datetime_format)
    end_string = global_finish_time.strftime(time_format) if same_day else global_finish_time.strftime(datetime_format)
    
    # Build folder name and full base pathing
    timespan_string = " ".join([start_string, "to", end_string])
    base_path = os.path.join(out_path, timespan_string)
    
    # Create fourcc code
    fourcc = cv2.VideoWriter_fourcc(*fcc_code)    
    
    
    # Define chunk creating function which returns actual video writer objects
    def startNewChunk(start_time, end_time, 
                      base_path, is_same_day, timelapse_factor, file_ext, fourcc, fps, frameSize, isColor):
        
        # Get start/end dates and times for convenience
        start_date_string = start_time.strftime("%Y-%m-%d")
        start_time_string = start_time.strftime("%H_%M_%S")
        end_time_string = end_time.strftime("%H_%M_%S")
        
        # Get a string to represent timelapse factor, if timelapsing!
        tl_string = " (TLx{})".format(timelapse_factor) if timelapse_factor > 1 else ""
        
        # Build output pathing, based on whether the file crosses days
        videoOutPath = base_path if is_same_day else os.path.join(base_path, start_date_string)
        videoOutName = "".join([start_date_string, " ", start_time_string, 
                                " to ", end_time_string, tl_string, file_ext])
    
        # Build final output pathing and make sure it exists
        videoOutSource = os.path.join(videoOutPath, videoOutName)
        checkSavePath(videoOutSource)
        
        # Some feedback
        print("")
        print("Starting new video chunk:")
        print(os.path.basename(videoOutSource))
        
        return cv2.VideoWriter(videoOutSource, fourcc, fps, frameSize, isColor), videoOutSource
    
    
    
    # Return the new chunk builder, without the start/end time arguments populated
    return partial(startNewChunk, 
                   base_path = base_path, 
                   is_same_day = same_day, 
                   timelapse_factor = timelapse_factor,
                   file_ext = file_ext, 
                   fourcc = fourcc,
                   fps = fps,
                   frameSize = frameSize,
                   isColor = isColor)

# .....................................................................................................................
    
def getFPS_background(video_source, fps_pipe, stop_event):
    
    # Some feedback
    print("\nCalculating FPS in the background...")
    
    # Get internal videoObj ref
    temp_video_obj, _, _ = setupVideoCapture(video_source, verbose=False)
    
    # Set up run timing
    fps_minRunTime = dt.timedelta(seconds=10)
    fps_startTime = dt.datetime.now()
    fps_endTime = fps_startTime + fps_minRunTime
    
    # Set up FPS calculation variables
    fps_blockDelta = dt.timedelta(seconds=2)
    fps_blockTime = fps_startTime + fps_blockDelta
    fps_frameCount = 0
    fps_countList = []
    
    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
    
    # Loop through frames and check frame timing
    while temp_video_obj.isOpened():
        
        # Get frames
        (receivedFrame, inFrame) = temp_video_obj.read()
        if not receivedFrame: 
            print("")
            print("Error getting FPS! Lost connection...")
            print("")
            videoObj.release()
            raise IOError
                
        # Get current time and frame count
        fps_currentTime = dt.datetime.now()
        fps_frameCount += 1
        
        # Record frame counts for each block of time
        if fps_currentTime >= fps_blockTime:
            
            # Record the number of frames since the last frame block
            fps_countList.append(fps_frameCount)
            fps_frameCount = 0
            
            # Truncate the fps list if needed, very inefficiently
            if len(fps_countList) > 50:
                fps_countList = fps_countList[:50]
            
            # Update block timing
            fps_blockTime = fps_currentTime + fps_blockDelta
        
        
        # Stop looping when an event signal is sent, as long as we've run past the target end time
        if stop_event.is_set():
            if fps_currentTime > fps_endTime: 
                break
        
    #  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .  .
        
    # Calculate the median FPS and send it back to the main process
    temp_video_obj.release()
    medianFrameCount = median(fps_countList)
    medianFPS = medianFrameCount/fps_blockDelta.total_seconds()
    fps_pipe.send(medianFPS)

# .....................................................................................................................

def setGlobalStart(threshold_seconds=15):
    
    # Get the current time 
    time_now = dt.datetime.now()
    next_minute = dt.datetime(time_now.year, time_now.month, time_now.day, time_now.hour, 1 + time_now.minute)
    
    # Update the start time if the 'next minute' is too soon by adding 1 more minute
    time_delta = next_minute - time_now
    too_soon = (time_delta < dt.timedelta(seconds=threshold_seconds))
    return next_minute if not too_soon else next_minute + dt.timedelta(minutes=1)

# .....................................................................................................................
    
# .....................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Initialize variables

# Recording parameters
recording_settings = {"file_ext": ".avi",
                      "fcc_code": "X264",
                      "fps": None,
                      "frameSize": None,
                      "isColor": True}

# Text aesthetics (for timestamps)
text_padding = 5
bgThick, fgThick = 2, 1
bgColor, fgColor = (0, 0, 0), (255, 255, 255)
timestamp_text = {"org": (0, 0),
                  "fontFace": cv2.FONT_HERSHEY_SIMPLEX,
                  "fontScale": 0.4,
                  "lineType": cv2.LINE_AA}



# ---------------------------------------------------------------------------------------------------------------------
#%% Get video source


'''
# Ask user for input type
promptString = "\n".join(["Select the desired input type by entering the corresponding number.",
                          "  1 - RTSP (default)",
                          "  2 - Webcam",
                          "  3 - Video file",
                          "  4 - Last used",
                          ""])
userResponse = input(promptString)
inputSelect = numericalUserInput(user_input = userResponse, 
                                 return_type = int, 
                                 default_value = 1)

'''

# Check if a history file exists
historyFilePath = os.path.join("local/config/history.json")
historyFile = loadHistoryFile(historyFilePath, searchFor="videoSource")


# Use the history source if the user accepts it. Otherwise, prompt with an RTSP input
if historyFile is not None:
    videoSource = historyFile["videoSource"]
else:
    
    # Try to get user input for reading an RTSP stream
    try:
        rtsp_settings = rtspFromCommandLine(errorOut=True)
        pass
    except:
        print("")
        print("Cancelled RTSP input!")
        hardQuit()

    # Build RTSP string out of user input settings
    videoSource, blockIP = rtspString(**rtsp_settings)

# Figure out the input type. Should be either video, rtsp or webcam (can't use images!!)
sourceType = typeOfSource(videoSource)
if sourceType["image"]:
    print("")
    print("Cannot record image files!")
    hardQuit()
    
    
# Save history for re-use
historyDict = {"videoSource": videoSource}
saveHistoryFile(historyFilePath, historyDict)


# ---------------------------------------------------------------------------------------------------------------------
#%% Check video info

try:
    videoObj, vidWH, vidFPS = setupVideoCapture(videoSource)
except IOError:
    print("")
    print("Couldn't open video source! Tried:")
    print(videoSource)
    hardQuit()
    
# Close the video reference for now, so the background FPS process can grab frames
videoObj.release()

    
# ---------------------------------------------------------------------------------------------------------------------
#%% Figure out the recording frame rate

# Allocate space for keeping track of the video FPS
realFPS = None
fps_proc = None

# Use a background process to estimate the video FPS (while the user enters CLI inputs) if using rtsp/webcam
if not sourceType["video"]:
    
    # Start a background process to measure the average FPS of the video stream, while the user enters parameters
    fps_stopEvent = Event()
    fps_stopEvent.clear()
    fps_recvPipe, fps_sendPipe = Pipe(duplex=False)
    
    # Begin background process!
    fps_proc = Process(target = getFPS_background, 
                       args = (videoSource, fps_sendPipe, fps_stopEvent))
    fps_proc.start()

else:    
    # For video files, just use the video FPS
    realFPS = vidFPS


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up video scaling
    
# Ask user for video size scaling
promptString = "\n".join(["Enter video down-scaling factor.",
                          "(Stream dimensions: {} x {}):".format(*vidWH),
                          ""])
userResponse = input(promptString).strip()

# If the user skips an input, just use a scale of 1, otherwise try to interpret input as a float
convError, videoScale = numericalUserInput(userResponse, float, 1.0)    
if convError:
    print("")
    print("Could not convert input", userResponse, "to a float!")
    print("Video dimensions will remain unscaled!")

# Set up video scaling
scaledWH = (int(vidWH[0]*(1/videoScale)), int(vidWH[1]*(1/videoScale)))


# ---------------------------------------------------------------------------------------------------------------------
#%% Get timelapse factor

# Ask user for timelapse factor
promptString = "\n".join(["Enter timelapse factor.", 
                          "(default 1):",
                          ""])
userResponse = input(promptString).strip()

# If the user skips the input, use a factor of 1 otherwise try to interpret the input as an integer
convError, tlFactor = numericalUserInput(userResponse, int, 1)
if convError:
    print("")
    print("Could not convert input", userResponse, "to an integer!")
    print("Timelapse factor will be set to 1!")
    
    
# ---------------------------------------------------------------------------------------------------------------------
#%% Configure timestamps

# Ask user for the location of the timestamp
promptString = "\n".join(["Enter timestamp location.",
                          "tl: top-left", 
                          "tr: top-right",
                          "bl: bottom-left",
                          "br: bottom-right",
                          "(leave empty to disable timestamps):",
                          ""])
userResponse = input(promptString).strip().lower()


# Check if the user entered a valid location to enable timestamps
timestampEnabled = (userResponse in ["tl", "tr", "bl", "br"])
if timestampEnabled:
    
    # Figure out which edges the text to aligned to
    left_just = (userResponse in ["tl", "bl"])
    top_just = (userResponse in ["tl", "tr"])
    
    # Find the size of the timestamp text
    xsize, ysize = cv2.getTextSize("00:00:00", 
                                   timestamp_text["fontFace"],
                                   timestamp_text["fontScale"],
                                   thickness=fgThick)[0]
    # Get text position
    xloc = text_padding if left_just else (scaledWH[0] - xsize - text_padding - 1)
    yloc = ysize + text_padding if top_just else (scaledWH[1] - text_padding - 1)
    
    # Update text positioning
    timestamp_text = {**timestamp_text, **{"org": (xloc, yloc)}}


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up recording
   
# .....................................................................................................................
# Create base recording path

output_directory = os.path.join(os.path.expanduser("~/Desktop"), "VideoChunking")


# .....................................................................................................................
# Ask user for total run time

promptString = "\n".join(["Enter the total recording time, in minutes.",
                          "(default 2):",
                          ""])
userResponse = input(promptString)
convError, total_recording_time = numericalUserInput(userResponse, float, default_value = 2)
if convError:
    print("")
    print("Could not convert input", userResponse, "to a number!")
    print("Total recording time will be 2 minutes!")
    
    
# .....................................................................................................................
# Ask user for chunk time
    
promptString = "\n".join(["Enter video chunk size, in minutes.",
                          "(default 1):",
                          ""])
userResponse = input(promptString)
convError, chunk_size = numericalUserInput(userResponse, float, default_value = 1)
if convError:
    print("")
    print("Could not convert input", userResponse, "to a number!")
    print("Chunk size will be 2 minutes!")


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up video capture


# End background tasks before beginning the main loop
if fps_proc:
    
    # Send the stop command and wait for data in the receiving pipe
    fps_stopEvent.set()    
    fps_pipe_full = fps_recvPipe.poll(5)
    
    # Record the background FPS if it was captured, otherwise used the read value
    if fps_pipe_full:
        realFPS = fps_recvPipe.recv()
        print("")
        print("Background FPS calculation success!")
    else:
        realFPS = vidFPS
        print("")
        print("Background FPS calculation failed!")    
    print("Using FPS value:", "{:.1f}".format(realFPS))
        
    # Wait a bit for the process to finish
    fps_proc.join(5)
    fps_proc.terminate() # Just in case, doesn't cause errors if the process has already joined
    
    # Close pipes
    fps_recvPipe.close()
    fps_sendPipe.close()


# Reset video object
videoObj, _, _ = setupVideoCapture(videoSource, verbose=False)


# ---------------------------------------------------------------------------------------------------------------------
#%% Set up start/end time    

# Set global start and end times for stream capturing
global_start_time = setGlobalStart()#dt.datetime.now() + dt.timedelta(seconds=5)
global_end_time = global_start_time + dt.timedelta(minutes=total_recording_time)

# Update recording settings
recording_settings = {**recording_settings, "fps": realFPS, "frameSize": scaledWH}

# Create chunk making function
createNewChunk = chunkWriter(output_directory, global_start_time, global_end_time, tlFactor, **recording_settings)


# ---------------------------------------------------------------------------------------------------------------------
#%% Video loop

# Create a display window
dispWindow = SimpleWindow("Display")

# Set up video timing
videoTimer = Timer(source_type = sourceType, 
                   vidFPS = realFPS,
                   global_start = global_start_time, 
                   global_finish = global_end_time,
                   chunk_size_mins = chunk_size)

# Default to not record!
videoOut = None

# Some feedback before streaming
print("")
print("Capturing video source:")
print(videoSource)
print("")
print("Saving to:")
print(output_directory)
print("")
print("Start recording: ", global_start_time.strftime("%Y-%m-%d %H:%M:%S"))
print("Finish recording:", global_end_time.strftime("%Y-%m-%d %H:%M:%S"))

try:
    while True:
        
        # .............................................................................................................
        # Grab frames
        
        (receivedFrame, inFrame) = videoObj.read()
        
        if not receivedFrame:
            print("")
            print("No frames!")
            break
        
        # Get the frame timing
        videoTimer.getTime()
        
        # .............................................................................................................
        # Global control of video, depending on start and end times
        
        # Skip frames before target start time
        if videoTimer.waitToStart():
            if sourceType["video"]: videoObj.set(1, 0) # Bit of a hack to reset video files
            continue
        
        if videoTimer.isFinished():
            print("")
            print("Time's up!")
            break
        
        # .............................................................................................................
        # Update recording chunks if needed
        
        if videoTimer.needNewChunk():
            
            # Stop the current recording
            if videoOut:
                videoOut.release()
            
            # Get chunk timing
            start_end_times = videoTimer.newChunkInfo()
            
            # Start a new recording
            videoOut, fileReference = createNewChunk(*start_end_times)
        
        
        # .............................................................................................................
        # Process the incoming frames
        
        # Scale the incoming frames
        scaledFrame = cv2.resize(inFrame, dsize=scaledWH)
        
        
        # .............................................................................................................
        # Draw time of recording (if enabled)
        
        if timestampEnabled:
            
            # Figure out the time text to draw
            timeString = videoTimer.timestamp().strftime("%H:%M:%S")
            
            # Draw timestamp on to the frame
            cv2.putText(scaledFrame, timeString, color=bgColor, thickness=bgThick, **timestamp_text)
            cv2.putText(scaledFrame, timeString, color=fgColor, thickness=fgThick, **timestamp_text)
            
        
        # .............................................................................................................
        # Display frames
        
        winExists = dispWindow.imshow(scaledFrame)
        if not winExists: break
    
        reqBreak, keyPress = breakByKeypress(1)
        if reqBreak: break
    
    
        # .............................................................................................................
        # Record frames
        
        if videoTimer.frameCount() % tlFactor == 0:
            videoOut.write(scaledFrame)
            

# Cleaning close video resources if we get a keyboard interrupt
except KeyboardInterrupt:
    print("")
    print("Keyboard interrupt!")
    
# Comment out to make debugging easier!
except Exception as err:
    print("")
    print("Unknown error:")
    print(err)


# ---------------------------------------------------------------------------------------------------------------------
#%% Clean up

# Clean up resources
videoObj.release()
cv2.destroyAllWindows()

# Stop recording
videoOut.release()

    
# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap











