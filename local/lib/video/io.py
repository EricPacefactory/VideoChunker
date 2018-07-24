#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:01:29 2018

@author: eo
"""

import os
import cv2
import datetime as dt


# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes



# ---------------------------------------------------------------------------------------------------------------------
#%% Define video functions

# .....................................................................................................................

def scaleToTarget(vidWH, targetWH, fitInTarget=True):
    
    
    '''
    outputs:
        - scaledWH: tuple containing dimensions scaled to best match the target WH.
                    Does not change the aspect ratio
        
    inputs:
        - vidWH: size of original video frame to be resized
        - targetWH: desired size of frame after scaling
        - fitInTarget (optional): If True, the output dimensions will be equal to or less than targetWH
                                If False, the output dimensions will be scaled as much as possible, without
                                BOTH width/height being larger than the target dimensions
    '''
    
    # Get the scaling factor 
    scaleFractW = targetWH[0] / vidWH[0]
    scaleFractH = targetWH[1] / vidWH[1] 
    
    # Set frame scaling based on min or max scale fractions
    dimScale = min(scaleFractW, scaleFractH) if fitInTarget else max(scaleFractW, scaleFractH)    
    scaledWH = (int(dimScale*vidWH[0]), int(dimScale*vidWH[1]))
    
    return scaledWH

# .....................................................................................................................

def setupVideoCapture(source, verbose=True):
    
    # OpenCV constants
    vc_width = 3
    vc_height = 4
    vc_fps = 5
    vc_fourcc = 6
    vc_framecount = 7 
        
    # Set up video capture object
    videoObj = cv2.VideoCapture(source)
    if not videoObj.isOpened():
        print("")
        print("Couldn't open video object. Tried:")
        print(source)
        print("Closing...")
        print("")
        raise IOError
        
    # Check for webcam inputs
    isWebcam = (type(source) is int)   

    # Try to set the video name based on the input source
    if isWebcam:
        videoName = "Webcam-{}".format(source)
    else:    
        if "rtsp" in source.lower():
            # Try to grab the IP numbers out of the RTSP string
            splitSource = source.replace("@", ".").replace(":", ".").replace("/", ".").split(".")
            numsInSource = [int(eachEntry) for eachEntry in splitSource if eachEntry.isdigit()]
            only8bits = [str(eachNum) for eachNum in numsInSource if eachNum < 256]
            guessIPString = ".".join(only8bits[:4])
            videoName = " - ".join(["RTSP", guessIPString])
        else:
            videoName = os.path.basename(source)
    
    # Get video info
    vidWidth = int(videoObj.get(vc_width))
    vidHeight = int(videoObj.get(vc_height))
    vidFPS = videoObj.get(vc_fps)
    
    # Check that the FPS is valid (some RTSP reads are wrong)
    if not (5 < vidFPS < 61):
        print("")
        print("Error with FPS. Read as:", vidFPS)
        print("Expecting value between 5-61")
        print("Assuming: 30")
        vidFPS = 30
    
    # Try to grab video info that may not be available on RTSP
    try:
        vidFCC = int(videoObj.get(vc_fourcc)).to_bytes(4, 'little').decode()
        totalFrames = max(-1, int(videoObj.get(vc_framecount)))
        totalRunTime = -1 if totalFrames < 0 else ((totalFrames - 1)/vidFPS)
    except Exception as e:
        print("")
        print("Couldn't get video info... RTSP stream?")
        vidFCC = "Unknown"
        totalFrames = -1
        totalRunTime = -1
    vidFCC = "Unknown" if vidFCC == "\x00\x00\x00\x00" else vidFCC
    
    # Print out video information
    if verbose:
        currentDate = dt.datetime.now()
        print("")
        print(currentDate.strftime("%Y-%m-%d"))
        print("Video:", videoName)
        print("Dimensions:", vidWidth, "x", vidHeight, "@", vidFPS, "FPS")
        print("Codec:", vidFCC)
        print("Total Frames:", totalFrames)
        
        # Print out different run time units, depending on the video length
        if totalRunTime > 3600:
            print("Run time:", "{:.1f} hours".format(totalRunTime/3600))
        elif totalRunTime > 60:
            print("Run time:", "{:.1f} mins".format(totalRunTime/60))
        else:
            print("Run time:", "{:.1f} seconds".format(totalRunTime))
    
    # Some final formating
    vidWH = (vidWidth, vidHeight)
    
    return videoObj, vidWH, vidFPS

# .....................................................................................................................
    
def setupVideoRecording(recPath, recName, recWH, recFPS=30, recFCC="X264", recEnabled=True):
    
    videoOut = None
    if recEnabled:
        # OpenCV property constant
        outputColorImage = True
        
        # Check if file name has extension, if not, use .avi
        recFilename, recFileExt = os.path.splitext(recName)
        if recFileExt == "":
            recFileExt = ".avi"
        recName = "".join([recFilename, recFileExt])
        
        # Define output variables
        fourcc = cv2.VideoWriter_fourcc(*recFCC)
        videoOutSource = os.path.join(recPath, recName)
        
        # Create video writer based on input parameters
        videoOut = cv2.VideoWriter(videoOutSource, 
                                   fourcc, 
                                   recFPS,
                                   recWH, 
                                   outputColorImage)
        
        # Feedback
        print("")
        print("Recording enabled! Saving as:")
        print(videoOutSource)
    else:
        print("")
        print("Recording not enabled!")
    
    return videoOut

# .....................................................................................................................
    

# .....................................................................................................................
    
def setupVideoRecordingV2(recPath, recName, recWH, recFPS=30, recTimelapse=1, recFCC="X264", recEnabled=True):
    
    raise NotImplementedError
    
    videoOut = None
    if recEnabled:
        
        # Check if file name has extension, if not, use .avi
        recFilename, recFileExt = os.path.splitext(recName)
        if recFileExt == "":
            recFileExt = ".avi"
        recName = "".join([recFilename, recFileExt])
        
        # Define output variables
        fourcc = cv2.VideoWriter_fourcc(*recFCC)
        videoOutSource = os.path.join(recPath, recName)
        
        # Create video writer based on input parameters
        outputColorImage = True     # OpenCV property constant, named here for clarity
        videoOut = cv2.VideoWriter(videoOutSource, 
                                   fourcc, 
                                   recFPS,
                                   recWH, 
                                   outputColorImage)
        
        # Feedback
        print("")
        print("Recording enabled! Saving as:")
        print(videoOutSource)
        
        # Create helper class that can reshape incoming frames for the video recorder
        class VideoRecorder:
            
            def __init__(self, videoWriter, frameWH, timelapse=1):
                
                self._frameCount = 0
                self._shapeHW = (frameWH[1], frameWH[0])
                self._sizeWH = (frameWH[0], frameWH[1])
                self._timelapse = timelapse
                self._videoOut = videoWriter
                self._gaveWarning = False
                
            def write(self, inFrame):
                
                if self._frameCount % self._timelapse == 0:
                    
                    needReshape = False
                    inShapeHW = inFrame.shape[:2]
                    inChannels = inFrame[2] if len(inFrame.shape) > 2 else 1
                    
                    needReshape = (inShapeHW != self._shapeHW)
                    needBGR = (inChannels != 3)
                    
                    recFrame = inFrame.copy if (needReshape or needBGR) else inFrame
                    
                    
                    # Resize the image if needed
                    if needReshape:
                        recFrame = cv2.resize(recFrame, dsize=self._sizeWH)
                        
                    # Convert a grayscale image into a 3 channel image if needed
                    if needBGR:
                        recFrame = cv2.cvtColor(recFrame, cv2.COLOR_GRAY2BGR)
                        
                    # Record video frame
                    self._videoOut.write(recFrame)
                    
                    # Give warning, if needed
                    if needReshape or needBGR:                            
                        if not self._gaveWarning:
                            # Feedback
                            print("")
                            print("Video recording warning:")
                            print("Input frame dimensions do not match recording dimensions!")
                            print("Got dimensions:")
                            print(inShapeHW, inChannels)
                            print("Expected:")
                            print(self._shapeWH, 3)
                            self._gaveWarning = True

                # Increment frame count, regardless of whether a frame was recorded or not
                self._frameCount += 1
            
            def release(self):                    
                self._videoOut.release()
            
            
        
    else:
        print("")
        print("Recording not enabled!")
    
        # Create empty video recorder with dummy functions so that recording functions can be called, but do nothing
        class VideoRecorder:
            
            def __init__(self,  *args, **kwargs):
                return
            
            def write(self, *args, **kwargs):
                
                return
            
            def release(self, *args, **kwargs):
                return
    
    
    
    return VideoRecorder(videoOut, recWH, recTimelapse)

# .....................................................................................................................

def typeOfSource(video_source, output_as_string=False):
    
    '''
    output:
        - Either a string indicating the source type or a dictionary with keys representing
            all possible source types and boolean values to indicate the input source type
            
        Possible source types:
            "rtsp"
            "image"
            "video"
            "webcam"
        
    input:
        - video_source: reference to the video source (matching OpenCV cv2.VideoCapture(video_source))
        - output_as_string: if true, this function returns the source type as a string, otherwise
            the function returns a dictionary
    '''
    
    # Some useful variables
    image_types = [".jpg", ".png", ".bmp"]
    video_types = [".avi", ".mov", ".mpeg", ".mpg", ".mp4", ".m4v"]
    
    # Initialze outputs
    isImage, isRTSP, isVideo, isWebcam = [False] * 4
    
    # First check if we have a webcam input, since we can't parse this as a string
    isWebcam = (type(video_source) == int) or (type(video_source) == float)
    
    # Check the file extension (for checking if we have an image or video)
    source_extension = os.path.splitext(video_source)[1].lower() if not isWebcam else ""
    
    # Figure out what kind of input we have
    isRTSP = ("rtsp" in video_source.lower()) if not isWebcam else False
    isImage = source_extension in image_types
    isVideo = source_extension in video_types
    
    # Quick sanity check
    if sum([isImage, isVideo, isRTSP, isWebcam]) != 1:
        print("")
        print("Could not determine the input video type! Got:")
        print("Source:", video_source)
        print("  Image:", isImage)
        print("  RTSP:", isRTSP)
        print("  Video:", isVideo)
        print("  Webcam:", isWebcam)
        print("")
        raise TypeError

    # Build a dictionary of source type results
    sourceType = {"rtsp": isRTSP,
                  "image": isImage,
                  "video": isVideo,
                  "webcam": isWebcam}
    
    # Return only name of the video source type as a string if outputting as a string
    if output_as_string:
        return [eachKey for eachKey, eachValue in sourceType.items() if eachValue][0]

    # If not outputting as a string, return the source type dictionary
    return sourceType

# .....................................................................................................................

# ---------------------------------------------------------------------------------------------------------------------
#%% Define file functions

def loadBackground(backgroundSource, videoObjRef=None):
    
    # Check that the file is there before loading it in
    if os.path.isfile(backgroundSource):
        
        # Background file exists, try to load it
        try:
            bgImage = cv2.imread(backgroundSource)
            print("")
            print("Background file found:")
            print(backgroundSource)
        except Exception as error:
            print("")
            print("Error loading background image!")
            print("")
            raise error
        
    else:
        
        # Feedback about the missing background image
        print("")
        print("No background file found! Searched:")
        print(backgroundSource)
        
        # If not video object is provided, give an error. Otherwise, use it to get a background frame
        if videoObjRef is None:
            print("")
            raise FileNotFoundError

        # Try to grab a frame from the video as a background
        if videoObjRef.isOpened():
            (receivedFrame, inFrame) = videoObjRef.read()
        else:
            receivedFrame, inFrame = False, None
        
        # If the frame grab failed, we're screwed!
        if not receivedFrame:
            print("")
            print("Tried to grab a background frame from the video source, but failed!")
            print("")
            raise IOError
            
        # Use the background frame
        bgImage = inFrame
        
        # Reset the video object so we don't skip the first frame
        pos_frames = 1
        videoObjRef.set(pos_frames, 0)
    
    return bgImage

# .....................................................................................................................



# ---------------------------------------------------------------------------------------------------------------------
#%% Demo
    
if __name__ == "__main__":
    # Useful shortcut
    #closeall = cv2.destroyAllWindows()
    pass

# ---------------------------------------------------------------------------------------------------------------------
#%% Scrap
    

