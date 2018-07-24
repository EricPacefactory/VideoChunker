# VideoChunker
Script used for recording an RTSP stream (or video file) into multiple smaller files. The files may be stitched together if needed, using a companion script called VideoStitching.
The script is controlled entirely through a command line interface.

Tested on:

Tested on:
- Ubuntu 16.04
- Python 3.5.2

Requires:
- OpenCV (3.3.1+)

OpenCV can be installed from pip, but has only been tested using a manual installation.
The pip installation seems to have unreliable video recording!

Note, this file automatically saves files to the desktop of the computer running it!