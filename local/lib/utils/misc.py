#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 27 11:57:02 2018

@author: eo
"""

import os

# ---------------------------------------------------------------------------------------------------------------------
#%% Define classes




# ---------------------------------------------------------------------------------------------------------------------
#%% Define functions


# .....................................................................................................................

# Function for quitting scripts in spyder IDE without crashing the kernel
def hardQuit():
    print("")    
    if any('SPYDER' in name for name in os.environ):
        raise SystemExit()  # Hard crash to stop that menacing spyder
    else:
        quit()  # Works nicely everywhere else!

# .....................................................................................................................

# Function for warning about working within a dropbox folder
def dropboxCatcher(workingPath=None, raiseError=True):
    
    if workingPath is None:
        workingPath = os.getcwd()
    
    if "dropbox" in workingPath.lower():
        print("")
        print("***********************************************")
        print("***********************************************")
        print("")
        print("WARNING:")
        print("Working out of a Dropbox folder!")
        print("")
        print("***********************************************")
        print("***********************************************")
        if raiseError:
            raise EnvironmentError
        
# .....................................................................................................................
            
# Function for crashing if python 2 is used
def doNotUsePython2():
    from sys import version_info as python_version
    if python_version.major < 3:
        print("")
        print("This script must be called with python3")
        print("Aborting...")
        raise EnvironmentError