# 
# Code Repository for our CS542 Final Project: Follow Through
#
# Collaborators:
# Eli Saracino (esaracin@bu.edu)
# Sameer C.
# Junior S.
#

To run PoseAnalysis from basic linux system:

unset DISPLAY XAUTHORITY
xvbf-run python PoseAnalysis.py input_mp4s/filename.mp4


ToDo:

Manipulate parameters for OpenPose code to make skeletons smoother and more accurate:
- smoothening
- tweaking parameters

Create average template skeletons for each point in shot:
- filtering through the image db for base/begin ascent/release images (roughly 45 degrees)
	- We want base of shot
	- Beginning of ascent
	- Release of ball
- Create an average skeleton from them for use later

Taking an input video w/ skeleton and creating a tuple:
-computing length, release angles, etc. based on template matching with average skeleton from previous step

Apply previous algorithm to each video (from 2k, user-input videos (us lol))

Compare models using regression (user v pro, user v chosen pro)

lol a paper
