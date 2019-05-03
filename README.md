# 
# Code Repository for our CS542 Final Project: Follow Through
#
# Collaborators:
# Eli Saracino (esaracin@bu.edu)
# Sameer C.
# Junior S.
#

The code in these directories pertains to the solution to our Final Project, Follow Through.
The code can be divided into two main parts, each part being relevant to one of our 
subtasks, and their descriptions will be, similarly, divided as such.

The first part is the task of improving the skeleton's drawn by OpenPose to follow the motions of the subject
more smoothly, to better facilitate the use of this technology as the backbone of our second task...

The second task involves using OpenPose to create a system for methodically turning a basketball player's jumpshot
into a vector that can be used to build a model of a given player's performance, which can allow them to improve.


Task One: Skeleton Smoothing:
Pertinent Files:

Task Two: Using OpenPose to build an analytic approach to improving the performance of a basketball player:
Pertinent Files:
*IMPORTANT NOTE1*:
	Most of the code that uses OpenPose to create the skeletons (which is the majority of the code in this part)
	*requires* the weight and structure files of the MPII-trained Neural Net to
    	be located locally in "pose_models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt" (for the protofile) 
    	and "pose_models/pose/mpi/pose_iter_160000.caffemodel" (for the weight file). For your convenience, we have 
	included the script getModels.sh, which consists of several wget commands to download these (rather heavy)
	files for you.
*END IMPORTANT NOTE1*:

*IMPORTANT NOTE2*: 
	if you run into the following error:

		"cannot connect to X server"

	while running any of the following python scripts, you must run the following commands in your terminal:

		unset DISPLAY XAUTHORITY
		xvfb-run python [script.py that gave error] [any arguments of script.py]
	
	note that this may require the installation of xvfb-run, which is easily installable with apt-get!
*END IMPORTANT NOTE2*


PoseAnalysisPhoto.py:
	This python script takes as input a single .jpg photo and applies the OpenPose skeleton technology to 
	draw a skeleton of the subject's joints over the original image. It's output is a written .jpg photo
	that is a copy of the input image, but in which the skeleton has been applied. Viable input images are 
	supplied in basketball_photos/base/ and basketball_photos/release/. Run as follows:

		xvfb-run python PoseAnalysisPhoto.py ../basketball_photos/release/release001.png
	
	and the resulting photo will be saved to output_jpgs/release001.jpg.

AveragePoseAnalysis.py:
	This script takes the previous task a step further, and creates an average skeleton from *all* input photos
	in the specified input directory. Run as follows:
	
		xvfb-run python AveragePostAnalysis.py ../basketball_photos/release/
	
	and an average template skeleton will be written to output_jpgs/average_release.jpg. Also, a .pickle file 
	with the set of joints corresponding to that average skeleton will be written to average_joints/average_joints_release.pickle





To Average photos from an input directory into a single, centered, skeleton:
xvfb-run python AveragePoseAnalysis.py ../basketball_photos/base/ 


ToDo:

Manipulate parameters for OpenPose code to make skeletons smoother and more accurate:
- smoothening
- tweaking parameters

Create average template skeletons for each point in shot:
- filtering through the image db for base/begin ascent/release images (roughly 45 degrees)
	- We want base of shot
	- Beginning of ascent
	- Release of ball
- Create an average skeleton from them for use later: edit single frame template code to average the templates into one? Or simply use one of the output images..
- First need to draw frames that are just the skeletons themselves on a blank background (np.zeros for the blank background frames): after computing the skeleton, draw it on a blank frame

Taking an input video w/ skeleton and creating a tuple:
-computing length, release angles, etc. based on template matching with average skeleton from previous step

Apply previous algorithm to each video (from 2k, user-input videos (us lol))

Compare models using regression (user v pro, user v chosen pro)

lol a paper
