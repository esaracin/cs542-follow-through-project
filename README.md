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

UserModel.py:
	This code shouldn't be called directory, but contains the blueprint class for a given Follow Through user. At time of submission,
	the functionality of this class is limited, but still useful. Namely, a user can be created, and sample video's of their jumpshot
	can be added using the add_sample() method to start to build their profile. Additionally, the get_vector() method can be used
	to return the specific measurements relating to that user's stats.

model_test.py:
	This code was the first attempt to test our UserModel class. It goes through each of the videos of Professional Basketball players
	in input_mp4s/2kvids/ and creates a simple model corresponding to each user, which it saves to test_files/pro_user_dict.pickle for processing
	in later code. Note that it uses another .pickle file, pro_users_seen.pickle, so that the building of these models could be done over time, with	
	breaks inbetween. **As long as these pickle files still exist in the test_files/ directory, running model_test.py will do nothing, as it will recognize
	that all of the necessary models have already been built.** Still, in case it's necessary, you can run this file as follows:

		python model_test.py

	and listen to your machine's fan go haywire if you want to ;).

compute_pro_distances.py:
	This code provides the first test of the Follow Through technology. Reading in the .pickle files written by model_test.py, this script
	runs two tests:
	1)	For every pro model in the .pickle file, it gets the vector for each other pro, and computes the Euclidean distance between the 
		given pro and the current pro being examined. It then sorts these distances, and lists, for the given pro, the three other pro's
		who's shot-vector is most like their own.
	2)	It then builds a test user, which uses one of the held-out sample videos of Michael Jordan, and compute's *its* distance from every pro.
		Again, it lists the top three closest pro models (notice that the closest pro to the held out Michael Jordan video is, thankfully, Michael Jordan)
		for this sample user.

	Run it from the command line as follows:

		python compute_pro_distances.py
	
	and view the printed output. **Note that task 2) in the above may take some time, as creating and adding a video 
	to the model for a test user is computationally intensive.**
		
	



