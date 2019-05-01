import numpy as np
import cv2
import time
import sys
import glob
import pickle

# Globally define our CV2 Information that will be used to add every sample
protoFile = "pose_models/pose/mpi/pose_deploy_linevec_faster_4_stages.prototxt"
weightsFile = "pose_models/pose/mpi/pose_iter_160000.caffemodel"
nPoints = 15
POSE_PAIRS = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 14], [14, 8], [8, 9], [9, 10], [14, 11],
                  [11, 12], [12, 13]]

inWidth = 368
inHeight = 368
threshold = 0.1


class UserModel(object):

    def __init__(self):
        # Number of videos sampled for this user
        self.num_samples = 0

        # Features specific to this user's model, initially 0 or None while no
        # videos have been sampled
        self.avg_ascent_time = 0
        self.avg_shot_length = 0

        self.avg_base_angle = None
        self.avg_release_angle = None

    def get_vector(self):
        '''
            Returns this user's model statistics as a numpy array, for
            comparison with any other such vector.
        '''
        return np.array([self.avg_ascent_time, self.avg_shot_length, 
                         self.avg_base_angle, self.avg_release_angle])


    def add_sample(self, jpg, thresh=.08):
        '''
            Given a called UserModel object, and .jpg video file, does
            template matching to add the statistics for this model to our
            user's average
        '''
        # Read the network into Memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        # Read in our averaged templates to be matched
        base_template = 'output_jpgs/average_base.jpg' 
        with open('average_joints/average_joints_base.pickle', 'rb') as handle:
            base_joints = pickle.load(handle)

        release_template = 'output_jpgs/average_release.jpg'
        with open('average_joints/average_joints_release.pickle', 'rb') as handle:
            release_joints = pickle.load(handle)

        _, base = cv2.VideoCapture(base_template).read()
        _, release = cv2.VideoCapture(release_template).read()
        h, w = base.shape[0], base.shape[1]

        # Read in the User's sample video
        cap = cv2.VideoCapture(jpg)
        hasFrame, frame = cap.read()

        vid_writer = cv2.VideoWriter('./test.jpg',
                         cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                         10, (w, h))

        while True:
            if not hasFrame:
                break
            # Draw the skeleton over the most recent frame
            skeleton_frame, joints = self.draw_skeleton(frame, net, h, w)

            # At this point, we have a Blank Skeleton to Template Match with.
            # Compare it with our averaged templates
            # Max of training that's still base is 790: 800 threshold?
            print('comparison with base template: ')
            print(self.compare_joints(joints, base_joints))

            print('comparison with release template: ')
            print(self.compare_joints(joints, release_joints))
            print()
            
            vid_writer.write(skeleton_frame)

            hasFrame, frame = cap.read()

        # Update user metadata, given this sample
        self.num_samples += 1
        vid_writer.release()
        return

    def compare_joints(self, j1, j2):
        '''
            Takes two dictionaries of joints and returns the sum of pairwise
            Euclidean distances between each joint
        '''

        dist = 0
        for joint in j1:
            pointA = np.array(j1[joint])
            pointB = np.array(j2[joint])
            dist += np.linalg.norm(pointA - pointB)

        return dist

    def compare_images(self, img1, img2):
        count = 1
        for r in range(img1.shape[0]):
            for c in range(img1.shape[1]):
                if np.any(img1[r, c]) and np.any(img2[r, c]):
                    count += 1

        return count / (img1.shape[0] * img1.shape[1])

    def draw_skeleton(self, frame, net, h, w):
        '''
            Helper method that takes a frame and the net object, and returns a
            blank frame with the corresponding skeleton template over it, as
            well as a list of that skeleton's joints, as tuples.
        '''
        t = time.time()

        blank_frame = np.zeros((h, w, 3), np.uint8)
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
    
        # converted to a input blob (like Caffe) so that it can be fed to the network
        inpBlob = cv2.dnn.blobFromImage(frame, 1.0 / 255, (inWidth, inHeight),
                                    (0, 0, 0), swapRB=False, crop=False)
        net.setInput(inpBlob)
     
        # makes a forward pass through the network, i.e. making a prediction
        output = net.forward() # 4D matrix, 1: image ID, 2: index of a keypoint, 3: height, 4: width of output map
        H = output.shape[2]
        W = output.shape[3]
    
        # Empty list to store the detected keypoints
        points = []
        for i in range(nPoints):
            # confidence map of corresponding body's part.
            probMap = output[0, i, :, :]

            # Find global maxima of the probMap.
            minVal, prob, minLoc, point = cv2.minMaxLoc(probMap)

            # Scale the point to fit on the original image
            x = (frameWidth * point[0]) / W
            y = (frameHeight * point[1]) / H

            if prob > threshold:
                cv2.circle(frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Draw Skeleton
        joints = []
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                joints.append((points[partA], points[partB]))

        # Normalize the center point to be half the width of the image,
        # and one-third the height.
        center_point = tuple(joints[1][0])

        x_diff = (blank_frame.shape[1] // 2) - center_point[0]
        y_diff = (blank_frame.shape[0] // 4) - center_point[1]

        average_joints = {i: point for i in range(len(points))}
        for joint, pair in zip(joints, POSE_PAIRS):
            pointA = [int(val) for val in joint[0]]
            pointB = [int(val) for val in joint[1]]

            pointA[0] = int(pointA[0] + x_diff)
            pointA[1] = int(pointA[1] + y_diff)
            pointB[0] = int(pointB[0] + x_diff)
            pointB[1] = int(pointB[1] + y_diff)

            pointA = tuple(pointA)
            pointB = tuple(pointB)

            cv2.line(blank_frame, pointA, pointB, (0, 255, 255), 3, lineType=cv2.LINE_AA)
            cv2.circle(blank_frame, pointA, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
            cv2.circle(blank_frame, pointB, 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)


            # Add the points from this line to the joint dictionary, if they haven't
            # already been added
            first = pair[0]
            second = pair[1]
            average_joints[first] = pointA
            average_joints[second] = pointB

        return blank_frame, average_joints

for f in glob.iglob('../basketball_photos/base/*'):
    print(f)
    u = UserModel()
    u.add_sample(f)
