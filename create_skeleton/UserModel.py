import numpy as np
import cv2
import time
import sys

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


    def add_sample(self, jpg):
        '''
            Given a called UserModel object, and .jpg video file, does
            template matching to add the statistics for this model to our
            user's average
        '''
        # Read the network into Memory
        net = cv2.dnn.readNetFromCaffe(protoFile, weightsFile)

        base_template = 'output_jpgs/average_base.jpg' 
        release_template = 'output_jpgs/average_release.jpg'

        cap = cv2.VideoCapture(jpg)
        hasFrame, frame = cap.read()

        vid_writer = cv2.VideoWriter('./test.jpg',
                             cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 
                             10, (frame.shape[1], frame.shape[0]))
        while True:
            if not hasFrame:
                break

            skeleton_frame = self.draw_skeleton(frame, net)
            vid_writer.write(skeleton_frame)

            # At this point, we have a Blank Skeleton to Template Match with.
            # Center the Skeleton, and compare it with our average templates!
            hasFrame, frame = cap.read()
        return

    def draw_skeleton(self, frame, net):
        t = time.time()


            #frameCopy = np.copy(frame)
        blank_frame = np.zeros((frame.shape[0], frame.shape[1], 3), np.uint8)

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
                cv2.circle(blank_frame, (int(x), int(y)), 8, (0, 255, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.putText(blank_frame, "{}".format(i), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                            lineType=cv2.LINE_AA)

                # Add the point to the list if the probability is greater than the threshold
                points.append((int(x), int(y)))
            else:
                points.append(None)

        # Draw Skeleton
        for pair in POSE_PAIRS:
            partA = pair[0]
            partB = pair[1]

            if points[partA] and points[partB]:
                cv2.line(blank_frame, points[partA], points[partB], (0, 255, 255), 3, lineType=cv2.LINE_AA)
                cv2.circle(blank_frame, points[partA], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)
                cv2.circle(blank_frame, points[partB], 8, (0, 0, 255), thickness=-1, lineType=cv2.FILLED)

                cv2.imshow('Output-Skeleton', frame)


        return blank_frame

u = UserModel()
u.add_sample('../basketball_photos/base/001.jpg')
