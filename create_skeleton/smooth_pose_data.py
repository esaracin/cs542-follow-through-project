import pandas as pd
import numpy as np
import cv2, os
import scipy
from scipy import signal
import csv

circle_color, line_color = (0,0,255), (255, 0, 0)
window_length, polyorder = 13, 6
sd = "2kvids/allen1"
input_source = "2kvids/allen1.mp4"

# Get pose data - data is generated by OpenPose
df = pd.read_csv('2kvids/allen1.csv')

cap = cv2.VideoCapture(input_source)
hw = 720
hasFrame, frame = cap.read()
out = cv2.VideoWriter('allen1_Smooth.avi',
                      cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame.shape[1], frame.shape[0]))

# There are 15 points in the skeleton
pairs = [[0,1], # head
         [1,2],[1,5], # shoulders
         [2,3],[3,4],[5,6],[6,7], # arms
         [1,14],[14,11],[14,8], # hips
         [8,9],[9,10],[11,12],[12,13]] # legs

# Smooth it out
for i in range(30): df[str(i)] = signal.savgol_filter(df[str(i)], window_length, polyorder)

frame_number = 0
while True:
    print(frame_number)
    ret, img = cap.read()
    if not ret: break
    #img = np.zeros_like(img)        
    values = np.array(df.values[frame_number], int)
    
    points, lateral_offset = [], 18
    points = list(zip(values[:15]+lateral_offset, values[15:]))

    cc = 0
    for point in points:
        cc += 90
        xy = tuple(np.array([point[0], point[1]], int))
        cv2.circle(img, xy, 5, (cc,cc,cc), 5)

    # Draw Skeleton
    for pair in pairs:
        partA = pair[0]
        partB = pair[1]
        cv2.line(img, points[partA], points[partB], line_color, 3, lineType=cv2.LINE_AA)
    
    cv2.imshow('Output-Skeleton', img)
    k = cv2.waitKey(1000)
    if k == 27: break
    out.write(img)
    frame_number+=1
cv2.destroyAllWindows()