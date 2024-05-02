import numpy as np
import cv2
import matplotlib.pyplot as plt
from libs.landmark import face_landmark

# paths
source_path = 'dataset/val/n000029/0080_01.jpg'
target_path = 'image_bank/n000004.npy'
out_path = "2d_warp.jpg"

# load image
source_img = cv2.imread(source_path)

# corresponding points (in 2D)
source_points = face_landmark(source_img)[:,0:2]
destination_points = np.load(target_path)[:,0:2]

# calculate transformation matrix
M, mask = cv2.findHomography(source_points.astype(np.float32), destination_points.astype(np.float32))

# warp image
warped_img = cv2.warpPerspective(source_img, M, (500, 500)) # adjust width and height of output

# save result
cv2.imwrite(out_path, warped_img)
print("result saved")