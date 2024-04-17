import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def face_landmark(image):
    base_options = python.BaseOptions(model_asset_path='./checkpoints/models/face_landmarker_v2_with_blendshapes.task')
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                        output_face_blendshapes=True,
                                        output_facial_transformation_matrixes=True,
                                        num_faces=1)
    detector = vision.FaceLandmarker.create_from_options(options)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
    detection_result = detector.detect(mp_image)
    
    face_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in detection_result.face_landmarks[0]])
    face_landmarks[:, 0] *= image.shape[1]
    face_landmarks[:, 1] *= image.shape[0]

    return face_landmarks

def find_H(image1, image2):
    face_landmarks1 = face_landmark(image1)
    face_landmarks2 = face_landmark(image2)
    num_points = face_landmarks1.shape[0]

    # Compute homography
    A = []
    for i in range(num_points):
        x1, y1, z1 = face_landmarks1[i]
        x2, y2, z2 = face_landmarks2[i]
        # 3d
        A.append([x1, y1, z1, 1, 0, 0, 0, 0, 0, 0, 0, 0, -x2])
        A.append([0, 0, 0, 0, x1, y1, z1, 1, 0, 0, 0, 0, -y2])
        A.append([0, 0, 0, 0, 0, 0, 0, 0, x1, y1, z1, 1, -z2])
    
    A = np.array(A)
    _, _, V = np.linalg.svd(A)
    H = np.eye(4)
    H[0] = V[-1][:4]
    H[1] = V[-1][4:8]
    H[2] = V[-1][8:12]
    H[3,-1] = V[-1][-1]
    H = H / H[-1, -1]

    # h_pts1 = np.concatenate([face_landmarks1, np.ones((num_points, 1))], axis=1)
    # proj = np.dot(H, h_pts1.T).T
    # proj = proj[:, :3] / proj[:, 3].reshape(-1, 1)
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # ax.scatter(face_landmarks1[:, 0], face_landmarks1[:, 1], face_landmarks1[:, 2], c='r')
    # ax.scatter(proj[:, 0], proj[:, 1], proj[:, 2], c='b')
    # ax.set_xlabel('X Label')
    # ax.set_ylabel('Y Label')
    # ax.set_zlabel('Z Label')
    # ax.view_init(elev=90, azim=0)
    # plt.savefig("3d_points.jpg")
    # raise ValueError

    return H
    

if __name__ == '__main__':
    image_path = "dataset/val/n000001/0007_01.jpg"
    front_face_path = "dataset/val/n000001/0047_01.jpg"
    image = cv2.imread(image_path)
    front_face = cv2.imread(front_face_path)

    H = find_H(image, front_face)

    # face_landmarks = face_landmark(image)

    # # Draw landmarks
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # axes[0].scatter(face_landmarks[:, 0], face_landmarks[:, 1], c='r', s=1)
    # axes[0].set_title("Original Image")
    # axes[0].axis("off")


    # face_landmarks = face_landmark(front_face)
    # axes[1].imshow(cv2.cvtColor(front_face, cv2.COLOR_BGR2RGB))
    # axes[1].scatter(face_landmarks[:, 0], face_landmarks[:, 1], c='r', s=1)
    # axes[1].set_title("Front Face")
    # axes[1].axis("off")

    # plt.savefig("face_landmarks.jpg")
