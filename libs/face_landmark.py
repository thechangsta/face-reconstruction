import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from collections import defaultdict
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
    return H, (face_landmarks1, face_landmarks2)

def homography_transform(points, H):
    h_points = np.ones((points.shape[0], 4))
    h_points[:, :3] = points

    warped_points = np.dot(H, h_points.T).T
    warped_points = warped_points / warped_points[:, -1].reshape(-1, 1)
    return warped_points[:, :3]

def inverse_homography_transform(points, H):
    H_inv = np.linalg.inv(H)
    return homography_transform(points, H_inv)

def find_triangles(edges):
    graph = defaultdict(set) 
    for u, v in edges:
        graph[u].add(v)
        graph[v].add(u)

    triangles = set()
    for u, v in edges:
        common_neighbors = graph[u].intersection(graph[v])
        for w in common_neighbors:
            triangle = tuple(sorted((u, v, w)))
            triangles.add(triangle)
    
    return np.array(list(triangles))

def find_areas(points, triangles):
    areas = []
    for triangle in triangles:
        a, b, c = points[triangle]
        area = 0.5 * np.linalg.norm(np.cross(b - a, c - a))
        areas.append(area)
    return np.array(areas)

if __name__ == '__main__':

    verbose_dir  = "verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)

    source_path = "dataset/val/n000001/0007_01.jpg"
    target_path = "dataset/val/n000001/0047_01.jpg"
    source_image = cv2.imread(source_path)
    target_image = cv2.imread(target_path)

    H, (source_lmks, target_lmks) = find_H(source_image, target_image)
    triangles = find_triangles(mp.solutions.face_mesh.FACEMESH_TESSELATION)
    source_areas = find_areas(source_lmks, triangles)
    target_areas = find_areas(target_lmks, triangles)
    print("=====================================")
    print("Homography Matrix:")
    print(H)
    print("Number of triangles:", triangles.shape[0])
    print("Max, Min triangle area in source image:", np.max(source_areas), np.min(source_areas))
    print("Max, Min triangle area in target image:", np.max(target_areas), np.min(target_areas))

    # Draw landmarks
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    axes[0,0].scatter(source_lmks[:, 0], source_lmks[:, 1], c='r', s=1)
    axes[0,0].set_title("Source Face")
    axes[0,0].axis("off")

    warped_lmks = homography_transform(source_lmks, H)
    axes[0,1].imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    axes[0,1].scatter(warped_lmks[:, 0], warped_lmks[:, 1], c='r', s=1)
    axes[0,1].set_title("Warped Source Face - MSE: {:.4F}".format(np.mean((warped_lmks - target_lmks) ** 2)))
    axes[0,1].axis("off")

    inverse_warped_lmks = inverse_homography_transform(warped_lmks, H)
    axes[0,2].imshow(cv2.cvtColor(source_image, cv2.COLOR_BGR2RGB))
    axes[0,2].scatter(inverse_warped_lmks[:, 0], inverse_warped_lmks[:, 1], c='r', s=1)
    axes[0,2].set_title("Inverse Warped Source Face - MSE: {:.4F}".format(np.mean((inverse_warped_lmks - source_lmks) ** 2)))
    axes[0,2].axis("off")

    axes[1,0].imshow(cv2.cvtColor(target_image, cv2.COLOR_BGR2RGB))
    axes[1,0].scatter(target_lmks[:, 0], target_lmks[:, 1], c='r', s=1)
    axes[1,0].set_title("Target Face")
    axes[1,0].axis("off")

    demo_triangle_idx = 0
    for line in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        axes[1,1].plot(source_lmks[line, 0], source_lmks[line, 1], c='b', alpha=0.1)
    axes[1,1].scatter(source_lmks[triangles[demo_triangle_idx], 0], source_lmks[triangles[demo_triangle_idx], 1], c='r', s=3)
    axes[1,1].fill(source_lmks[triangles[demo_triangle_idx], 0], source_lmks[triangles[demo_triangle_idx], 1], c='r', alpha=0.5)
    axes[1,1].set_title("Source Face Landmarks")
    axes[1,1].set_aspect('equal')
    axes[1,1].invert_yaxis()

    for line in mp.solutions.face_mesh.FACEMESH_TESSELATION:
        axes[1,2].plot(target_lmks[line, 0], target_lmks[line, 1], c='b', alpha=0.1)
    axes[1,2].scatter(target_lmks[triangles[demo_triangle_idx], 0], target_lmks[triangles[demo_triangle_idx], 1], c='r', s=3)
    axes[1,2].fill(target_lmks[triangles[demo_triangle_idx], 0], target_lmks[triangles[demo_triangle_idx], 1], c='r', alpha=0.5)
    axes[1,2].set_title("Target Face Landmarks")
    axes[1,2].set_aspect('equal')
    axes[1,2].invert_yaxis()
    
    plt.tight_layout()
    plt.savefig(os.path.join(verbose_dir, 'face_landmark.png'))
    print("Face landmarks saved at", os.path.join(verbose_dir, 'face_landmark.png'))

    print("=====================================")

