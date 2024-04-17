import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from libs.interpolation import interpolate_triangle_density
from libs.landmark import face_landmark, find_triangles, color_query_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Interpolation')
    parser.add_argument('--image', type=str, default='dataset/val/n000001/0007_01.jpg', help='Path to image')
    parser.add_argument('--density', type=float, default=0.5, help='Density of points in triangle')
    args = parser.parse_args()

    imamge_path = args.image
    density = args.density

    verbose_dir  = "verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)
    image = cv2.imread(imamge_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    ##### get face landmarks and triangles
    landmarks = face_landmark(image)
    triangles = find_triangles()


    ##### from sparse landmarks to dense landmarks
    augmented_lmks = landmarks.copy()
    for triangle in triangles:
        A, B, C = landmarks[triangle]
        points = interpolate_triangle_density(A, B, C, density)
        augmented_lmks = np.vstack((augmented_lmks, points))

    ##### color query
    queried_image = color_query_image(image, augmented_lmks)

    ##### Plot the result
    out_path = os.path.join(verbose_dir, 'face_interpolation.png')
    fig, axes = plt.subplots(1, 2, figsize=(12, 8))
    axes[0].imshow(image)
    axes[0].scatter(augmented_lmks[:,0], augmented_lmks[:,1], s=3, c='r')
    axes[0].set_title("Augmented Landmarks - Density: {:.1f}".format(density))
    axes[0].axis("off")
    axes[1].imshow(queried_image)
    axes[1].set_title("Color Query")
    axes[1].axis("off")
    fig.suptitle("Face Interpolation")
    fig.tight_layout()

    plt.savefig(out_path)
    print("Result saved at:", out_path)