import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt
from libs.interpolation import interpolate_triangle_density
from libs.landmark import find_triangles, color_query_image, find_H_from_image, homography_transform, color_query, inverse_homography_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Face Interpolation')
    parser.add_argument('--query_image', type=str, default='dataset/val/n000001/0007_01.jpg', help='Path to image')
    parser.add_argument('--front_image', type=str, default='dataset/val/n000001/0047_01.jpg', help='Path to front face image')
    
    parser.add_argument('--density', type=float, default=0.5, help='Density of points in triangle')
    args = parser.parse_args()

    query_image_path = args.query_image
    front_image_path = args.front_image
    density = args.density

    verbose_dir  = "verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)
    query_image = cv2.imread(query_image_path)
    query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)
    front_image = cv2.imread(front_image_path)
    front_image = cv2.cvtColor(front_image, cv2.COLOR_BGR2RGB)

    ##### get face landmarks and triangles
    H, (front_lmks, _) = find_H_from_image(front_image, query_image)
    triangles = find_triangles()


    ##### from sparse landmarks to dense landmarks
    augmented_lmks = front_lmks.copy()
    for triangle in triangles:
        A, B, C = front_lmks[triangle]
        points = interpolate_triangle_density(A, B, C, density)
        augmented_lmks = np.vstack((augmented_lmks, points))

    ##### color query
    homography_transformed_lmks = homography_transform(augmented_lmks, H)
    queried_image = color_query_image(query_image, homography_transformed_lmks) 

    colors, valid_lmks = color_query(query_image, homography_transformed_lmks) 
    inverse_homography_transformed_lmks = inverse_homography_transform(valid_lmks, H)
    max_x, max_y, max_z = np.max(inverse_homography_transformed_lmks, axis=0).astype(int)
    min_x, min_y, min_z = np.min(inverse_homography_transformed_lmks, axis=0).astype(int)
    out = np.zeros((max_y - min_y + 1, max_x - min_x + 1, 3), dtype=np.uint8)
    offset = np.array([min_x, min_y])
    out[inverse_homography_transformed_lmks[:, 1].astype(int) - offset[1], inverse_homography_transformed_lmks[:, 0].astype(int) - offset[0]] = colors
    


    ##### Plot the result
    out_path = os.path.join(verbose_dir, 'demo.png')
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0,0].imshow(query_image)
    axes[0,0].set_title("Query Image")
    axes[0,0].axis("off")

    axes[0,1].scatter(homography_transformed_lmks[:, 0], homography_transformed_lmks[:, 1], c='r', s=1)
    axes[0,1].set_title("Warped Front Face")
    axes[0,1].set_aspect('equal')
    axes[0,1].invert_yaxis()


    axes[0,2].imshow(queried_image)
    axes[0,2].set_title("Color Query")
    axes[0,2].axis("off")

    axes[1,0].imshow(front_image)
    axes[1,0].scatter(front_lmks[:, 0], front_lmks[:, 1], c='r', s=1)
    axes[1,0].set_title("Front Image")
    axes[1,0].axis("off")

    axes[1,1].scatter(augmented_lmks[:, 0], augmented_lmks[:, 1], c='r', s=1)
    axes[1,1].set_title("Augmented Landmarks - Density: {:.1f}".format(density))
    axes[1,1].set_aspect('equal')
    axes[1,1].invert_yaxis()
    
    axes[1,2].imshow(out)
    axes[1,2].set_title("Color Query")
    axes[1,2].axis("off")
    
    fig.tight_layout()
    plt.savefig(out_path)
    print("Result saved at:", out_path)