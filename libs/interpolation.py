import os
import numpy as np
import matplotlib.pyplot as plt

# generate an R2 sequence
def r2_sequence(num):
    # R2 sequence
    points = []
    g = 1.32471795724474602596
    a1 = 1.0/g
    a2 = 1.0/(g*g)
    for n in range(0,num):
        x = (0.5+a1*n)
        y = (0.5+a2*n)
        points.append((x%1,y%1))
    return points


# generates points in a triangle using paralleogram method applied to R2 sequence
def interpolate_triangle_points(A, B, C, num_points):
    # vectors of triangle
    AC = C - A
    AB = B - A
    
    # generate r1 and r2
    sequence = r2_sequence(num_points)
    
    # parallelogram method.....
    points = np.zeros((num_points,3))
    # generate point in paralleogram using r1 and r2 from R2 sequence
    for i in range(num_points):
        r1,r2 = sequence[i]
        if (r1+r2) < 1:
            points[i]= (r1 * AC) + (r2 * AB) + A
        else:
            points[i] = (1-r1) * AC + (1-r2) * AB + A
    return points
   
# generates points in a triangle using paralleogram method and pt density
def interpolate_triangle_density(A, B, C, density, verbose=False):
    # vectors of triangle
    AC = C - A
    AB = B - A
    
    # get area of triangle
    cross_product = np.cross(AC, AB)
    area = 0.5 * np.linalg.norm(cross_product)
    # calculate number of points needed
    num_points = int(density * area)
    
    # generate r1 and r2
    sequence = r2_sequence(num_points)
    
    # parallelogram method.....
    points = np.zeros((num_points,3))
    # generate point in paralleogram using r1 and r2 from R2 sequence
    for i in range(num_points):
        r1,r2 = sequence[i]
        
        if (r1+r2) < 1:
            points[i]= (r1 * AC) + (r2 * AB) + A
        else:
            points[i] = (1-r1) * AC + (1-r2) * AB + A

    if verbose:
        print(str(num_points) + " points generated")
        
    return points

if __name__ == "__main__":

    verbose_dir  = "verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)

    pt1 = np.array([0, 0, 0])
    pt2 = np.array([2, 0, 0])
    pt3 = np.array([1, 1, 0])
    print("Triangle Points:")
    print(" - pt1: {}".format(pt1))
    print(" - pt2: {}".format(pt2))
    print(" - pt3: {}".format(pt3))

    # interpolate with fixed number of points
    num_points = 1000
    interpolated_pts_fp = interpolate_triangle_points(pt1, pt2, pt3, num_points) # fixed points

    print("Interpolate with Fixed Number of Points: {}".format(num_points))
    print(" - Shape: {}".format(interpolated_pts_fp.shape))

    # interpolate with fixed density
    density = 200
    interpolated_pts_fd = interpolate_triangle_density(pt1, pt2, pt3, density) # fixed density
    
    print("Interpolate with Fixed Density: {}".format(density))
    print(" - Shape: {}".format(interpolated_pts_fd.shape))
    print("=====================================")

    # Plot the result
    out_path = os.path.join(verbose_dir, 'demo_interpolation.png')
    fig, ax = plt.subplots(1, 2, figsize=(12, 8), subplot_kw={'projection': '3d'})
    ax[0].set_title("Fixed #Points: {}".format(num_points))
    ax[0].scatter(interpolated_pts_fp[:,0], interpolated_pts_fp[:,1], interpolated_pts_fp[:,2], c='r', s=1)
    ax[1].set_title("Fixed Density: {}".format(density))
    ax[1].scatter(interpolated_pts_fd[:,0], interpolated_pts_fd[:,1], interpolated_pts_fd[:,2], c='r', s=1)

    fig.suptitle("Demo interpolation.py")
    plt.tight_layout()
    plt.savefig(out_path)
    print("Results saved to: {}".format(out_path))
