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
            points[i]= (r1 * AC) + (r2 * AB)
        else:
            points[i] = (1-r1) * AC + (1-r2) * AB
    return points
   
# generates points in a triangle using paralleogram method and pt density
def interpolate_triangle_density(A, B, C, density):
    # vectors of triangle
    AC = C - A
    AB = B - A
    
    # get area of triangle
    cross_product = np.cross(AC, AB)
    area = 0.5 * np.linalg.norm(cross_product)
    
    # calculate number of points needed
    num_points = int(density * area)
    print(str(num_points) + " points generated")
    
    # generate r1 and r2
    sequence = r2_sequence(num_points)
    
    # parallelogram method.....
    points = np.zeros((num_points,3))
    # generate point in paralleogram using r1 and r2 from R2 sequence
    for i in range(num_points):
        r1,r2 = sequence[i]
        if (r1+r2) < 1:
            points[i]= (r1 * AC) + (r2 * AB)
        else:
            points[i] = (1-r1) * AC + (1-r2) * AB
    return points

if __name__ == "__main__":

    verbose_dir  = "verbose"
    if not os.path.exists(verbose_dir):
        os.makedirs(verbose_dir)

    # Example usage 1
    pt1 = np.array([0, 0, 0])
    pt2 = np.array([2, 0, 0])
    pt3 = np.array([1, 1, 0])
    num_points = 1000
    interpolated_pts = interpolate_triangle_points(pt1, pt2, pt3, num_points)

    # Plot the points
    out_path = os.path.join(verbose_dir, 'interpolated_points.png')
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    x_coords = interpolated_pts[:, 0]
    y_coords = interpolated_pts[:, 1]
    z_coords = interpolated_pts[:, 2]
    ax.scatter(x_coords, y_coords, z_coords)
    plt.savefig(out_path)

    print("Triangle Points:")
    print(" - pt1: {}".format(pt1))
    print(" - pt2: {}".format(pt2))
    print(" - pt3: {}".format(pt3))
    print("Interpolated Points:")
    print("Shape: {}".format(interpolated_pts.shape))
    print("Saved to {}".format(out_path))
    print("====================================")


    # Example usage 2
    pt1 = np.array([0, 0, 0])
    pt2 = np.array([4, 0, 1])
    pt3 = np.array([1, 0, 0])

    # density is pts per unit of area
    out_path = os.path.join(verbose_dir, 'interpolated_points_density.png')
    density = 200
    interpolated_pts = interpolate_triangle_density(pt1, pt2, pt3, density)

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(projection='3d')
    x_coords = interpolated_pts[:, 0]
    y_coords = interpolated_pts[:, 1]
    z_coords = interpolated_pts[:, 2]

    ax.scatter(x_coords, y_coords, z_coords)
    plt.savefig(out_path)

    print("Triangle Points:")
    print(" - pt1: {}".format(pt1))
    print(" - pt2: {}".format(pt2))
    print(" - pt3: {}".format(pt3))
    print("Density: {}".format(density))
    print("Interpolated Points:")
    print("Shape: {}".format(interpolated_pts.shape))
    print("Saved to {}".format(out_path))
    print("====================================")