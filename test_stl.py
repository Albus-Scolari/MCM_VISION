import open3d as o3d
from Source.UTILS.pcd_numpy_utils import *
import matplotlib.pyplot as plt
import numpy as np
filename = "C:\\Users\\alberto.scolari\\Downloads\\pointcloud.txt"
points = np.loadtxt(filename)
pcd = NumpyToPCD(points)


o3d.visualization.draw_geometries([pcd])

#radii = [1,1,1,1]
pcd.estimate_normals()
#mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
#    pcd, o3d.utility.DoubleVector(radii))
#o3d.visualization.draw_geometries([mesh])

mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=15)
densities = np.asarray(densities)
density_colors = plt.get_cmap('plasma')(
    (densities - densities.min()) / (densities.max() - densities.min()))
density_colors = density_colors[:, :3]
density_mesh = o3d.geometry.TriangleMesh()
density_mesh.vertices = mesh.vertices
density_mesh.triangles = mesh.triangles
density_mesh.triangle_normals = mesh.triangle_normals
density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
o3d.visualization.draw_geometries([density_mesh])
o3d.visualization.draw_geometries([mesh])
o3d.io.write_triangle_mesh("C:\\Users\\alberto.scolari\\Downloads\\mesh.stl", density_mesh)