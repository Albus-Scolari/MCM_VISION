from numpy.core.numeric import count_nonzero
from pyntcloud import PyntCloud 
import numpy as np
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import pandas as pd
import os
import sys
import pdb
import sys
from os.path import dirname, abspath
from pathlib import Path
include_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(include_dir)
print(include_dir)
from Source.UTILS.FrankCommons import *
from Source.UTILS.FrankUtilities import *
from Source.UTILS.pcd_numpy_utils import *
from sklearn.decomposition import PCA
#from utils_STEP import *
from PIL import Image
import copy
import random
import cv2
from pykdtree.kdtree import KDTree
import math
global count
count = 0
from sklearn.metrics import mean_squared_log_error
from scipy.optimize import minimize_scalar
from scipy.spatial import ConvexHull
from scipy.spatial import Delaunay
from scipy.stats import wasserstein_distance
from scipy.spatial import distance


def Edge_Detection(points, k_n, thresh):
    """
    input: points numpy array list, k_n and thresh parameters that influence
            the edge detection
    output: pointcloud of just the edge extracted
    """
    #pcd1 = PyntCloud.from_file("/ArtificialPointClouds/bunny.pcd")
    #pcd1 = PyntCloud.from_file("/TetrahedronMultiple.pcd")
    #pcd1 = PyntCloud.from_file("/ArtificialPointClouds/CubeFractal2.pcd")
    #output_dir = "./detected_edge/"

    #if not os.path.exists(output_dir):
    #    os.makedirs(output_dir)
    clmns = ['x','y','z']
    points_pd = pd.DataFrame(data=points,columns=clmns)
    pcd1 = PyntCloud(points_pd)
    # define hyperparameters


    pcd_np = np.zeros((len(pcd1.points),6))

    # find neighbors
    kdtree_id = pcd1.add_structure("kdtree")
    k_neighbors = pcd1.get_neighbors(k=k_n, kdtree=kdtree_id) 

    # calculate eigenvalues
    ev = pcd1.add_scalar_field("eigen_values", k_neighbors=k_neighbors)

    x = pcd1.points['x'].values 
    y = pcd1.points['y'].values 
    z = pcd1.points['z'].values 

    e1 = pcd1.points['e3('+str(k_n+1)+')'].values
    e2 = pcd1.points['e2('+str(k_n+1)+')'].values
    e3 = pcd1.points['e1('+str(k_n+1)+')'].values

    sum_eg = np.add(np.add(e1,e2),e3)
    sigma = np.divide(e1,sum_eg)
    sigma_value = sigma
    #pdb.set_trace()
    #img = ax.scatter(x, y, z, c=sigma, cmap='jet')

    # visualize the edges
    sigma = sigma>thresh

    #fig = plt.figure()
    #ax = fig.add_subplot(111, projection='3d')

    ## Visualize each one of the eigenvalues
    ##img = ax.scatter(x, y, z, c=e1, cmap='jet')
    ##img = ax.scatter(x, y, z, c=e2, cmap='jet')
    ##img = ax.scatter(x, y, z, c=e3, cmap='jet')

    ## visualize the edges
    #img = ax.scatter(x, y, z, c=sigma, cmap='jet')
    ##img = ax.scatter(x, y, z, c=sigma, cmap=plt.hot())

    #fig.colorbar(img) 
    #plt.show() 

    # Save the edges and point cloud
    thresh_min = sigma_value < thresh
    sigma_value[thresh_min] = 0
    thresh_max = sigma_value > thresh
    sigma_value[thresh_max] = 255

    pcd_np[:,0] = x
    pcd_np[:,1] = y
    pcd_np[:,2] = z
    pcd_np[:,3] = sigma_value

    edge_np = np.delete(pcd_np, np.where(pcd_np[:,3] == 0), axis=0) 

    clmns = ['x','y','z','red','green','blue']
    pcd_pd = pd.DataFrame(data=pcd_np,columns=clmns)
    pcd_pd['red'] = sigma_value.astype(np.uint8)

    #pcd_points = PyntCloud(pd.DataFrame(data=pcd_np,columns=clmns))
    pcd_points = PyntCloud(pcd_pd)
    edge_points = PyntCloud(pd.DataFrame(data=edge_np,columns=clmns))

    # pcd_points.plot()
    # edge_points.plot()

    #PyntCloud.to_file(pcd_points,output_dir+'pointcloud_edges.ply')   # Save the whole point cloud by painting the edge points
    #PyntCloud.to_file(edge_points,output_dir+'edges.ply')             # Save just the edge points
    return edge_np[:,0:3]

def PCA_alignment(Source, Target):
    """
    input: Source and Target Pointclouds as numpy.array,
            in order to work the pointcloud must have a similar shape
    output: Ann, aligned Source pointcloud, RT1, RT2, am, bm rotation and translation used
    """

    A = copy.deepcopy(Source)
    B = copy.deepcopy(Target)
    #VOXEL_SIZE = 1
    #A = NumpyToPCD(A)
    #B = NumpyToPCD(B)
    #A =  A.voxel_down_sample(voxel_size=1)
    #B =  B.voxel_down_sample(voxel_size=1)
    #A = PCDToNumpy(A)
    #B = PCDToNumpy(B)
    ##Shull = ConvexHull(A[:,0:2], incremental=True, qhull_options=None)
    ##A = A[Shull.vertices]
    ##Thull = ConvexHull(B[:,0:2], incremental=True, qhull_options=None)
    ##B = B[Thull.vertices]

    #A = alpha_shape(A[:,0:2], alpha=0.7, only_outer=True) #, incremental=True, qhull_options=None)
    #B = alpha_shape(B[:,0:2], alpha=0.7, only_outer=True)

    #za = np.zeros(len(A))
    #zb = np.zeros(len(B))
    #A  =np.asarray([A[:,0],A[:,1],za]).T
    #B  =np.asarray([B[:,0],B[:,1],zb]).T
    #A = outerboundary2D(A)
    #B = outerboundary2D(B)

    #o3d.visualization.draw_geometries([NumpyToPCD(A).paint_uniform_color([1,0,0]), NumpyToPCD(B).paint_uniform_color([0,0,1])])
    #A = o3d.geometry.keypoint.compute_iss_keypoints(A)
    #B = o3d.geometry.keypoint.compute_iss_keypoints(B)
    #A = PCDToNumpy(A)
    #B = PCDToNumpy(B)
    # A = np.reshape(A,(3,len(A)))
    am = np.mean(A,axis=0)
    # A-=am
    #B = np.reshape(B,(3,len(B)))
    #traslo target in 0
    bm = np.mean(B,axis=0)
    B-=bm
    # M = np.cov(A)
    #calcolo orientamento source 
    pca=PCA(n_components=3)
    pca.fit(A)
    v1 = pca.components_[0]
    v2 = pca.components_[1]
    v3 = pca.components_[2]
    norm = np.linalg.norm(v1)
    v1= list(v1/norm)
    norm = np.linalg.norm(v2)
    v2= list(v2/norm)
    norm = np.linalg.norm(v3)
    v3= list(v3/norm)
    RT1 = np.asarray(FrankCommons.VersorsToRotMatrix(v1, v2, v3))
    print(np.linalg.det(RT1))
    if np.linalg.det(RT1)<0:
         RT1[2,:]*=-1
    #allineo source con xyz
    An = (RT1.T@A.T).T   #rotazione inversa 
    #calcolo orientamento Target 
    pca1=PCA(n_components=3)
    pca1.fit(B)
    v1 = pca1.components_[0]
    v2 = pca1.components_[1]
    v3 = pca1.components_[2]
    #v4 = pca1.components_[3]
    #v5 = pca1.components_[4]
    #v6 = pca1.components_[5]
    norm = np.linalg.norm(v1)
    v1= list(v1/norm)
    norm = np.linalg.norm(v2)
    v2= list(v2/norm)
    norm = np.linalg.norm(v3)
    v3= list(v3/norm)
    v2_ = copy.deepcopy(v2)
    v1_ = copy.deepcopy(v1)
    v3_ = copy.deepcopy(v3)
   
        
    RT2_1 = np.asarray(FrankCommons.VersorsToRotMatrix(v1, v2, v3))
    RT2_2 = np.asarray(FrankCommons.VersorsToRotMatrix(list(-np.asarray(v1_)), list(np.asarray(v2_)), list(np.asarray(v3_))))
    RT2_3 = np.asarray(FrankCommons.VersorsToRotMatrix(list(np.asarray(v1_)), list(-np.asarray(v2_)), list(np.asarray(v3_))))
    RT2_4 = np.asarray(FrankCommons.VersorsToRotMatrix(list(-np.asarray(v1_)), list(-np.asarray(v2_)), list(-np.asarray(v3_))))

    print(np.linalg.det(RT2_1))
    if np.linalg.det(RT2_1)<0:
         RT2_1[2,:]*=-1
    if np.linalg.det(RT2_2)<0:
         RT2_2[2,:]*=-1
    if np.linalg.det(RT2_3)<0:
         RT2_3[2,:]*=-1
    if np.linalg.det(RT2_4)<0:
         RT2_4[2,:]*=-1
    #try the 2 rotations

    Atemp1 = (RT2_1@copy.deepcopy(An).T).T
    Atemp2 = (RT2_2@copy.deepcopy(An).T).T
    Atemp3 = (RT2_3@copy.deepcopy(An).T).T
    Atemp4 = (RT2_4@copy.deepcopy(An).T).T

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
   
    
    #error1 = mean_squared_log_error(Atemp1-np.min(Atemp1,axis=0) ,(copy.deepcopy(B)-np.min(copy.deepcopy(B),axis=0))[ind1])
    #error2 = mean_squared_log_error(Atemp2-np.min(Atemp2,axis=0),(copy.deepcopy(B)-np.min(copy.deepcopy(B),axis=0))[ind2])
    #error3 = mean_squared_log_error(Atemp3-np.min(Atemp3,axis=0) ,(copy.deepcopy(B)-np.min(copy.deepcopy(B),axis=0))[ind3])
    #error4 = mean_squared_log_error(Atemp4-np.min(Atemp4,axis=0) ,(copy.deepcopy(B)-np.min(copy.deepcopy(B),axis=0))[ind4])

    #print(error1)
    #o3d.visualization.draw_geometries([NumpyToPCD(Atemp1-np.mean(Atemp1,axis=0)).paint_uniform_color([1, 0, 0]),
    #                                  NumpyToPCD(copy.deepcopy(B)-np.mean(copy.deepcopy(B),axis=0))])
    #print(error2)

    #o3d.visualization.draw_geometries([NumpyToPCD(Atemp2-np.mean(Atemp2,axis=0)).paint_uniform_color([1, 0, 0]),
    #                                  NumpyToPCD(copy.deepcopy(B)-np.mean(copy.deepcopy(B),axis=0))])
    #print(error3)

    #o3d.visualization.draw_geometries([NumpyToPCD(Atemp3-np.mean(Atemp3,axis=0)).paint_uniform_color([1, 0, 0]),
    #                                  NumpyToPCD(copy.deepcopy(B)-np.mean(copy.deepcopy(B),axis=0))])
    #print(error4)

    #o3d.visualization.draw_geometries([NumpyToPCD(Atemp4-np.mean(Atemp4,axis=0)).paint_uniform_color([1, 0, 0]),
    #                                  NumpyToPCD(copy.deepcopy(B)-np.mean(copy.deepcopy(B),axis=0))])
    #
    #if error1<error2 and error1<error3 and error1<error4:
    #    RT2  = RT2_1
    #    print("1")
    #    #o3d.visualization.draw_geometries([NumpyToPCD(Atemp1-np.min(Atemp1,axis=0)).paint_uniform_color([0, 0, 1]), NumpyToPCD(copy.deepcopy(Target)-np.min(copy.deepcopy(Target),axis=0)).paint_uniform_color([1, 0, 0])])
    #elif error2 <error1 and error2 < error3 and error2<error4:
    #    RT2 = RT2_2
    #    print("2")
    #   # o3d.visualization.draw_geometries([NumpyToPCD(Atemp2-np.min(Atemp2,axis=0)).paint_uniform_color([0, 0, 1]), NumpyToPCD(copy.deepcopy(Target)-np.min(copy.deepcopy(Target),axis=0)).paint_uniform_color([1, 0, 0])])
    #elif error3 <error1 and error3 < error2 and error3<error4:
    #    RT2 = RT2_3
    #    print("3")

    #   # o3d.visualization.draw_geometries([NumpyToPCD(Atemp3-np.min(Atemp3,axis=0)).paint_uniform_color([0, 0, 1]), NumpyToPCD(copy.deepcopy(Target)-np.min(copy.deepcopy(Target),axis=0)).paint_uniform_color([1, 0, 0])])
    #elif error4 <error1 and error4 < error2 and error4<error3:
    #    RT2 = RT2_4
    #    print("4")

    #calcolo bounding box, errore è prodotto scalare assi
    aaa = [Atemp1, Atemp2,Atemp3,Atemp4]
    RT2_ = np.asarray([RT2_1,RT2_2,RT2_3, RT2_4])   
    
    errors = []
    for i in range(len(aaa)):
        a = aaa[i]-np.mean(aaa[i],axis=0)
        b = copy.deepcopy(B)-np.mean(B,axis=0)
        #o3d.visualization.draw_geometries([NumpyToPCD(a).paint_uniform_color([0, 0, 1]), NumpyToPCD(b).paint_uniform_color([1, 0, 0])])
        rot, off, err= ICP_Registration(a,b,npoints =2000)
        errors.append(err)
    print(errors)
    ind = np.argmin(errors)
    aaa = list(np.asarray(aaa)[ind])
    RT2= list(np.asarray(RT2_)[ind])
    #o3d.visualization.draw_geometries([NumpyToPCD(aaa-np.median(aaa,axis=0)).paint_uniform_color([0, 0, 1]), NumpyToPCD(copy.deepcopy(B)-np.median(copy.deepcopy(B),axis=0)).paint_uniform_color([1, 0, 0])])
    #errors = np.asarray([error1, error2, error3, error4])





      #  o3d.visualization.draw_geometries([NumpyToPCD(Atemp4-np.min(Atemp4,axis=0)).paint_uniform_color([0, 0, 1]), NumpyToPCD(copy.deepcopy(Target)-np.min(copy.deepcopy(Target),axis=0)).paint_uniform_color([1, 0, 0])])
    Ann = (RT2@An.T).T
    am1 = np.mean(Ann,axis=0)
    Ann-=am1
    Ann +=bm
    #o3d.visualization.draw_geometries([NumpyToPCD(copy.deepcopy(Ann)).paint_uniform_color([0, 0, 1]), NumpyToPCD(copy.deepcopy(Target)).paint_uniform_color([1, 0, 0])])
    # x0 = [0,0,0,0]
    # bounds = [(-180,180),(-100,100),(-100,100),(-100,100)]
    # res = minimize(error_fun,x0, args = (Ann, B),bounds = bounds)
    # Rotate by 90° to correct if 
    thet = [-90, 0 , 90]
    #errors = []
    # for EZ in thet:
    #     EZ = math.radians(EZ)
    #     Source = copy.deepcopy(Ann)
    #     Target = copy.deepcopy(B)
    #     RT = [[math.cos(EZ), -math.sin(EZ), 0],
    #         [math.sin(EZ), math.cos(EZ), 0],
    #         [0, 0, 1]]
    #     Source = (RT@Source.T).T    
    #     Source -= np.mean(Source,axis = 0)
    #     Source += np.mean(Target,axis = 0)
    #     tree = KDTree(Target)
    #     dist = tree.query(Source)[0]
    #     errors.append(np.sqrt(np.sum(dist**2)/len(dist)))

    #     o3d.visualization.draw_geometries([NumpyToPCD(Source).paint_uniform_color([0, 0, 1]), NumpyToPCD(Target).paint_uniform_color([1, 0, 0])])
    #print(errors)

    return Ann, RT1, RT2, am1, bm, errors[ind] 
def error_fun(x0, Source,Target):
    #rotazione attorno a Z
    EZ, OX, OY,OZ = x0
    EZ = math.radians(EZ)
    RT = [[math.cos(EZ), -math.sin(EZ), 0],
          [math.sin(EZ), math.cos(EZ), 0],
          [0, 0, 1]]

    Source = (RT@Source.T).T + [OX,OY,OZ]
    Source -= np.mean(Source,axis = 0)
    Source += np.mean(Target,axis = 0)
    tree = KDTree(Target)
    dist = tree.query(Source)[0]
    error =np.sqrt(np.sum(dist**2)/len(dist))
    #o3d.visualization.draw_geometries([NumpyToPCD(Source).paint_uniform_color([0, 0, 1]), NumpyToPCD(Target).paint_uniform_color([1, 0, 0])])
    print(error)
    return error
def ICP_Registration(Source,Target,npoints = 50000,bound_l = 0, bound_d = 0):
        '''
        ICP registration algorithm, found optimal value to
        compute trasformation to align Source pointcloud
        to target point cloud.
        return: RT and offset to align source to target
        '''
        
        if len(Source)>npoints:
            Source = np.asarray(random.sample(list(Source), npoints))
        if len(Target)>npoints:
            Target = np.asarray(random.sample(list(Target), npoints))
        args = (Source, Target)

        x = [0,0,0,0,0,0]
        if bound_l != 0 :
           # bnds = ((-bound_l,bound_l),(-bound_l,bound_l),(-bound_l,bound_l),(-bound_d,bound_d),(-bound_d,bound_d),(-bound_d,bound_d))
            res = minimize(Error_ICP_function, x, args = args,method='Powell', tol=0.0000001) # , bounds = bnds)
        else:
            res = minimize(Error_ICP_function, x, args = args,method='Powell', tol=0.0000001)

        [OX,OY,OZ,EX,EY,EZ] = res.x
        RT = Euler2RotMatrix([EZ,EY,EX])
        RT = np.asarray(RT)
        off = [OX,OY,OZ]
    
        return RT, np.asarray(off), float(res.fun)

def Error_ICP_function(x,Source,Target):
    """
    cost function fo ICP minimization
    """
    global count

    [OX,OY,OZ,EX,EY,EZ] = x
    
    RT = Euler2RotMatrix([EZ,EY,EX])
    RT = np.asarray(RT)

    Source_New = (RT@Source.T).T + [OX,OY,OZ]
    # if count%20 ==0:
    #     o3d.visualization.draw_geometries([NumpyToPCD(Source_New).paint_uniform_color([1, 0, 0]),NumpyToPCD(Target).paint_uniform_color([0, 0, 1])])
    tree = KDTree(Target)
    distances, index = tree.query(Source_New)
    #trsh = np.percentile(distances,99)
    #bol = distances<trsh
    #distances = distances[bol]
    errors = np.power(np.sum(distances**5/len(distances)), 0.2) 
    #errors  = np.sqrt(np.square(np.log(Source_New + 1) - np.log(Target[index] + 1)).mean())
    #errors = mean_squared_log_error(Source_New-np.min(Source_New,axis=0), Target[index]-np.min(Target[index],axis=0))
    count+=1
    
    #print(x)
    #print(errors)
    return errors

def extract_fpfh(pcd, voxel_size):
  radius_normal = voxel_size * 2
  pcd.estimate_normals(
      o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

  radius_feature = voxel_size * 5
  fpfh = o3d.pipelines.registration.compute_fpfh_feature(
      pcd, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
  return np.array(fpfh.data).T

def outerboundary2D(points):
    #translate to origin
    center = np.mean(points,axis=0)
    points = points-center
    #convert in polar cordinates
    r = np.sqrt(points[:,0]**2+points[:,1]**2)
    theta = np.arctan2(points[:,1],points[:,0])
    #just xy
    out_r = []
    out_t = []
    
    for i in range(1,720):
        temp_T = copy.deepcopy(theta)
        temp_R = copy.deepcopy(r)
        th0 = math.radians(i)
        th1 = math.radians(i+1)
        #select points inside the slice
        bol1 = temp_T>th0
        temp_T = temp_T[bol1]
        bol2 = temp_T<th1
        temp_T = temp_T[bol2]
        temp_R = temp_R[bol1]
        temp_R = temp_R[bol2]
        #
        if temp_R.size!=0:
            ind = np.argmax(temp_R)
            r_n = temp_R[ind]
            t_n = temp_T[ind]
            out_r.append(r_n)
            out_t.append(t_n)
    out_r = np.asarray(out_r)
    out_t = np.asarray(out_t)
    x = out_r*np.cos(out_t)
    y = out_r*np.sin(out_t)
    z = np.zeros(len(x))
    out = np.asarray([x,y,z]).T+center
    return out

def alpha_shape(points, alpha, only_outer=True):
    """
    Compute the alpha shape (concave hull) of a set of points.
    :param points: np.array of shape (n,2) points.
    :param alpha: alpha value.
    :param only_outer: boolean value to specify if we keep only the outer border
    or also inner edges.
    :return: set of (i,j) pairs representing edges of the alpha-shape. (i,j) are
    the indices in the points array.
    """
  
    

    tri = Delaunay(points)
    edges = []
    # Loop over triangles:
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = points[ia]
        pb = points[ib]
        pc = points[ic]
        # Computing radius of triangle circumcircle
        # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
        a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
        b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
        c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
        s = (a + b + c) / 2.0
        area = np.sqrt(s * (s - a) * (s - b) * (s - c))
        circum_r = a * b * c / (4.0 * area)
        if circum_r < alpha:
            edges.append(pa)
            edges.append(pb)
            edges.append(pc)

    return np.asarray(edges)