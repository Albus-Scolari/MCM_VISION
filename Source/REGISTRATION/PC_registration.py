import open3d as o3d
import sys
from pathlib import Path
from os.path import dirname, abspath
# include path python
include_dir = str(Path(__file__).resolve().parent.parent.parent)
sys.path.append(include_dir)
print(include_dir)
#import teaserpp_python
import numpy as np 
import copy
#from helpers import pcd2xyz, extract_fpfh, get_teaser_solver, Rt2T, find_correspondences
import Source.REGISTRATION.ICP_utils as ICP_utils 
from Source.UTILS.pcd_numpy_utils import *
from scipy.spatial.transform import Rotation as Rot
import cv2
from pykdtree.kdtree import KDTree
import threading,queue
import alphashape

#
#def mp_align(Source,Target,num,out_que):
#                RT_TOT = []
#                T_TOT = []
#                I = np.asarray([[1,0,0],
#                            [0,1,0],
#                            [0,0,1]])
#                Source_a,  RT1, RT2, am, bm=ICP_utils.PCA_aligndment( copy.deepcopy(Source),copy.deepcopy(Target),num=num)

#                RT_TOT.append(RT1.T)
#                T_TOT.append(np.asarray([0,0,0]))
#                RT_TOT.append(RT2)
#                T_TOT.append(np.asarray([0,0,0]))
#                T_TOT.append(-am)
#                RT_TOT.append(I)
#                T_TOT.append(bm)
#                RT_TOT.append(I)

#                Source_nn = (RT1.T@Source.T).T
#                Source_nn = (RT2@Source_nn.T).T
#                Source_nn = Source_nn -am
#                Source_nn = Source_nn +bm

             

#                Source_nn_pcd = copy.deepcopy(NumpyToPCD(Source_nn))
            
         
         
#                temp_source = copy.deepcopy(Source_nn)
#                #temp_target_pcd = copy.deepcopy(Source_nn_pcd)
#                temp_target = copy.deepcopy(Target)

#                #######################################  
#                ####### ICP with Keypoints ############
#                #######################################

#                RT3, O, error = ICP_utils.ICP_Registration(temp_source,temp_target,1000,bound_l=10,bound_d=0.1)
#                Source_nnn = copy.deepcopy((RT3@temp_source.T).T + O)
#                #Source_nnn = copy.deepcopy((RT3.T@Source_nn.T).T - O) #use inverse rotation and offset

#                RT_TOT.append(RT3)
#                T_TOT.append(O)

      

#                # # local refinement using ICP
        
#                #########################################
#                ######## final ICP Registration #########
#                #########################################

#                #now I invert again source and target
#                #temp_source_pcd2 = copy.deepcopy(Target_pcd)
#                temp_source2 = copy.deepcopy(Source_nnn)
#                #temp_target_pcd2 = copy.deepcopy(Source_nnn_pcd)
#                temp_target2 = copy.deepcopy(Target)

#                RT4, O2,error = ICP_utils.ICP_Registration(temp_source2,temp_target2)
#                Source_nnnn = copy.deepcopy((RT4@temp_source2.T).T+O2) 
#                #Source_nnnn = copy.deepcopy((RT4.T@Source_nnn.T).T - O2)


#                RT_TOT.append(RT4)
#                T_TOT.append(O2)

#                out_que.put([RT_TOT, T_TOT, error])
#                print(
#                #return [RT_TOT, T_TOT, error]
class PointCloudRegistration():
    def DenseAlignment(self,Source,Target):
        '''
        input: CAD pointcloud (Source) and Local dense scan of the workpiece (Target) as numpy array
        output: Transformed Source, rotation list RT_ls and translation list T_ls
        '''
        print('beginning dense alignment')
        ##
        ## ICP registration
        ##
        RT_TOT = []
        T_TOT = []
        #invert source and target since its easier to align the piece with the total
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
        # o3d.visualization.draw_geometries([NumpyToPCD(Source).paint_uniform_color([1, 0, 0]),NumpyToPCD(Target).paint_uniform_color([0, 1, 0]), frame]) # used just to show the initial position
        temp_source = copy.deepcopy(Target)
        temp_target = copy.deepcopy(Source)
        RT, O, error = ICP_utils.ICP_Registration(temp_source,temp_target,npoints = 10000, bound_l = 10, bound_d  = 0.1 )

        Source_n = copy.deepcopy((RT.T@Source.T).T - O)
        # o3d.visualization.draw_geometries([NumpyToPCD(Source_n).paint_uniform_color([0, 0, 0]),NumpyToPCD(Source).paint_uniform_color([1, 0, 0]),
        #                         NumpyToPCD(Target).paint_uniform_color([0, 1, 0]), frame])          # used just to show the final position

        RT_TOT.append(RT.T)
        T_TOT.append(-O)

        return Source_n, RT_TOT, T_TOT


    def SparseAlignment(self,Source_or, Target,VOX,alpha = False, VISUALIZE = False):
        '''
        input: Source and Target pointcloud as numpy array, 
                alignment is performed stepwise, first the principal direction are aligned,
                then the footprint in teh XY plane is aligned
                then an icp registration of the keypoints is performed
                then a final icp registration of the whole pointclouds downsampled randomly
        output: Transformed Source pointcloud and ordered list of RT (rotation) and T (traslation) 
        '''
        VOXEL_SIZE =VOX  #Main parameter that affect the footprint alignment, need to be choosen wisely
        RT_TOT = []
        T_TOT = []
        #VISUALIZE = False
        
        ######## ADDED RANDOM ROTATION TO TEST + NOISE #########
        ########################################################
        #____________________________________________________________________________________________________________________________
        # mu = 0
        # sigma = 0
        # noise0 = np.random.normal(mu,sigma, size = (len(Source),3))
        # Source = Source+noise0
        # ROT = np.asarray(Rot.random().as_matrix())              # random rotation to test consistency
        # Source = (ROT@Source.T).T
        # frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
        # RT_TOT.append(ROT)
        # T_TOT.append(np.asarray([0,0,0]))
        if VISUALIZE:
            o3d.visualization.draw_geometries([NumpyToPCD(Source_or),NumpyToPCD(Target)])
        # # #____________________________________________________________________________________________________________________________
        #o3d.visualization.draw_geometries([NumpyToPCD(Source_or).paint_uniform_color([0, 0, 1]),NumpyToPCD(Target).paint_uniform_color([1.0, 0, 0.0])])
        
        Source_or = NumpyToPCD(Source_or)
        Target = NumpyToPCD(Target)

        Source_or =  Source_or.voxel_down_sample(voxel_size=1)
        Target =  Target.voxel_down_sample(voxel_size=1)
        Source_or = PCDToNumpy(Source_or)
        Target = PCDToNumpy(Target)
        if alpha==True:

            i = 0
            v1 = Source_or[:,0:2]
            v2 = Target[:,0:2]

            while True:
                try:
                    v1 = np.asarray(alphashape.alphashape(v1, 0.1+i).exterior.coords.xy).T
                    v2 = np.asarray(alphashape.alphashape(v2, 0.1+i).exterior.coords.xy).T
                    break
                except:
                    i-=0.01

                    print("*")
                    pass
            Source_or = np.zeros((len(v1),3))
            Target = np.zeros((len(v2),3))

            Source_or[:,0:2] = v1
            Target[:,0:2]= v2
        #Source_or = ICP_utils.alpha_shape(Source_or[:,0:2], alpha=0.5, only_outer=True) #, incremental=True, qhull_options=None)
        #Target = ICP_utils.alpha_shape(Target[:,0:2], alpha=0.5, only_outer=True)

        #za = np.zeros(len(Source_or))
        #zb = np.zeros(len(Target))
        #Source_or  =np.asarray([Source_or[:,0],Source_or[:,1],za]).T
        #Target  =np.asarray([Target[:,0],Target[:,1],zb]).T
        # Source_pcd = NumpyToPCD(Source)
        # Target_pcd = NumpyToPCD(Target)

       
       # o3d.visualization.draw_geometries([NumpyToPCD(Source_or).paint_uniform_color([0, 0, 1]),NumpyToPCD(Target).paint_uniform_color([1.0, 0, 0.0])])
        #disegno origine
        #____________________________________________________________________________________________________________________________
        ###########################
        ##### PCA ALIGNMENT #######
        ###########################
        #threshold = 2.5
        #num=-1
        #error_fin=10
        #while error_fin>threshold:
                #####################################
            ###### MEAN-SHIFT TO TARGET #########
        #####################################
        Source = copy.deepcopy(Source_or)
        source_pmedio = np.mean(Source,axis=0)
        target_pmedio = np.mean(Target,axis=0)
        Source = Source - source_pmedio + target_pmedio
        I = np.asarray([[1,0,0],
                        [0,1,0],
                        [0,0,1]])
        T_TOT.append(-source_pmedio)
        RT_TOT.append(I)
        T_TOT.append(target_pmedio)
        RT_TOT.append(I)
        #if num>-1:
        #    S_ls = [Source, Source, Source,Source]
        #    T_ls = [Target, Target, Target,Target]
        #    n_ls = [0,1,2,3]
        #    out_queue=[queue.Queue(),queue.Queue(),queue.Queue(),queue.Queue()]
        #    thrs_ls = []
        #    for n in n_ls:
        #        t1 = threading.Thread(target = mp_align, args = (S_ls[n], T_ls[n], n,out_queue[n]))
        #        t1.start()
        #        thrs_ls.append(t1)
        #    res = []
        #    for i in range(len(thrs_ls)):
        #        thrs_ls[i].join()
        #        res.append(out_queue[i].get())
        #    res = np.asarray(res)
        #    ind = np.argmin(res[:,2])
        #    error = res[2][ind]
        #    RT_TOT = res[0][ind]
        #    T_TOT = res[1][ind]
        #else:
        Source_a,  RT1, RT2, am, bm, _ =ICP_utils.PCA_alignment( copy.deepcopy(Source),copy.deepcopy(Target))

        RT_TOT.append(RT1.T)
        T_TOT.append(np.asarray([0,0,0]))
        RT_TOT.append(RT2)
        T_TOT.append(np.asarray([0,0,0]))
        T_TOT.append(-am)
        RT_TOT.append(I)
        T_TOT.append(bm)
        RT_TOT.append(I)

        Source_nn = (RT1.T@Source.T).T
        Source_nn = (RT2@Source_nn.T).T
        Source_nn = Source_nn -am
        Source_nn = Source_nn +bm

        if VISUALIZE:
            o3d.visualization.draw_geometries([NumpyToPCD(Source_nn).paint_uniform_color([0, 0, 1]),NumpyToPCD(Target).paint_uniform_color([1.0, 0, 0.0])])

        Source_nn_pcd = copy.deepcopy(NumpyToPCD(Source_nn))
        # if VISUALIZE:
        #     o3d.visualization.draw_geometries([Target_pcd.paint_uniform_color([1.0, 0, 0.0]),NumpyToPCD(Source_nn).paint_uniform_color([0, 0, 1])])
        #____________________________________________________________________________________________________________________________
        ######################################
        ##### TEASER_PP REGISTRATION #########          feature recognition alignement based on teaser++ package, used for footstamp alignment in xy plane
        ######################################
         
         
        temp_source = copy.deepcopy(Source_nn)
        #temp_target_pcd = copy.deepcopy(Source_nn_pcd)
        temp_target = copy.deepcopy(Target)

        #######################################  
        ####### ICP with Keypoints ############
        #######################################

        RT3, O, error = ICP_utils.ICP_Registration(temp_source,temp_target,1000)
        Source_nnn = copy.deepcopy((RT3@temp_source.T).T + O)
        #Source_nnn = copy.deepcopy((RT3.T@Source_nn.T).T - O) #use inverse rotation and offset

        RT_TOT.append(RT3)
        T_TOT.append(O)

      

        if VISUALIZE:
            o3d.visualization.draw_geometries([NumpyToPCD(Source_nnn).paint_uniform_color([0, 0, 1]) ,NumpyToPCD(Target).paint_uniform_color([1, 0, 0]), NumpyToPCD(Source_nn).paint_uniform_color([0,1,1])])  #plot
        # # local refinement using ICP
        
        #########################################
        ######## final ICP Registration #########
        #########################################

        #now I invert again source and target
        #temp_source2 = copy.deepcopy(Source_nnn)
        #temp_target2 = copy.deepcopy(Target)
        
        temp_source2 = copy.deepcopy(Target)
        temp_target2 = copy.deepcopy(Source_nnn)

        RT4, O2,error = ICP_utils.ICP_Registration(temp_source2,temp_target2,bound_l = 10)
        #Source_nnnn = copy.deepcopy((RT4.T@temp_source2.T).T-O2) 

        Source_nnnn = copy.deepcopy((RT4.T@Source_nnn.T).T - O2)

        #temp_source2 = copy.deepcopy(Source_nnn)
        #temp_target2 = copy.deepcopy(Target)
        #RT4_2, O2_2,error2 = ICP_utils.ICP_Registration(temp_source2,temp_target2)
        #Source_nnnn = copy.deepcopy((RT4.T@temp_source2.T).T-O2) 
        #if error1<error2:
        #    RT_TOT.append(RT4_1.T)
        #    T_TOT.append(-O2_1)
        #else:
        RT_TOT.append(RT4.T)
        T_TOT.append(-O2)

        error_fin = error
        print("error_fin : ", error_fin)
                
            #if error_fin>threshold:
            #    RT_TOT = []
            #    T_TOT = []
            #num+=1
            
            #if num==3:
            #    break
        #if VISUALIZE:
       # o3d.visualization.draw_geometries([NumpyToPCD(Source_nnnn).paint_uniform_color([0, 0, 1]),NumpyToPCD(Target).paint_uniform_color([1, 0, 0])])
        return Source_nnnn, RT_TOT, T_TOT, error
    
    

    def Manual_SparseAlignment(self,Source,Target):
        '''
        input: Source and Target PointClouds as numpy arrays,
                need to display the pointcloud in order to get user point input
        output: Source transformed and RT , T transformation
        '''   
        Source_pcd = NumpyToPCD(Source)
        Target_pcd = NumpyToPCD(Target)
        picked_id_source = self.pick_points(Source_pcd)
        picked_id_target = self.pick_points(Target_pcd)
        RT_TOT = []
        T_TOT = []
        assert (len(picked_id_source) >= 3 and len(picked_id_target) >= 3)
        assert (len(picked_id_source) == len(picked_id_target))
        corr = np.zeros((len(picked_id_source), 2))
        corr[:, 0] = picked_id_source
        corr[:, 1] = picked_id_target
        
        # estimate rough transformation using correspondences
        print("Compute a rough transform using the correspondences given by user")
        p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        trans_init = p2p.compute_transformation(Source_pcd,Target_pcd,
                                                o3d.utility.Vector2iVector(corr))
        # point-to-point ICP for refinement
        print("Perform point-to-point ICP refinement")
        threshold = 0.03  # 3cm distance threshold
        reg_p2p = o3d.pipelines.registration.registration_icp(
            Source_pcd, Target_pcd, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        self.draw_registration_result(NumpyToPCD(Source), NumpyToPCD(Target), reg_p2p.transformation)
        print("")
        transf = reg_p2p.transformation
        RT = [transf[0][0:3],
             transf[1][0:3],
             transf[2][0:3]]
        T = np.asarray([transf[0][3], transf[1][3], transf[2][3]])
        RT_TOT.append(RT)
        T_TOT.append(T)
        Source_pcd.transform(transf)
        Source_n = PCDToNumpy(Source_pcd)
         #now I invert source and target, easier to align sparse scan to cad
        temp_source_pcd = copy.deepcopy(Target_pcd)
        temp_source = PCDToNumpy(temp_source_pcd)
        temp_target_pcd = copy.deepcopy(Source_pcd)
        temp_target = PCDToNumpy(temp_target_pcd)
        RT3, O2,error = ICP_utils.ICP_Registration(temp_source,temp_target,npoints=30000)
        temp_source_n = copy.deepcopy((RT3@temp_source.T).T+O2) 
        Source_nn = copy.deepcopy((RT3.T@Source_n.T).T - O2) #inverse rotation and translation
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
        o3d.visualization.draw_geometries([NumpyToPCD(Source_nn).paint_uniform_color([1, 0, 0]), Target_pcd.paint_uniform_color([0, 1, 0]), frame])

        RT_TOT.append(RT3.T)
        T_TOT.append(-O2)


        return Source_nn, RT_TOT, T_TOT, error

    def pick_points(self,pcd):
        print("")
        print(
            "1) Please pick at least three correspondences using [shift + left click]"
        )
        print("   Press [shift + right click] to undo point picking")
        print("2) After picking points, press 'Q' to close the window")
        vis = o3d.visualization.VisualizerWithEditing()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()  # user picks points
        vis.destroy_window()
        print("")
        return vis.get_picked_points()

    def Transformation_with_list(self,Points,RT_ls, T_ls):
        """
        input: Points, np.array(N,3) that need to be transformed, RT_ls rotation matrix list, T_ls translation vector list, need to be of the same length
        output: rototranslated points list
        """
        temp = copy.deepcopy(Points)
        for i in range(len(RT_ls)):
            RT_i = RT_ls[i]
            T_i = T_ls[i]
            temp = (RT_i@temp.T).T+T_i    
        return temp

    def draw_registration_result(self,source, target, transformation):
        source_temp = copy.deepcopy(source)
        target_temp = copy.deepcopy(target)
        source_temp.paint_uniform_color([1, 0.706, 0])
        target_temp.paint_uniform_color([0, 0.651, 0.929])
        source_temp.transform(transformation)
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=500)
        o3d.visualization.draw_geometries([source_temp, target_temp, frame])

    