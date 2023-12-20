#from alphashape.alphashape import alphasimplices
from sklearn.base import RegressorMixin
from Source import VimbaCameraDriver as vcd
from tkinter import messagebox as mbox, IntVar
import tkinter as tk
from vimba import *
import numpy as np
import glob
import threading
import open3d as o3d
from Source.UTILS.pcd_numpy_utils import *
from Source.REGISTRATION import PC_registration
import pyforms
from pyforms.basewidget import BaseWidget
from pyforms.controls   import ControlFile
from pyforms.controls   import ControlText
from pyforms.controls   import ControlSlider
from pyforms.controls   import ControlPlayer
from pyforms.controls   import ControlButton
from pyforms.controls   import ControlCheckBox
from pyforms.controls   import ControlNumber
from pyforms.controls   import ControlImage
from pyforms.controls   import ControlProgress
from pyforms.controls   import ControlCombo
from pyforms.controls import ControlMatplotlib
from scipy.spatial import ConvexHull
import skimage.morphology, skimage.data
import cv2
import random
import time
import copy
import sklearn.cluster as cluster
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
#from Source import STEPAnalyzer
from Source import helper
from Source.UTILS.pcd_numpy_utils import *
from Source.REGISTRATION.ICP_utils import *
from scipy.spatial.transform import Rotation as Rot
#import torch
import math
import multiprocessing as mp
#import sys
from pathlib import Path
from os.path import dirname, abspath
# include path python
from Source.UTILS.FrankUtilities import *
#from descartes import PolygonPatch
class application(BaseWidget):
    Cameras_ID = ["DEV_1AB22C010C0B","DEV_1AB22C010C0A", "DEV_1AB22C010C0C"]
    Cameras_connected = []
    acquire = None
    scene_points = []
    hh = helper.Helperhandler()
    match_cad = []
    match_cad_temp = []
    match_det_temp = []
    cad_views = []
    match_det = []
    include_dir = str(Path(__file__).resolve().parent)
    sys.path.append(include_dir)
    origin = None
    def __init__(self):
        super(application,self).__init__('Humans - Hub')
        self.connect_cameras_button  =ControlButton("Connect Cameras")
        self.connect_cameras_button.value = self.connect_cameras
        self.camera1_connected = ControlCheckBox("Cam 1", default = False, enabled = False)
        self.camera2_connected = ControlCheckBox("Cam 2", default = False, enabled = False)
        self.camera3_connected = ControlCheckBox("Cam 3", default = False, enabled = False)
        self.image = None
        self.thresh_image = None
        self.object_image = None
        self.grab_image = ControlButton("Capture Image")
        self.grab_image.value = self.grab_stitched_image
        self.stitched_image = ControlImage()
        self.pool = mp.Pool()

        self.change_exposure_button = ControlButton("set exposure time")
        self.change_exposure_button.value = self.change_exposure
        self.exposure_setpoint = ControlText('Exposure Time [us]', default = '10000')
        self.step_filepath = ControlText('STEP filepath', default = self.include_dir + "\\DATA\\STEP_FILE")
        self.load_views = ControlButton("Load Views")
        self.load_views.value = self.buttonLoadViews
        self.cad_image1 = ControlImage()
        self.cad_image2 = ControlImage()
        self.cad_image3 = ControlImage()
        self.calibrate_camera = ControlButton("calibrate camera")
        self.calibrate_camera.value = self.buttonCalibrateCamera
        self.filter_image = ControlButton("filter image")
        self.filter_image.value = self.button_filter_image
        self.threshold_slider = ControlSlider(default=211)
        self.threshold_slider.min = 0
        self.threshold_slider.max = 255
        self.threshold_slider2 = ControlSlider(default=6)
        self.threshold_slider2.min = 0
        self.threshold_slider2.max = 100
        self.find_object = ControlButton("find objects")
        self.find_object.value = self.button_find_object
        self.match_cad_object = ControlButton("match cad object")
        self.match_cad_object.value = self.buttonMatchingObject
        self.align_matched = ControlButton("align matched")
        self.align_matched.value = self.buttonAlignMatched
        self.plot_scene = ControlButton("plot scene")
        self.plot_scene.value = self.buttonPlotCurrentScene
        self.find_origin = ControlButton("find origin")
        self.find_origin.value = self.findOrigin
        self.remove_background = ControlButton("remove background")
        self.remove_background.value = self.buttonRemoveBackground
        self.remove_shadows = ControlButton("remove shadows")
        self.remove_shadows.value = self.buttonRemoveShadows


        self.formset = [ {
        'a:Cameras':[
            ("connect_cameras_button", "camera1_connected", "camera2_connected", "camera3_connected"),
            ("grab_image","remove_background","remove_shadows", "filter_image", "threshold_slider","threshold_slider2","find_origin", "find_object","exposure_setpoint", "change_exposure_button"),
            ("stitched_image"),
            ("calibrate_camera")

        ],
        'b:Step Load':[
            ("step_filepath", "load_views"),
            ("cad_image1","cad_image2","cad_image3"),
            #("pointcloud_scene"),
            ("plot_scene"),
            ("match_cad_object", "align_matched")

        ]
        }
        ]
        pass





    
    
    def buttonLoadViews(self):
        cam = vcd.Vimba_camera()
        self.cad_views=[]
        pxmm = cam.pxmm
        folder  = str(self.step_filepath.value)
        #step_ls =  ["Paratacco SX SBK.stp", "leva freno RACE piegata.stp", "SUPP PEDANA DX kawz900.stp", "Tendi catena DX scr.stp", "manubri 52 DX.stp", "manubri 52 SX.stp"  ]
        #for kk in range(len(step_ls)):
        #stp = step_ls[kk]

        
        files = glob.glob(folder+"/*.stp")

        for f in files:

            points = self.hh.LoadSTEP(f)
            ##flatten pointcloud
            self.cad_points = points
            #o3d.visualization.draw_geometries([NumpyToPCD(points).paint_uniform_color([1,0,0])])
            # convert pointcloud in image
            view0 = copy.deepcopy(np.asarray(points))
            img1,_ = self.hh.PointCloudToImage(points)
            temp1 = copy.deepcopy(points)
            temp1[:,0] = points[:,1]
            temp1[:,1] = points[:,2]
            temp1[:,2] = points[:,0]
            #temp1[:,0] = np.zeros(len(temp1))
            view1 = copy.deepcopy(np.asarray(temp1))

            #o3d.visualization.draw_geometries([NumpyToPCD(temp1).paint_uniform_color([1,0,0])])

            img2,_ =self.hh.PointCloudToImage(temp1)
            temp2 = copy.deepcopy(points)
            temp2[:,0] = points[:,2]
            temp2[:,1] = points[:,0]
            temp2[:,2] = points[:,1]
            #
            view2 = copy.deepcopy(np.asarray(temp2))
            #temp2[:,1] = np.zeros(len(temp2))
            #o3d.visualization.draw_geometries([NumpyToPCD(temp2).paint_uniform_color([1,0,0])])
            img3,_ = self.hh.PointCloudToImage(temp2)

            # temp2 = copy.deepcopy(points)
            # temp2[:,1] = points[:,2]
            # temp2[:,0] = points[:,0]
            # temp2[:,2] = points[:,1]
            ##
            # img4 = self.hh.PointCloudToImage(temp2)

            self.cad_views.append([view0,view1,view2])
            #self.cad_views.append(view1)
            #self.cad_views.append(view2)


        #####back1 = cv2.imread("D:/RepoGithub/humans-RME/DATA/back1.jpg")
        #####back2 = cv2.imread("D:/RepoGithub/humans-RME/DATA/back2.jpg")
        #####back3 = cv2.imread("D:/RepoGithub/humans-RME/DATA/back3.jpg")
        #####back4 = cv2.imread("D:/RepoGithub/humans-RME/DATA/back4.jpg")
        #####back5 = cv2.imread("D:/RepoGithub/humans-RME/DATA/back5.jpg")
        #####back6 = cv2.imread("D:/RepoGithub/humans-RME/DATA/back6.jpg")
        #######create random views
        #####for i in range(200):
        #####    if kk == 3 or kk==4 or kk == 5:
        #####        temp3 = copy.deepcopy(temp2)
        #####    elif kk==0:
        #####        temp3 = copy.deepcopy(temp1)
        #####    elif kk== 1 or kk ==2:
        #####        temp3 = copy.deepcopy(view0)



        #####    if kk in [0,1,2,3]:
        #####        if random.random()>0.5:
        #####            eul = [random.random()*3,random.random()*0.7,random.random()*0.7]
        #####        else:
        #####            eul = [random.random()*3,random.random()*0.7+3.14,random.random()*0.7]
        #####    elif kk in [4,5]:
        #####        ran = random.random()
        #####        if ran>0.25:
        #####            eul = [random.random()*3,random.random()*0.5,random.random()*0.5]
        #####        elif ran>=0.25 or ran<0.5:
        #####            eul = [random.random()*3,random.random()*0.5+1.57,random.random()*0.5]
        #####        elif ran>=0.5 or ran<0.75:
        #####            eul = [random.random()*3,random.random()*0.5+3.14,random.random()*0.5]
        #####        elif ran>=0.75 or ran<1:
        #####            eul = [random.random()*3,random.random()*0.5+4.71,random.random()*0.5]
                

        #####    ROT= np.asarray(Euler2RotMatrix(eul))
        #####    #ROT = np.asarray(Rot.random().as_matrix())              # random rotation to test consistency
        #####    temp3 = (ROT@temp3.T).T
        #####    #o3d.visualization.draw_geometries([NumpyToPCD(temp3).paint_uniform_color([1,0,0])])
        #####    img = self.hh.PointCloudToImage(temp3)
        #####    if kk in [1,2,3,4,5]:
        #####        img = 255-img+70
        #####    elif kk ==0:
        #####        img = 255-img
        #####    #mask = 255-np.random.rand(img.shape[0],img.shape[1])*50
        #####    #bol = img>254
        #####    #img[bol] = mask[bol]
        #####    #add a bigger background
        #####    h, w = img.shape
        #####    print(h)
        #####    #back = 255-np.random.rand(img.shape[0]*3,img.shape[1]*3)*30
        #####    back =255*np.ones((img.shape[0]*3,img.shape[1]*3))

        #####    yoff = int(random.random()*img.shape[0]*2)
        #####    xoff = int(random.random()*img.shape[1]*2)

        #####    back[yoff:yoff+h, xoff:xoff+w] = img


        #####    mask = back>250
        #####    back = back.astype(np.uint8)
        #####    back = cv2.cvtColor(back, cv2.COLOR_GRAY2BGR)

        #####    #add a random back ground
        #####    t = random.random()
        #####    if t<0.14:
        #####        back0 = 255-np.random.rand(back.shape[0],back.shape[1])*60
        #####        back0 = back0.astype(np.uint8)
        #####        back0 = cv2.cvtColor(back0, cv2.COLOR_GRAY2BGR)
        #####        back[mask] = back0[mask]
        #####    elif t>=0.14 and t<0.28:
        #####        back1 = cv2.resize(back1, (back.shape[1],back.shape[0]), interpolation= cv2.INTER_LINEAR)
        #####        back[mask] =  back1[mask]

        #####    elif t>=0.28 and t<0.42:
        #####        back2 = cv2.resize(back2, (back.shape[1],back.shape[0]), interpolation= cv2.INTER_LINEAR)
        #####        back[mask] =  back2[mask]
        #####    elif t>=0.42 and t<0.56:
        #####        back6 = cv2.resize(back6, (back.shape[1],back.shape[0]), interpolation= cv2.INTER_LINEAR)
        #####        back[mask] =  back6[mask]
        #####    elif t>=0.56 and t<0.7:
        #####        back3 = cv2.resize(back3,(back.shape[1],back.shape[0]), interpolation= cv2.INTER_LINEAR)
        #####        back[mask] =  back3[mask]
        #####    elif t>=0.7 and t<0.84:
        #####        back4 = cv2.resize(back4,(back.shape[1],back.shape[0]), interpolation= cv2.INTER_LINEAR)
        #####        back[mask] =  back4[mask]
        #####    elif t>=0.84 and t<1:
        #####        back5 = cv2.resize(back5,(back.shape[1],back.shape[0]), interpolation= cv2.INTER_LINEAR)
        #####        back[mask] =  back5[mask]


        #####    c_y = (yoff+h/2)/back.shape[0]
        #####    c_x = (xoff+w/2)/back.shape[1]
        #####    width = w/back.shape[1]
        #####    height = h/back.shape[0]

        #####    #points_box = np.asarray([[xoff,yoff],[xoff+w,yoff+h],[xoff+w,yoff],[xoff,yoff+h]])
        #####    #points_box = np.array(points_box).reshape((-1,1,2)).astype(np.int32)
        #####    #cv2.drawContours( back,[points_box],0,(0,0,255),20)

        #####    cv2.imwrite("D://RepoGithub//humans-RME//test_render//cad"+str(kk)+"_"+str(i)+".jpg", back)
        #####    f = open("D://RepoGithub//humans-RME//test_render//cad"+str(kk)+"_"+str(i)+".txt","w")
        #####    f.write(str(kk)+" "+str(c_x)+" "+str(c_y)+" "+str(width)+" "+str(height))
        #####    f.close()
        #cv2.imwrite("cad1.png", img1)
        self.cad_image1.value = np.squeeze(img1)
        self.cad_image2.value = np.squeeze(img2) 
        self.cad_image3.value = np.squeeze(img3)

        #ax = plt.figure()
       
        #self.pointcloud_scene.value =  self.plot_data
        print("CAD file loaded")
    
    def buttonPlotCurrentScene(self):
        draw = []
        if self.origin:
            frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=10,origin = self.origin)
            draw.append(frame)
        else:
            frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=10)
            draw.append(frame)
        i = 0
        for p in self.scene_points:
            if i==0:
                r = 0
                g = 0
                b = 0
            else:
                r = random.random()
                g = random.random()
                b = random.random()
            c = [r,g,b]
            draw.append(NumpyToPCD(p).paint_uniform_color(c))
            i+=1
        o3d.visualization.draw_geometries(draw)


    def buttonMatchingObject(self):
        """
        matching object in image based on bounding box
        """ 
        tim = time.time()
        if len(self.cad_views)==0:
            print("no cad loaded")
            return

        t_ls = []
        for i in range(len(self.cad_views)):
            views = self.cad_views[i]
            t1  = threading.Thread(target = self.match_object_thread, args = ([views]))
            t1.start()
            t_ls.append(t1)
        
        for t in t_ls:
            t.join()

            
        #self.match_cad = np.squeeze(self.match_cad)
        #self.match_det = np.squeeze(self.match_c)
        print("total matching  time : ", time.time()-tim)
        for i in range(len(self.match_cad)):
            mc = self.match_cad[i]
            md = self.match_det[i]
            for j in range(len(mc)):
                cad_pts = copy.deepcopy(mc[j])
                cad_pts[:,2] = 0
                cad_pts[:,0] = cad_pts[:,0] - np.mean(cad_pts[:,0])
                cad_pts[:,1] = cad_pts[:,1] - np.mean(cad_pts[:,1])
                det_pts = copy.deepcopy(md[j])
                det_pts[:,2] = 0
                #translate in 0
                det_pts[:,0] = det_pts[:,0] - np.mean(det_pts[:,0])
                det_pts[:,1] = det_pts[:,1] - np.mean(det_pts[:,1])
            
                o3d.visualization.draw_geometries([NumpyToPCD(det_pts).paint_uniform_color([0,0,1]), NumpyToPCD(cad_pts).paint_uniform_color([1,0,0])])
        

    def shapeMatcher(self,points1, points2):
        #try:
            #### use different indicator to match the two pointcloud
            points2 = Edge_Detection(points2, 5, 0.001)
            #print("shape indicator")
            #### 1) matchShapes
            points2[:,2] = points2[:,2]*0     

            #img1, points1 = self.hh.PointCloudToImage(points1)
            #img2, points2 = self.hh.PointCloudToImage(points2)
            #img1 = img1.astype(np.uint8)
            #img2 = img2.astype(np.uint8)


            #cam = vcd.Vimba_camera()
            ##v1 = self.hh.ImageToPoints(img1, cam)
            ##v2 = self.hh.ImageToPoints(img2, cam)


        
            #v1 = points1
            #v2 = points2
            #bol = v1[:,2]>0
            #v1 = v1[bol]
            #bol = v2[:,2]>0
            #v2 = v2[bol]
            ##v1 = v1[:,0:2]
            ##v2 = v2[:,0:2]
            Ann, RT1, RT2, am1, bm, error = PCA_alignment(points1, points2)
            print(error)
            #o3d.visualization.draw_geometries([NumpyToPCD(v1).paint_uniform_color([1,0,0]),NumpyToPCD(v2).paint_uniform_color([1,1,0])])
            return error

            ###v2 = alpha_shape(v2,0.5)
            ###v1 = alpha_shape(v1,0.5)

            ###z2 = np.zeros(len(v2))
            ###z1 = np.zeros(len(v1))
            ###v2  =np.asarray([v2[:,0],v2[:,1],z2]).T
            ###v1  =np.asarray([v1[:,0],v1[:,1],z1]).T



            ###v1 = np.array(v1).reshape((-1,1,2)).astype(np.int32)
            ###v2 = np.array(v2).reshape((-1,1,2)).astype(np.int32)


            ##r = int(random.random()*1000)


            ###o3d.visualization.draw_geometries([NumpyToPCD(v1).paint_uniform_color([1,0,0]),NumpyToPCD(v2).paint_uniform_color([1,1,0])])

            ###cv2.imwrite("m1"+str(r)+".jpg", img1)
            ###cv2.imwrite("m2"+str(r)+".jpg", img2)


        


            ####for i in range(len(cnt1)):
            ####    color = (random.random()*255, random.random()*255,random.random()*255)
            ####    cv2.drawContours(img1, cnt1, i, color, 2, cv2.LINE_8, hierarchy1, 0)
            ####for i in range(len(cnt2)):
            ####    color = (random.random()*255, random.random()*255,random.random()*255)
            ####    cv2.drawContours(img2, cnt2, i, color, 2, cv2.LINE_8, hierarchy2, 0)

            ###cv2.imwrite("m1_cnt"+str(r)+".jpg", img1)
            ###cv2.imwrite("m2_cnt"+str(r)+".jpg", img2) 
            ###hull2 = ConvexHull(v2).points[:,0:2].astype(np.int32)
            ###hull1 = ConvexHull(v1).points[:,0:2].astype(np.int32)
        
            ###hull2 =np.squeeze(cv2.convexHull(v2))
            ###hull1 =np.squeeze(cv2.convexHull(v1))

            ##alpha = 0.4
            ##hull1 = alphashape.alphashape(v1, alpha)
            ##hull2 = alphashape.alphashape(v2, alpha)

            ###hull1 =  (np.asarray(hull1.exterior.coords.xy).T).astype(np.int32)
            ###hull2 =  (np.asarray(hull2.exterior.coords.xy).T).astype(np.int32)
            ##hull1 = PolygonPatch(hull1,fill=True).get_verts().astype(np.int32)
            ##hull2 = PolygonPatch(hull2,fill=True).get_verts().astype(np.int32)

            ###invert x y
            ##t = copy.deepcopy(hull1[:,0])
            ##hull1[:,0] =copy.deepcopy(hull1[:,1])
            ##hull1[:,1] = copy.deepcopy(t)
            ##t =copy.deepcopy(hull2[:,0])
            ##hull2[:,0] =copy.deepcopy(hull2[:,1])
            ##hull2[:,1] =copy.deepcopy(t)

            ###t = copy.deepcopy(v1[:,0])
            ###v1[:,0] =copy.deepcopy(v1[:,1])
            ###v1[:,1] = copy.deepcopy(t)
            ###t = copy.deepcopy(v2[:,0])
            ###v2[:,0] =copy.deepcopy(v1[:,1])
            ###v1[:,1] = copy.deepcopy(t)


            ##img11 = cv2.fillPoly(img1, pts=[hull1], color=(255,255,255))
            ##img22 = cv2.fillPoly(img2, pts=[hull2], color=(255,255,255))

            ###labels = skimage.morphology.label(img11)
            ###labelCount = np.bincount(labels.ravel())
            ###background = np.argmin(labelCount)
            ###img11[labels != background] = 255

            ###labels = skimage.morphology.label(img22)
            ###labelCount = np.bincount(labels.ravel())
            ###background = np.argmin(labelCount)
            ###img22[labels != background] = 255


            ##cv2.imwrite("m1_fill"+str(r)+".jpg", img11)
            ##cv2.imwrite("m2_fill"+str(r)+".jpg", img22) 
            ###img111 = cv2.resize(img11, (300, 300))
            ###img222 = cv2.resize(img22, (300, 300))
            ###cv2.imwrite("m1_resize.jpg", img111)
            ###cv2.imwrite("m2_resize.jpg", img222) 
            ##shape = cv2.matchShapes(img11,img22,cv2.CONTOURS_MATCH_I1,0.0)
            ###print("match shape :")
            ###print(shape)
        
            ##cnt1, hierarchy1 = cv2.findContours(img1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            ##cnt2, hierarchy2 = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            ##area1 = cv2.contourArea(cnt1[-1])
            ##area2 = cv2.contourArea(cnt2[-1])

            ##len1 = cv2.arcLength(cnt1[-1],True)
            ##len2 = cv2.arcLength(cnt2[-1],True)

            ##delta_area = abs(area1-area2)/area1
            ###print("delta area : ", delta_area)
            ##delta_len = abs(len1-len2)/len1
            ###print("delta_len :" ,delta_len)
        
        #    return shape,delta_area,delta_len

        #except:
        #    return 1000,1000,1000


    def buttonAlignMatched2(self):
        errors = []
        RT = []
        T = []
        t = time.time()
        for i in range(len(self.match_det)):
            Source = copy.deepcopy(self.match_cad[i])
            
            Target = copy.deepcopy(self.match_det[i])

            #estraggo edges
            Source = Edge_Detection(Source, 5, 0.001)
            #Target = Edge_Detection(Target, 5, 0.001)
           # o3d.visualization.draw_geometries([NumpyToPCD(Source).paint_uniform_color([1,0,0]), NumpyToPCD(Target).paint_uniform_color([0,0,1])])
            Source[:,2] = 0
            Target[:,2] = 0
            

            reg_hl = PC_registration.PointCloudRegistration()
            Source_transf, RT_ls, Tls, error = reg_hl.SparseAlignment(Source,Target,50)
            o3d.visualization.draw_geometries([NumpyToPCD(Source_transf).paint_uniform_color([1,0,0]), NumpyToPCD(Target).paint_uniform_color([0,0,1])])
            errors.append(error)
            RT.append(RT_ls)
            T.append(Tls)
        ind = np.argmin(errors)
        e = errors[ind]
        delta = [abs(errors[i]-e)/e for i in range(len(errors))]
        print("deltas : " , delta)
        ### check if multiple match
        bol = np.asarray(delta)<0.15
        #bol = list(np.where(bol).astype(np.int8))
        #RT_ls = list(np.asarray(RT)[bol])
        #Tls =list(np.asarray(T)[bol])
        self.match_cad =list(np.asarray(self.match_cad)[bol])
        for i in range(len(RT_ls)):
            self.match_cad[i] = reg_hl.Transformation_with_list(self.match_cad[i],RT_ls[i],Tls[i])
            I = np.asarray([[1,0,0],
                            [0,1,0],
                            [0,0,1]])
            #RT_ls.append(I)
            T_ls = [0,0,-min(self.match_cad[i][:,2])]
            self.match_cad[i] = reg_hl.Transformation_with_list(self.match_cad[i],[I],[T_ls])
            self.scene_points.append(copy.deepcopy(self.match_cad[i]))
    
        self.match_det = []

        self.match_cad = []

        print("finished alignement")
        print(time.time()-t)

    def buttonAlignMatched(self):
        
        t = time.time()
        
        reg_hl = PC_registration.PointCloudRegistration()
        for i in range(len(self.match_cad)):
            errors = []
            RT = []
            T = []
            Source = []
            Target = []
            pools = []
            mc = self.match_cad[i]
            md = self.match_det[i]
            for j in range(len(mc)):
                Source.append( copy.deepcopy(mc[j]))
                Target.append( copy.deepcopy(md[j]))
                #pools.append(self.pool)
            res = self.pool.starmap(mp_match_cad,zip(Source,Target))
            for r in res:
                errors.append(r[3])
                print("error : ", r[3])
                RT.append(r[1])
                T.append(r[2])

           # errors = np.asarray(errors)*np.asarray(self.match_boxes)
            print(errors)
            ind = np.argmin(errors)
            e = errors[ind]
            delta = [abs(errors[i]-e)/e for i in range(len(errors))]
            ### check if multiple match, remove huge errors
            bol = np.asarray(delta)<2
            print("deltas : " , delta)
            #bol = list(np.where(bol).astype(np.int8))
            RT_ls = list(np.asarray(RT)[bol])
            Tls = list(np.asarray(T)[bol])
            mc =list(np.asarray(mc)[bol])
            for i in range(len(RT_ls)):
                mc[i] = reg_hl.Transformation_with_list(mc[i],RT_ls[i],Tls[i])
                I = np.asarray([[1,0,0],
                                [0,1,0],
                                [0,0,1]])
                #RT_ls.append(I)
                T_ls = [0,0,-min(mc[i][:,2])]
                mc[i] = reg_hl.Transformation_with_list(mc[i],[I],[T_ls])
                self.scene_points.append(copy.deepcopy(mc[i]))
    
        self.match_det = []
        self.match_boxes = []
        self.match_cad = []

        print("finished alignement")
        print(time.time()-t)
    

    def connect_cameras(self,all_available = True):
        all_available=False
        self.Cameras_connected = []
        if all_available==True:
            print("connecting to all available cameras")
            res = 1
            num = 0
            while res>0:
                cam = vcd.Vimba_camera()
                res = cam.connect(num=num)
                if res>0:
                    print("connected to :", cam.cam.get_id())
                    cam.load_config_file(self.include_dir + "\\DATA\\"+str(cam.cam.get_id())+"_intrinsic.json")   #load if available config file
                    self.Cameras_connected.append(cam)
                num+=1
            return 1
        else:
            for i in range(len(self.Cameras_ID)):
                id = self.Cameras_ID[i]
                cam = vcd.Vimba_camera()
                res = cam.connect(camera_id = id)
                not_conn = []
                if res>0:
                    print("connected to :", cam.cam.get_id())
                    cam.load_config_file(self.include_dir + "\\DATA\\"+ str(cam.cam.get_id())+"_intrinsic.json")   #load if available config file
                    self.Cameras_connected.append(cam)
                    if i==0:
                        self.camera1_connected.value = True
                    elif i==1:
                        self.camera2_connected.value = True
                    elif i==2:
                        self.camera3_connected.value = True
                else:
                    not_conn.append(id)
            
            if len(self.Cameras_connected)==len(self.Cameras_ID):
                print("connected to all listed cameras")

                return 1 
            else:
                print("not connected to camera id : ", not_conn)
                return -1
    


    def buttonCalibrateCamera(self):

        if len(self.Cameras_connected)==0:
            print("No camera Connected, Please Connect to cameras")
            return -1
        else:
            print("which camera you want to calibrate?")
            i = 0
            for c in self.Cameras_connected:
                print(str(c.cam.get_id()) + "  input " + str(i))
                i+=1
            id = int(input("Enter desired number : "))
        camera = self.Cameras_connected[id]
        camera.intrinsic_calibration()


    def extrinsic_calibration(self):

        print('//////////////////////////////////////////')
        print('/// Starting extrinsic Calibration ///')
        print('//////////////////////////////////////////\n')
        while True:
            res = str(input("is acquisition needed? [Y/n]"))
            if res in["n","N"]:
                acquire = False
                break
            elif res in ["y","Y"]:
                acquire = True
                break
            else:
                print("wrong command\n")
                return
        filepath ="/calibration/"
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        #input("Position the Chessboard\n Press Enter to Continue")
        images = []

        for c in self.Cameras_connected:
            image = c.grab_image(filename =filepath+"/image"+str(i)+".png")
            images.append(np.squeeze(image))
            #all_points.append(self.hh.ImageToPoints(images[i],c))
            i+=1
        #all_points = np.concatenate(all_points,axis=0)

        if len(self.Cameras_connected)==0:
            print("No camera Connected, Please Connect to cameras")
            return -1
        else:
            print("which camera you want to calibrate?")
            i = 0
            for c in self.Cameras_connected:
                print(str(c.cam.get_id()) + "  input " + str(i))
                i+=1
            id = int(input("Enter desired number : "))
        camera = self.Cameras_connected[id]

        img = self.hh.stitch_images(images, self.Cameras_connected)

    def change_exposure(self):
        if len(self.Cameras_connected)==0:
            print("No camera Connected, Please Connect to cameras")
            return -1
        else:
            time = float(self.exposure_setpoint.value)
            for c in self.Cameras_connected:
                c.ChangeExposureTime(time)
            return 1 

    def grab_stitched_image(self):
        if len(self.Cameras_connected)==0:
            print("no cameras connected")
            return
        else:
            images = []
            all_points = []
            i = 0
            for c in self.Cameras_connected:
                image = c.grab_image(filename =self.include_dir+"//image"+str(i)+".jpg")
                images.append(np.squeeze(image))
                #all_points.append(self.hh.ImageToPoints(images[i],c))
                i+=1
            #all_points = np.concatenate(all_points,axis=0)
            print("scattato tutto")
            pxmm = self.Cameras_connected[0].pxmm
            img = self.hh.stitch_images(images, self.Cameras_connected)

            #img = np.squeeze(cv2.imread(self.include_dir + "\\DATA\\stitched_multiple.png"))
            #img = img.astype(np.uint8)
            #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            cv2.imwrite(self.include_dir+"//stitched_rme.png",img)
            self.stitched_image.value = img # np.squeeze(images[2])
            self.image = img

    def button_filter_image2(self):
            t = time.time()
            thrs = int(self.threshold_slider.value)
            thrs2 = int(self.threshold_slider2.value)

            back = cv2.imread("D:/RepoGithub/stitched_background_15000exp.png")
            back = cv2.cvtColor( back, cv2.COLOR_BGR2GRAY)
            back = back.astype(np.uint8)
            back = cv2.adaptiveThreshold(back,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thrs,thrs2)

            bol = back<0.1


            #ret, threshtest = cv2.threshold(self.image,thrs, 255, cv2.THRESH_BINARY) 
            thresh = self.image.astype(np.uint8)
            thresh = cv2.medianBlur(thresh,5)
            thresh = cv2.bilateralFilter(thresh ,9,75,75)
            thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thrs,thrs2)
            thresh[bol] = 255
            
            thresh = cv2.bitwise_not(thresh)
            element = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
            thresh = cv2.erode(thresh, element, iterations = 3)
            thresh = cv2.bitwise_not(thresh)
            self.stitched_image.value = thresh
            self.thresh_image = thresh
            print(time.time()-t)
    #def button_filter_image(self):
    #    t = time.time()
    #    #import and remove background
       
    #    img = self.image.astype(np.uint8)
    #    edges = cv2.Canny(img,int(self.threshold_slider.value),int(self.threshold_slider2.value) )
    #    self.stitched_image.value = edges


    def button_filter_image(self):
        t = time.time()
        #import and remove background
        back = cv2.imread(self.include_dir + "\\DATA\\stitched_background_15000exp.png")
        back = cv2.cvtColor( back, cv2.COLOR_BGR2GRAY)
        back = back.astype(np.uint8)
        thrs = int(self.threshold_slider.value)
        thrs1 = int(self.threshold_slider2.value) 
        thresh = self.image.astype(np.uint8)
        #parallel filtering for bigimage
        #dividing image in n_process 
        n_process = 8
        img_ls =  []
        img_ls1 =  []

        bk = []
        t_ls = []
        t1_ls=  []
        for i in range(n_process):
            img_ls.append(back[int(back.shape[0]/n_process*i):int(back.shape[0]/n_process*(i+1)),:])
            img_ls1.append(thresh[int(thresh.shape[0]/n_process*i):int(thresh.shape[0]/n_process*(i+1)),:])
            bk.append(True)
            t_ls.append(thrs)
            t1_ls.append(thrs1)

       
        res = self.pool.starmap(mp_filter,zip(img_ls,img_ls1,t_ls,t1_ls))

        for i in range(n_process):
            thresh[int(thresh.shape[0]/n_process*i):int(thresh.shape[0]/n_process*(i+1)),:] = res[i]
        self.stitched_image.value = thresh
        self.thresh_image = thresh
        print(time.time()-t)
        self.button_find_object()

    def button_find_object2(self):
        thresh = self.thresh_image
        thresh = thresh.astype(np.uint8)

        cam = vcd.Vimba_camera()
        pts = self.hh.ImageToPoints(thresh,cam)
    
        bol = pts[:,2]<0.5
        pts = pts[bol]
        index = np.random.choice(pts.shape[0],200000, replace = False)
        pts = pts[index]
        self.scene_points.append(pts)
        #o3d.visualization.draw_geometries([NumpyToPCD(pts)])
        temp = pts.copy()
        model = cluster.DBSCAN(eps= 10, min_samples = 300)
        model.fit_predict(temp)
        clusters = [[]for n in range(max(model.labels_)+1)]
        contours1 = [[]for n in range(max(model.labels_)+1)]
        for i in range(max(model.labels_)+1):
            bol = np.equal(model.labels_,np.ones((len(pts),))*i,)
            p = pts[bol]
            clusters[i].append(p)
            contours1[i].append(np.asarray([np.asarray(p[:,1]*cam.pxmm).astype(np.uint32),np.asarray(p[:,0]*cam.pxmm).astype(np.uint32)]).T)
        clusters = np.squeeze(clusters)
        contours1 = np.squeeze(np.asarray(contours1))
        self.det_objects = copy.deepcopy(clusters)
        
        contours = cv2.findContours(thresh, cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_NONE)[1]


        #contours = sorted(contours, key=cv2.contourArea,reverse=True) 
        #perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
        #listindex=[i for i in range(1000) if perimeters[i]>perimeters[0]/2]
        
        imgcont = self.thresh_image.copy()
        imgcont = imgcont.astype(np.uint8)

        detected_box = []
        for cnt in contours1:
            #cnt = cnt.astype(np.uint8)
            #cnt = cv2.cvtColor(cnt, cv2.COLOR_BGR2GRAY)
            cnt = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
            box = cv2.minAreaRect(cnt)
            points = cv2.boxPoints(box)
            points = np.int0(points)
            cv2.drawContours(imgcont,[points],0,(0,0,255),20)
        for cnt in clusters:
            cnt = cnt[:,0:2]
            cnt = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
            box = cv2.minAreaRect(cnt)
            box = list(box)
            width = box[1][0]
            height = box[1][1]

            if width>height:
                temp = copy.deepcopy(height)
                height = copy.deepcopy(width)
                width = copy.deepcopy(temp)
            detected_box.append([width,height])
        self.detected_box = detected_box
        self.stitched_image.value = imgcont



    def button_find_object(self):
        thresh = self.thresh_image
        thresh = thresh.astype(np.uint8)
        cam = vcd.Vimba_camera()
        n_process = 8
        img_ls =  []
        shifts = []
        for i in range(n_process):
            img_ls.append(thresh[int(thresh.shape[0]/n_process*i):int(thresh.shape[0]/n_process*(i+1)),:])
            shifts.append(int(thresh.shape[0]/n_process*i))
        res = self.pool.starmap(mp_image_to_points,zip(img_ls,shifts))
    
        pts = np.concatenate(res,axis=0)
        index = np.random.choice(pts.shape[0],50000, replace = False)
        pts = pts[index]
        self.scene_points.append(pts)



        temp = pts.copy()
        #index = np.random.choice(temp.shape[0],50000, replace = False)
        #temp = temp[index]
        t = time.time()
        model = cluster.DBSCAN(eps= 10, min_samples = 30)#, n_jobs= -1)
        model.fit_predict(temp)
        print(time.time()-t)
        clusters = [[]for n in range(max(model.labels_)+1)]
        contours1 = [[]for n in range(max(model.labels_)+1)]
        for i in range(max(model.labels_)+1):
            bol = np.equal(model.labels_,np.ones((len(temp),))*i,)
            p = temp[bol]
            clusters[i].append(p)
            contours1[i].append(np.asarray([np.asarray(p[:,1]*cam.pxmm).astype(np.uint32),np.asarray(p[:,0]*cam.pxmm).astype(np.uint32)]).T)
        clusters = np.squeeze(clusters)
        contours1 = np.squeeze(np.asarray(contours1))
        self.det_objects = copy.deepcopy(clusters)
        
       # contours = cv2.findContours(thresh, cv2.RETR_CCOMP ,cv2.CHAIN_APPROX_NONE)[1]


        #contours = sorted(contours, key=cv2.contourArea,reverse=True) 
        #perimeters = [cv2.arcLength(contours[i],True) for i in range(len(contours))]
        #listindex=[i for i in range(1000) if perimeters[i]>perimeters[0]/2]
        
        imgcont = self.thresh_image.copy()
        imgcont = imgcont.astype(np.uint8)

        detected_box = []
        for cnt in contours1:
            #cnt = cnt.astype(np.uint8)
            #cnt = cv2.cvtColor(cnt, cv2.COLOR_BGR2GRAY)
            cnt = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
            box = cv2.minAreaRect(cnt)
            points = cv2.boxPoints(box)
            points = np.int0(points)
            cv2.drawContours(imgcont,[points],0,(0,0,255),20)
        for cnt in clusters:
            cnt = cnt[:,0:2]
            cnt = np.array(cnt).reshape((-1,1,2)).astype(np.int32)
            box = cv2.minAreaRect(cnt)
            box = list(box)
            width = box[1][0]
            height = box[1][1]

            if width>height:
                temp = copy.deepcopy(height)
                height = copy.deepcopy(width)
                width = copy.deepcopy(temp)
            detected_box.append([width,height])
        self.detected_box = detected_box
        self.stitched_image.value = imgcont


    def buttonRemoveBackground(self):
        #kernel = cv2.getStructuringElement(cv.MORPH_ELLIPSE,(3,3))
        #fgbg = cv.bgsegm.createBackgroundSubtractorGMG()
        back = cv2.imread("D:/RepoGithub/stitched_background_15000exp.png")
        back = cv2.cvtColor( back, cv2.COLOR_BGR2GRAY)
        #imgcont = cv2.cvtColor( self.image, cv2.COLOR_GRAY2BGR)
        #back = cv2.normalize(back,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
        #imgcont = cv2.normalize(self.image,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        back = np.float32(back)
        imgcont = np.float32(self.image)
        rem_back = imgcont - back
        rem_back = np.uint8(rem_back)
        rem_back = cv2.normalize(rem_back,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)


        bol = rem_back<10
        rem_back[bol] = 255
	    

        self.stitched_image.value = rem_back
        self.image = rem_back

        
        #params = cv2.SimpleBlobDetector_Params()
    def buttonRemoveShadows(self):
        #params.blobColor= 255
        #params.filterByColor = True
        plane = self.image
        #dilated_img = cv2.dilate(plane, np.ones((7,7), np.uint8))
        for i in range(3):
            plane = cv2.medianBlur(plane,5)
        
            #self.stitched_image.value = bg_img
        

            plane = cv2.bilateralFilter( plane,9,75,75)
        
        self.stitched_image.value = plane
        self.image = plane

        ## Create a detector with the parameters
        #ver = (cv2.__version__).split('.')
        #if int(ver[0]) < 3 :
        #    detector = cv2.SimpleBlobDetector(params)
        #else : 
        #    detector = cv2.SimpleBlobDetector_create(params)


        ## Detect blobs.
        #keypoints = detector.detect(opening)

        ## Draw detected blobs as green circles.
        ## cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
        ## the size of the circle corresponds to the size of blob

        ##print str(keypoints)

        #img = opening.copy()
        #img = img.astype(np.uint8)
        #im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        #self.stitched_image.value = im_with_keypoints




    def findOrigin(self):
        image = copy.deepcopy(self.thresh_image)
        cam = vcd.Vimba_camera()
        #arUco dictionary selection
        arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
        params = cv2.aruco.DetectorParameters_create()
        corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters = params)
        print(corners)
        corners_mm = np.squeeze(np.asarray(copy.deepcopy(corners)))/cam.pxmm
        corners = np.array(corners).reshape((-1,1,2)).astype(np.int32)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        #image = cv2.aruco.drawDetectedMarkers(image, corners)
        cv2.drawContours(image,[corners],0,(0,255,0),20)
        self.stitched_image.value = image
        self.origin = [np.mean(corners_mm[:,1]),np.mean(corners_mm[:,0]),0]
        print(self.origin)
    
    def match_object_thread(self,views):
        cad_box = []
        match_cad_temp = []
        match_det_temp = []
        for v in views:
                v = v[:,0:2]
                v = np.array(v).reshape((-1,1,2)).astype(np.int32)
                box = cv2.minAreaRect(v)
                box = list(box)
                width = box[1][0]
                height = box[1][1]
                #points = cv2.boxPoints(box)
                #points = np.int0(points)
                #area = cv2.contourArea(points)
                #print(area)
                #perimeter = cv2.arcLength(points,True)
                #print(perimeter)
                if width>height:
                    temp = copy.deepcopy(height)
                    height = copy.deepcopy(width)
                    width = copy.deepcopy(temp)
                cad_box.append([width,height])
                #dx = max(v[:,0])-min(v[:,0])
                ##dy =  max(v[:,0])-min(v[:,0])
                #cad_box.append([dx*dy,dx+dy])

        det_box = self.detected_box

        ## loop on all detected
        thrsh = 0.3
        i=0
        k = 0
        match = []
        for d in det_box:
            k=0
            for c in cad_box:
                #print(str(c[0]) + " " + str(d[0]) + " " + str(c[1]) + " " + str(d[1]) + " " )
                print(str(abs(c[0]-d[0])/c[0]*100) + " " + str(abs(c[1]-d[1])/c[1]*100) )
                if abs(c[0]-d[0])/c[0] < thrsh and abs(c[1]-d[1])/c[1] <thrsh:
                    #we have a match
                    match.append([i,k])
                    #self.match_boxes.append(np.mean([abs(c[0]-d[0])/c[0],abs(c[1]-d[1])/c[1]]))
                    print("we have a match!")
                k+=1
                #print("i : "+ str(i) + " k : " + str(k))
                #print()
            i+=1
        for m in match:
            cad_pts = copy.deepcopy(views[m[1]])
            cad_pts[:,2] = 0
            #translate in 0
            cad_pts[:,0] = cad_pts[:,0] - np.mean(cad_pts[:,0])
            cad_pts[:,1] = cad_pts[:,1] - np.mean(cad_pts[:,1])
            det_pts = copy.deepcopy(self.det_objects[m[0]])
            det_pts[:,2] = 0
            #translate in 0
            det_pts[:,0] = det_pts[:,0] - np.mean(det_pts[:,0])
            det_pts[:,1] = det_pts[:,1] - np.mean(det_pts[:,1])
            match_cad_temp.append(copy.deepcopy(views[m[1]]))
            match_det_temp.append(copy.deepcopy(self.det_objects[m[0]]))
            #o3d.visualization.draw_geometries([NumpyToPCD(det_pts).paint_uniform_color([0,0,1]), NumpyToPCD(cad_pts).paint_uniform_color([1,0,0])])

        #among this matching select based on other features, shapeindicators

        points1  = []
        points2 = []
           
        for i in range(len(match_cad_temp)):
            points1.append(match_det_temp[i])
            points2.append(match_cad_temp[i])
        #parallel computation
        shape = self.pool.starmap(mp_shape_match,zip(points1,points2))
        #    shape.append(s)
        #    lenght.append(l)
        #    area.append(a)


        ind = np.argmin(shape)
        s = shape[ind]
        delta = [abs(shape[i]-s)/s for i in range(len(shape))]
        print(delta)
        #print(lenght)
        bol = np.asarray(delta)<0.3
        match_cad_temp = list(np.asarray(match_cad_temp)[bol])
        match_det_temp = list(np.asarray(match_det_temp)[bol])
            
        self.match_cad.append(match_cad_temp)
        self.match_det.append(match_det_temp)

            #print(time.time()-t)



def mp_filter(back, thresh,thrs,thrs1):
        """
        input: portion of the image that needs to be filtered
        output: filtered portion
        """
   
        #thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
        #thresh = back.astype(np.uint8)
        back = cv2.adaptiveThreshold(back,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thrs,thrs1)
        bol = back<0.1
        
        thresh = cv2.medianBlur(thresh,5)
        thresh = cv2.bilateralFilter(thresh ,9,75,75)
        thresh = cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,thrs,thrs1)

        thresh[bol] = 255


        thresh = cv2.bitwise_not(thresh)
        element = cv2.getStructuringElement(cv2.MORPH_RECT, (4,4))
        thresh = cv2.erode(thresh, element, iterations = 3)
        thresh = cv2.bitwise_not(thresh)
        return thresh

def mp_image_to_points(image,shift):
        cam = vcd.Vimba_camera()
        hh =  helper.Helperhandler()
        pts = hh.ImageToPoints(image,cam,shift=shift) 
        print(len(pts))
        bol = pts[:,2]<0.5
        pts = pts[bol]
        #index = np.random.choice(pts.shape[0],int(200000/8), replace = False)
        #pts = pts[index]
        return pts


def mp_match_cad(Source,Target):
        #estraggo edges
        Source = Edge_Detection(Source, 5, 0.001)
        #Target = Edge_Detection(Target, 5, 0.001)
        # o3d.visualization.draw_geometries([NumpyToPCD(Source).paint_uniform_color([1,0,0]), NumpyToPCD(Target).paint_uniform_color([0,0,1])])
        Source[:,2] = 0
        Target[:,2] = 0
        reg_hl = PC_registration.PointCloudRegistration()
        Source_transf, RT_ls, Tls, error = reg_hl.SparseAlignment(Source,Target,50)
        #o3d.visualization.draw_geometries([NumpyToPCD(Source_transf).paint_uniform_color([1,0,0]), NumpyToPCD(Target).paint_uniform_color([0,0,1])])
        return Source_transf, RT_ls, Tls, error



def mp_shape_match(points1,points2):
           #### use different indicator to match the two pointcloud
    points2 = Edge_Detection(points2, 5, 0.001)
    points2[:,2] = points2[:,2]*0
    Ann, RT1, RT2, am1, bm, error = PCA_alignment(points1, points2)
    print(error)

    return error





if __name__ == "__main__":   pyforms.start_app( application )

