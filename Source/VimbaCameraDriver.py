import numpy as np
import vimba as vb
import cv2
import OCC
from typing import Optional
import yaml
import os
import matplotlib.pyplot as plt
import open3d as o3d
from Source.UTILS.pcd_numpy_utils import *
import json
from numpyencoder import NumpyEncoder

def listDevices():
    "Checks the available cameras and returns a list of camera objects for further use."
    with vb.Vimba.get_instance() as vimba:
        return vimba.get_all_cameras()

"Camera class to use for every camera-related operation/feature like acquisition, set parameters, open/close, ecc.."
class Vimba_camera:
    def __init__(self) -> None:
        self.cam = object
        self.config_loaded = False
        self.mtx = 0  # coefficients computed with intrinsic calibration
        self.dist = 0 #
        self.x_coord = 0
        self.y_coord = 0
        self.z_coord=0
        self.rot = 0
        self.pxmm = 301.5/20
        self.x_length_px = 0
        self.y_length_px= 0 
    def ChangeExposureTime(self, time):
        try:
            with vb.Vimba.get_instance():
                with self.cam as cm:
                    exposure_time = cm.ExposureTime
                    c_time = exposure_time.get()
                    print("current exposure time is : ", c_time)
                    exposure_time.set(time)
                    print("new exposure time is : ", time)
            return 1
        except:
            print("unexpected error occurred changing exposure time")
            return -1
    def connect(self,camera_id=None, num = None):
        with vb.Vimba.get_instance() as vimba:                         #
            if camera_id:
                try:
                    self.cam = vimba.get_camera_by_id(camera_id)

                    return 1 
                except vb.VimbaCameraError:
                    print('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
                    return -1
            else:
                cams = vimba.get_all_cameras()
                if not cams:
                    print('No Cameras accessible. Abort.')
                    return -1
                try:
                    self.cam = cams[num]
                    return 1 
                except:
                    return -1

    #def disconnect(self,camera_id=None, num = None):
    #    with vb.Vimba.get_instance() as vimba:                         #
    #        if camera_id:
    #            try:
    #                self.cam = vimba.get_camera_by_id(camera_id)

    #                return 1 
    #            except vb.VimbaCameraError:
    #                print('Failed to access Camera \'{}\'. Abort.'.format(camera_id))
    #                return -1
    #        else:
    #            cams = vimba.get_all_cameras()
    #            if not cams:
    #                print('No Cameras accessible. Abort.')
    #                return -1
    #            try:
    #                self.cam = cams[num]
    #                return 1 
    #            except:
    #                return -1

    def grab_image(self,  filename = ''):
        with vb.Vimba.get_instance():
            with self.cam as cm:
                frame = cm.get_frame()
                frame.convert_pixel_format(vb.PixelFormat.Mono8)
                if filename != '':
                    cv2.imwrite(filename, frame.as_opencv_image())
                    print("saving image" + filename)
                img =  frame.as_opencv_image()
                #if self.config_loaded:
                #    undist = cv2.undistort(img, self.mtx, self.dist, None, self.mtx)
                #    return undist
                #else:
                print("disotrted image")
                return img

    def load_config_file(self,filename):
        try : 
            with open(filename,"r") as file:
                data = json.load(file)
            self.mtx = np.asarray(data.get("mtx"))
            self.dist = np.asarray(data.get("dist"))
            self.x_coord = data.get("x_coord")
            self.y_coord = data.get("y_coord")
            self.z_coord = data.get("z_coord")
            self.rot = data.get("rot")
            self.x_length_px = data.get("x_length_px")
            self.y_length_px = data.get("y_length_px")
            print("config file found, loaded succesfully\n", data)
            self.config_loaded = True
            return 1
        
        except:
            print("config file not found")
            return -1
    def write_to_config_file(self,filename,p,value):
        try : 
            with open(filename) as file:
                data = yaml.safe_load(file)
            with open(filename,"w") as file:
                data[p] = value
                yaml.dump(data,file)

            print("config file found, modified succesfully\n", data)
            return 1
        
        except yaml.YAMLError as e:
            print(e)
            return -1
    
    def intrinsic_calibration(self):
        """
        porcedure to perform intrinsic calibration
        need chessboard and camera to be moved in different positions
        """
        print('//////////////////////////////////////////')
        print('/// Starting Intrinsic Calibration ///')
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
        filepath ="/calibration/"+str(self.cam.get_id())
        if not os.path.isdir(filepath):
            os.mkdir(filepath)
        #input("Position the Chessboard\n Press Enter to Continue")
        images = []
        
        for i in range(20):
            if acquire:
                print('/// Move Chessboard in position : '+str(i)+'///\n')
                input('/// press Enter to grab Image ///\n')
                img = self.grab_image( filename =filepath+ '/intrinsic_calib'+str(i)+'.png')

            else:
                img = cv2.imread(filepath+ '/intrinsic_calib'+str(i)+'.png')
                # Convert to grayscale
            gray = img #cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow("calibration image " + str(i), gray)
            # cv2.waitKey(0)

            images.append(gray)
        square_size = 20
        nx= 6
        ny = 4
        objpoints = []
        imgpoints = []
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2)
        objp = objp * square_size
    # Find the chessboard corners
        for k in range(len(images)):
            i = images[k]
            print("starting images inspection : " + str(k))
            
            # objp[:,:,] =  np.mgrid[0:nx,0:ny].T.reshape(-1,2)
            chessboard_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE
            ret, corners = cv2.findChessboardCorners(i, (nx, ny), flags=chessboard_flags)
            # If found, draw corners
            print("finished inspection : "+ str(k))
            if ret == True:
                print("found something")
                # Draw and display the corners
                imgpoints.append(corners)
                objpoints.append(objp)
                cv2.drawChessboardCorners(i, (nx, ny), corners, ret)
                cv2.imwrite(filepath+"/detected_corners"+str(k)+".png", i)
                #plt.imshow(i)
                #print(corners)
        print("starting paramenter computation")
        #print(i.shape[1:])
        #print(i.shape[::-1])
        #print(i.shape[:-1])


        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, i.shape[:-1], None, None)
        print("finished parameter computation")
        print(ret)
        print(mtx)
        print(dist)
        print("saving do config file")
        filename = filepath + "/"+ str(self.cam.get_id()) + "_intrinsic.json"
        f = open(filename, "w")

        mtx = list(mtx)
        dist = list(dist)
        print(type(mtx))
        print(type(dist))
        intrinsic_params = {"mtx" : mtx, "dist" : dist}
        json.dump(intrinsic_params,f,cls=NumpyEncoder)

        pass

        
    
        #rot = self.rot




"metodi per semplificare acquisizioni e altre cose a prescindere da hardware"

#print(listDevices())