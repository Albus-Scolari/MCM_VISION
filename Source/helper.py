#import STEPAnalyzer
#import STEPMesher
#from numba import njit, jit
from tkinter.constants import W
from Source import STEPMesher, STEPAnalyzer
import open3d as o3d
from Source.UTILS.pcd_numpy_utils import *
import numpy as np
import cv2
import math
import numpy_indexed as npi
import copy
import time

class Helperhandler:

	def LoadSTEP(self,filepath,n):
		"""
		load the step model from file and convert it to 
		pointcloud (numpy array Nx3)

		"""
		sa = STEPAnalyzer.StepAnalyzer()
		job = sa.find_axis_color(filepath,[[0,1,0],[1,0,0]])
		if job.size==0:
			job=np.asarray([[0,0,0],[0,0,0]])
		points, mesh = STEPMesher.meshwithSTL(filepath, 5000,n = n, save_file='')
		#points = STEPMesher.meshwithOCC(filepath, save_file='')
		
		#o3d.visualization.draw_geometries([NumpyToPCD(points)])
		return points, job, mesh


	def PointCloudToImage(self,points,pxmm, savefile = ''):
		"""
		convert pointcloud in a depth map image in the desired xy plane
		"""
		pixel_size = 1
		##quantize data in xy
		points_q = np.round(np.array(points/pixel_size)).astype(int)
		x_list = points_q[:,0]
		y_list = points_q[:,1]
		x_max = x_list.max()
		x_min = x_list.min()
		y_max = y_list.max()
		y_min = y_list.min()
		points_q[:,1] = points_q[:,1]-y_min 
		points_q[:,0] = points_q[:,0]-x_min 
		xrange = x_max-x_min
		yrange = y_max-y_min
		Z = points_q[:,2]
		#print("max Z Point : ", max(Z), "min Z points : ", min(Z))
		#f = open(filepath + "z_data.txt", 'w')
		#f.write("Max Z Point : " + str(max(Z))
		#        + "\nMin Z Point : " + str(min(Z))+
		#            "\nmm per greyscale : " + str((max(Z)-min(Z))/256))
		#f.close()
		Z = np.round(np.array(Z-min(Z))*255/(max(Z)-min(Z)+1))
		grid_xy = np.zeros((xrange+1, yrange+1))
		count = 0
		for i in range(len(points_q[:,0])):
			if grid_xy[points_q[i,0],points_q[i,1]] == 0:
					grid_xy[points_q[i,0],points_q[i,1]] = 255 #Z[i]
			else: 
				grid_xy[points_q[i,0],points_q[i,1]] = max(Z[i],grid_xy[points_q[i,0],points_q[i,1]])
				count+=1
		if savefile!='':
			cv2.imwrite(savefile,grid_xy)

		im = cv2.resize(grid_xy, (int(grid_xy.shape[1]*pxmm), int(grid_xy.shape[0]*pxmm)))
		return im, points_q*pxmm
		#Image.fromarray((grid_xy).astype('uint8'), mode='L').save(filepath+'foto.png')
		##print(count)

	def points_to_image(self, points):
		"""
		points coordinates in pixel
		return image in pixel
		"""
		#voxel downsampling to remove point
		#
		xs = points[:,1]
		ys = points[:,0]
		ps =  points[:,2]
		
		xmax = math.ceil(max(xs))
		ymax = math.ceil(max(ys))
		img_size = [ymax+1, xmax+1]
		#img = np.zeros(img_size)

		coords = np.stack((ys, xs))
		abs_coords = np.ravel_multi_index(coords.astype(int), img_size)
		img = np.bincount(abs_coords,weights=ps , minlength=img_size[0]*img_size[1])
		#img = img.reshape(img_size)
		#np.put(img,abs_coords, ps, mode="wrap")
		img = img.reshape(img_size)
		
		return img

	
	def points_to_image2(self, points,pxmm,isCad = False):
		"""
		points coordinates in pixel
		return image in pixel
		"""
		#voxel downsampling to remove point
		#
		
		#img = np.zeros(img_size)
		#if isCAD:
		points = np.round(points*pxmm)  #mm2px
		ps =  points[:,2]

		unique, mean = npi.group_by(points[:,0:2]).mean(ps)
		xs = unique[:,0]
		ys = unique[:,1]
		#
		xmin = int(np.min(xs))
		ymin = int(np.min(ys))
		if xmin<0:
			xs = xs-xmin
		if ymin<0:
			ys = ys-ymin
		coords = np.stack((xs, ys))
		xmax = int(np.max(xs))
		ymax = int(np.max(ys))
		img_size = [xmax+1,ymax+1]
		abs_coords = np.ravel_multi_index(coords.astype(np.int32), img_size)
		average_values = np.bincount(abs_coords,weights = mean,minlength=img_size[0]*img_size[1] )
		img = average_values.reshape(img_size)
		img = img.astype(np.uint8)
		if isCad == False:
			bol = np.where(img==0)
			img[bol] = 128
		return img

	def ImageToPoints(self, img, cam, shift=0):
		img = np.asarray(img)
		# pxmm = 15
		#points = np.zeros((img.shape(0)*img.shape(1)))
		points_z = np.reshape(img, (img.shape[0]*img.shape[1],))
		xy_coords = np.flip(np.column_stack(np.where(img>= 0)), axis=1)
		points = np.asarray([xy_coords[:,1], xy_coords[:,0], points_z]).T
		if shift!=0:
			points[:,0] = points[:,0]+shift
		points = points/cam.pxmm
		points[:,1] = points[:,1]+cam.x_coord
		points[:,0] = points[:,0]+cam.y_coord
		#temp = copy.deepcopy(points[:,0])
		#points[:,0] = copy.deepcopy(points[:,1])
		#points[:,1] = temp
		#o3d.visualization.draw_geometries([NumpyToPCD(points)])
		#print("ciao")
		return points

	def blob_identification(self,im):
		detector = cv2.SimpleBlobDetector()
		keypoints = detector.detect(im)
		im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
		return im_with_keypoints

	def stitch_images(self,images, cams, scale_f = 1):

		n_images=[]
		if scale_f>1:
			for im in images:
				base = copy.deepcopy(im)
				im= cv2.resize(im, (int(im.shape[1]/scale_f), int(im.shape[0]/scale_f)))
				h, w = im.shape
				t,base = cv2.threshold(base,255,255,cv2.THRESH_BINARY)
				#img1= cv2.resize(base, (int(base.shape[1]/5), int(base.shape[0]/5)))
				#cv2.imshow("foto",img1)
				#cv2.waitKey(0)
				#put in base the image resized at the center
				hh,ww = base.shape
				h, w = im.shape
				yoff = round((hh-h)/2)
				xoff = round((ww-w)/2)
				result = base.copy()
				result[yoff:yoff+h, xoff:xoff+w] = im
				n_images.append(result)
			images = n_images
		height = int(images[0].shape[0]*3)
		width = int(images[0].shape[1]*1.2)
		blank_image = np.zeros((height,width), np.uint8)
		
		#create new images with offset
		i=0
		n_ims = []
		t = time.time()
		for im in images:
			temp = copy.deepcopy(blank_image)
			#rotate image
			im = self.rotate_image(im,cams[i].rot)
			y_offset = int(cams[i].x_coord * cams[i].pxmm)
			x_offset = int(cams[i].y_coord * cams[i].pxmm)
			temp[y_offset:y_offset+im.shape[0], x_offset:x_offset+im.shape[1]] = copy.deepcopy(im)
			#bol = temp<0.0001
			
			#temp = temp[bol]
			n_ims.append(np.int16(temp))
			i+=1
		#print(time.time()-t)
		#overlay images
		t = time.time()
		bol1 = (n_ims[0]>0.001)		
		a = (n_ims[0] + n_ims[1])/2*bol1
		bol2  =(n_ims[1]>0.001) 
		a = a*bol2
		temp1 = n_ims[0] + n_ims[1]-a
		temp1 = np.int16(temp1)
		bol1 = (temp1>0) 
		b = (temp1 + n_ims[2])/2*bol1
		bol2 = (n_ims[2]>0) 
		b = b*bol2
		temp1 =np.int16(temp1)
		fin_im = temp1 + n_ims[2] -b
		#fin_im = n_ims[0] + n_ims[1]+ n_ims[2]
		fin_im  = fin_im.astype(np.uint8)
		out=cv2.transpose(fin_im)
		out=cv2.flip(out,flipCode=0)
		#bol = np.where(out==0)
		#out[bol] = 0
		out = cv2.normalize(out,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
		#print(time.time()-t)

		return out

	def rotate_image(self,image, angle):
		  image_center = tuple(np.array(image.shape[1::-1]) / 2)
		  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
		  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
		  return result

	#		
#		NX = len(imgs)
#		for i in range(len(imgs):#N+1):
#			img = imgs[i]
#			[rows, cols] = img.shape #In pixel
            
#			#if self.CalibrationFilePath != "":
#			#	img = cv2.undistort(img, M, D)
#			if i == 0:
#				#if mm2px == 0: mm2px = cols / FieldOfView[0]
#				mm2px = 15
#				minX = 1e100
#				maxX = -1e100
				
#				#Z is 0 in local coordinate system
#				for j in range(len(points)):#(N):
#					minX = min(minX, points[j][0])
#					maxX = max(maxX, points[j][0])
#					#minY = min(minY, self.points[j][1])
#					#maxY = max(maxY, self.points[j][1])

#					DX = 0
#					if NX > 1:
#						DX = (maxX-minX)/(NX-1) # mm
#					DY = 0
#					if NY > 1: 
#						DY = (maxY-minY)/(NY-1) # mm
#					DXpx = round(DX * mm2px) #1835 #px
#					DYpx = round(DY * mm2px) #1450 #px
#					RES = np.ones((rows+DYpx*(NY-1),cols+DXpx*(NX-1)),dtype=np.uint8)*255
#					[nn, mm] = RES.shape

#				#Pay attention: images location are acquired with a canonical XY reference system (X positive right and Y up)
#				#               images are X positive right and Y down
#				xx = points[i-1][0]
#				yy = points[i-1][1]

#				borderPX = self.pxImageCrop
#				borderMX = self.pxImageCrop
#				borderPY = self.pxImageCrop
#				borderMY = self.pxImageCrop

#				if abs(xx-minX) < FrankCommons.tol:
#					borderMX = 0
#				if abs(xx-maxX) < FrankCommons.tol:
#					borderPX = 0
#				if abs(yy-minY) < FrankCommons.tol:
#					borderPY = 0
#				if abs(yy-maxY) < FrankCommons.tol:
#					borderMY = 0

#				x1 = 0 + borderMX
#				if DX != 0:
#					x1 = round((xx - minX) / DX * DXpx) + borderMX
#				y2 = nn - borderPY
#				if DY != 0:
#					y2 = nn - round((yy - minY) / DY * DYpx) - borderPY
#				x2 = x1 + cols - borderMX - borderPX
#				y1 = y2 - rows + borderMY + borderPY
#				comp = np.array([img[borderMY:rows-borderPY, borderMX:cols-borderPX], RES[y1:y2, x1:x2]])
#				RES[y1:y2, x1:x2] = comp.min(axis=0)
            
#				del img

#			#if self.pxImageCrop > 0: RES = RES[self.pxImageCrop:mm-self.pxImageCrop,self.pxImageCrop:nn-self.pxImageCrop]
#			if SaveOutput:
#				cv2.imwrite(FilePath + "/Stitched_Out.tiff", RES)

#			return RES
## def points_to_image_torch(xs, ys, ps, sensor_size=(180, 240)):
## 	xt, yt, pt = torch.from_numpy(xs), torch.from_numpy(ys), torch.from_numpy(ps)
#	# 	img = torch.zeros(sensor_size)
#	# 	img.index_put_((yt, xt), pt, accumulate=True)
#	# 	return img

