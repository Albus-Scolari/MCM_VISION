#import threading
#from pycuda import driver
import alphashape
import numpy as np
import open3d as o3d
from Source.UTILS.pcd_numpy_utils import *
from itertools import product
import json
#from descartes import PolygonPatch
#import os
#import sys
#import pandas as pd
#import numpy as np
#from descartes import PolygonPatch
#import matplotlib.pyplot as plt
#sys.path.insert(0, os.path.dirname(os.getcwd()))
#import alphashape
import copy
import cv2
from sklearn.decomposition import PCA
   
import Source.REGISTRATION.PC_registration as PC_registration
from pykdtree.kdtree import KDTree
from mpl_toolkits.mplot3d import Axes3D  
# Axes3D import has side effects, it enables using projection='3d' in add_subplot
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.interpolate import Rbf
from Source import helper


#def value_func_3d(x, y, z):
#    return 2 * x + 3 * y - z
#x = np.linspace(0, 4, 5)
#y = np.linspace(0, 5, 6)
#z = np.linspace(0, 6, 7)
#points = (x, y, z)
#values = value_func_3d(*np.meshgrid(*points, indexing='ij'))
#point = np.array([2.21, 3.12, 1.15])
##print(interpn(points, values, point))
#def combine_half_images(img1,img2):
#        temp = copy.deepcopy(img2)
#        temp[:,int(temp.shape[0]/3*2):-1] = img1[:,int(temp.shape[0]/3*2):-1]
#        return temp

#def stitch_images(images, x_coords,y_coords,pxmm):
#		height = int(images[0].shape[0]*3)
#		width = int(images[0].shape[1]*1.2)
#		blank_image = np.zeros((height,width), np.uint8)
#		i=0
#		n_ims = []
#		for im in images:
#			temp = copy.deepcopy(blank_image)
#			y_offset = int(x_coords[i] * pxmm)
#			x_offset = int(y_coords[i] * pxmm)
#			temp[y_offset:y_offset+im.shape[0], x_offset:x_offset+im.shape[1]] = im
#			n_ims.append(np.int16(temp))
#			i+=1
#		bol1 = (n_ims[0]>0.001)		
#		a = (n_ims[0] + n_ims[1])/2*bol1
#		bol2  =(n_ims[1]>0.001) 
#		a = a*bol2
#		temp1 = n_ims[0] + n_ims[1]-a
#		temp1 = np.int16(temp1)
#		bol1 = (temp1>0) 
#		b = (temp1 + n_ims[2])/2*bol1
#		bol2 = (n_ims[2]>0) 
#		b = b*bol2
#		temp1 =np.int16(temp1)
#		#temp1 = cv2.normalize(temp1,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#		fin_im = temp1 + n_ims[2] -b
#		fin_im  = fin_im.astype(np.uint8)
#		out=cv2.transpose(fin_im)
#		out=cv2.flip(out,flipCode=0)

#		out = cv2.normalize(out,None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
#		return out

#path = "C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\calibrazione_piano\\0\\"
#path1 = "C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\calibrazione_piano\\60\\"
#hh =  helper.Helperhandler()
#x_coords = []
#y_coords=[]
#for name in ["DEV_1AB22C010C0B_intrinsic.json","DEV_1AB22C010C0C_intrinsic.json", "DEV_1AB22C010C0A_intrinsic.json"]:
#    filename = path+name
#    with open(filename,"r") as file:
#        data = json.load(file)
#        x_coords.append(data.get("x_coord"))
#        y_coords.append(data.get("y_coord"))
######## 
#### rescaling imagesin order to maintain
pxmm = 301.5/20
#images=[]
#for i in range(3):
#        img1 = cv2.imread(path1+"image1_"+str(i)+".png")
#        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#        img1 = img1.astype(np.uint8)
#        images.append(img1)

#n_images= []
#scale_f = 1.063
#### base image is the same but all black
#for im in images:
#    base = copy.deepcopy(im)
#    im= cv2.resize(im, (int(im.shape[1]/scale_f), int(im.shape[0]/scale_f)))
#    h, w = im.shape
#    t,base = cv2.threshold(base,255,255,cv2.THRESH_BINARY)
#    #img1= cv2.resize(base, (int(base.shape[1]/5), int(base.shape[0]/5)))
#    #cv2.imshow("foto",img1)
#    #cv2.waitKey(0)
#    #put in base the image resized at the center
#    hh,ww = base.shape
#    h, w = im.shape
#    yoff = round((hh-h)/2)
#    xoff = round((ww-w)/2)
#    result = base.copy()
#    result[yoff:yoff+h, xoff:xoff+w] = im
#    #img1= cv2.resize(result, (int(result.shape[1]/5), int(result.shape[0]/5)))
#    #cv2.imshow("foto",img1)
#    #cv2.waitKey(0)
#    n_images.append(result)
#img11 = stitch_images(n_images, x_coords, y_coords, pxmm)
#images=[]
#for i in range(3):
#        img1 = cv2.imread(path1+"image2_"+str(i)+".png")
#        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#        img1 = img1.astype(np.uint8)
#        images.append(img1)
#n_images= []
#### base image is the same but all black
#for im in images:
#    base = copy.deepcopy(im)
#    im= cv2.resize(im, (int(im.shape[1]/scale_f), int(im.shape[0]/scale_f)))
#    h, w = im.shape
#    t,base = cv2.threshold(base,255,255,cv2.THRESH_BINARY)
#    #img1= cv2.resize(base, (int(base.shape[1]/5), int(base.shape[0]/5)))
#    #cv2.imshow("foto",img1)
#    #cv2.waitKey(0)
#    #put in base the image resized at the center
#    hh,ww = base.shape
#    h, w = im.shape
#    yoff = round((hh-h)/2)
#    xoff = round((ww-w)/2)
#    result = base.copy()
#    result[yoff:yoff+h, xoff:xoff+w] = im
#    #img1= cv2.resize(result, (int(result.shape[1]/5), int(result.shape[0]/5)))
#    #cv2.imshow("foto",img1)
#    #cv2.waitKey(0)
#    n_images.append(result)

#img22 = stitch_images(n_images, x_coords, y_coords, pxmm)
#img1 = combine_half_images(img11,img22)
##img1= cv2.resize(img1, (int(img1.shape[1]/7), int(img1.shape[0]/7)))
##cv2.imshow("foto",img1)
##cv2.waitKey(0)


#cv2.imwrite(path1+"\\stitched_scaled.png",img1)



img1 = cv2.imread("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\stitched_new.png")

### import origin 
with open("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\CONFIG\\machine_origin.txt") as f:
    lines = f.readlines()
    origin = np.asarray(lines[0].split(";")).astype(float)
    rotation = np.asarray([lines[1].split(";"),lines[2].split(";"),lines[3].split(";")]).astype(float)
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
reg_hl = PC_registration.PointCloudRegistration()
rot = [[0,1,0],
       [-1,0,0],
       [0,0,1]]
      #rotation -90° asse z
I = np.asarray([[1,0,0],
                [0,1,0],
                [0,0,1]])

temp = copy.deepcopy(img1)
gray = copy.deepcopy(img1)
gray = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
t, gray = cv2.threshold(gray,60,255,cv2.THRESH_BINARY)

#gray = cv2.medianBlur(gray,5)
#gray = cv2.medianBlur(gray,5)
#gray = cv2.medianBlur(gray,5)
#gray = cv2.medianBlur(gray,5)
#gray = cv2.medianBlur(gray,5)

img1= cv2.resize(gray, (int(temp.shape[1]/10), int(temp.shape[0]/10)))
cv2.imshow("foto",img1)
cv2.waitKey(0)


#gray = cv2.medianBlur(gray,5)
#gray = cv2.medianBlur(gray,5)
rows = gray.shape[0]



circles = cv2.HoughCircles(copy.deepcopy(gray), cv2.HOUGH_GRADIENT, 1, 100,
                            param1=20, param2=10,
                            minRadius=30, maxRadius=55)
    
print(len(circles))
centers = []
diameters = []
centers_mm = []
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
       # centers_mm.append(list(center))
        c_mm = np.asarray([i[1], i[0]])/pxmm
        #centers_mm.append(copy.deepcopy(c_mm))
        z = np.zeros((1,3))
        z[:,0] = float(c_mm[0])
        z[:,1] = float(c_mm[1])
        centers_mm.append(copy.deepcopy(z))
        c_mm =  reg_hl.Transformation_with_list(copy.deepcopy(z), [I,rot],[-origin,[0,0,0]])
        centers.append(c_mm)
        # circle center
        cv2.circle(temp, center, 1, (0, 100, 100), 3)
        # circle outline
        radius = i[2]
        if c_mm[0,0]>-5 and c_mm[0,0]<490 and c_mm[0,1]>-80 and c_mm[0,1]<80:
            diameters.append(radius/pxmm*2)
        cv2.circle(temp, center, radius, (255, 0, 255), 3)
centers = np.squeeze(centers)

print(np.mean(diameters))

img1= cv2.resize(temp, (int(temp.shape[1]/10), int(temp.shape[0]/10)))
cv2.imshow("foto",img1)
cv2.waitKey(0)
cv2.imwrite("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\piano_marcato_circles.png",temp)

#find new coordinate system axis
#find origing
centers_mm = np.asarray(centers_mm).reshape(len(centers_mm),3)
tree = KDTree(centers)
distances, index = tree.query(np.asarray([0,0,0]).reshape((1,3)), k=1)
new_origin = centers_mm[int(index)]
#x axis
pointsx = copy.deepcopy(centers)
pointsx_mm = copy.deepcopy(centers_mm)
bol = pointsx[:,0]>-230
pointsx = pointsx[bol]
pointsx_mm = pointsx_mm[bol]
bol = pointsx[:,0]<230
pointsx = pointsx[bol]
pointsx_mm = pointsx_mm[bol]

bol = pointsx[:,1]<5
pointsx = pointsx[bol]
pointsx_mm = pointsx_mm[bol]

bol = pointsx[:,1]>-5
pointsx = pointsx[bol]
pointsx_mm = pointsx_mm[bol]

pointsx = pointsx[np.argsort(pointsx[:,0])]
pointsx_mm = pointsx_mm[np.argsort(pointsx[:,0])]

pca=PCA(n_components=1)
pca.fit(pointsx_mm)
v2 = pca.components_[0]

pointsy = copy.deepcopy(centers)
pointsy_mm = copy.deepcopy(centers_mm)

bol = pointsy[:,0]>-5
pointsy = pointsy[bol]
pointsy_mm = pointsy_mm[bol]


bol = pointsy[:,0]<5
pointsy = pointsy[bol]
pointsy_mm = pointsy_mm[bol]

bol = pointsy[:,1]<195
pointsy = pointsy[bol]
pointsy_mm = pointsy_mm[bol]

bol = pointsy[:,1]>-50
pointsy = pointsy[bol]
pointsy_mm = pointsy_mm[bol]

pointsy = pointsy[np.argsort(pointsy[:,1])]
pointsy_mm = pointsy_mm[np.argsort(pointsy[:,1])]

pca=PCA(n_components=1)
pca.fit(pointsy_mm)
v1 = pca.components_[0]
n = np.linalg.norm(v1)
v1 = v1/n
n = np.linalg.norm(v2)
v2 = -v2/n
#o3d.visualization.draw_geometries([NumpyToPCD(pointsx).paint_uniform_color([1,0,0]),NumpyToPCD(pointsy).paint_uniform_color([0,0,1]),frame])

vectorx  = np.asarray([v1*k for k in range(1000)])
vectory = np.asarray([v2*k for k in range(1000)])

frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
o3d.visualization.draw_geometries([NumpyToPCD(vectorx).paint_uniform_color([1,0,0]),NumpyToPCD(vectory).paint_uniform_color([0,0,1]),frame,
                                   NumpyToPCD(pointsx_mm).paint_uniform_color([1,0,0]),NumpyToPCD(pointsy_mm).paint_uniform_color([0,0,1])])




v3 = [0,0,1]

RT =  [[v1[0], v2[0], v3[0]], [v1[1], v2[1], v3[1]], [v1[2], v2[2], v3[2]]]

new_rotation = np.asarray(RT)

#f = open("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\CONFIG\\machine_origin.txt","w" )
#f.write(str(new_origin[0]) + ";"+str(new_origin[1]) + ";"+str(new_origin[2]))
#f.write("\n")
#f.write(str(RT[0][0]) + ";"+str(RT[0][1]) + ";"+str(RT[0][2]) + "\n"+
#        str(RT[1][0]) + ";"+str(RT[1][1]) + ";"+str(RT[1][2]) + "\n"+
#        str(RT[2][0]) + ";"+str(RT[2][1]) + ";"+str(RT[2][2]) )
#f.close()

#filterout centers outside fov
bol = centers[:,0]>-210
centers = centers[bol]
bol = centers[:,0]<210
centers = centers[bol]
bol = centers[:,1]<190
centers = centers[bol]
bol = centers[:,1]>-35
centers = centers[bol]
centers = centers[np.argsort(centers[:,0])]
nc = copy.deepcopy(centers)
for j in range(40):
        c = centers[j*21:(j+1)*21,:]
        nc[j*21:(j+1)*21,:] = c[np.argsort(c[:,1])]
centers = nc
### compensation table creation
x = list(range(-200,200,10))
y = list(range(-30,180,10))
xy = np.array(list(product(x,y)))
xyz = np.zeros((len(xy),3))
xyz[:,0] = xy[:,0]
xyz[:,1] = xy[:,1]

#tree = KDTree(xyz)
#distances, index = tree.query(centers)
##x_error = []
##y_error = []

tree = KDTree(centers)
distances, index = tree.query(centers, k=2)
distances = distances.T[1]
print(np.mean(distances))

frame=o3d.geometry.TriangleMesh.create_coordinate_frame(size=100) 
o3d.visualization.draw_geometries([NumpyToPCD(centers).paint_uniform_color([1,0,0]),NumpyToPCD(xyz).paint_uniform_color([0,0,1]),frame])


x_error = []
y_error = []
#xyz = xyz[index]
for i in range(len(xyz)):
    p = xyz[i]
    c = centers[i]
    x_error.append(p[0]-c[0])
    y_error.append(p[1]-c[1])

x = xyz[:,0]
y =  xyz[:,1]
z = x_error
fig = plt.figure(figsize=plt.figaspect(0.5))
ax = fig.add_subplot(2, 2, 1, projection='3d')
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
ax.set_zlim(-10, 10)
fig.colorbar(surf, shrink=0.5, aspect=10)
z = y_error
ax = fig.add_subplot(2, 2, 2, projection='3d')
surf = ax.plot_trisurf(x, y, z, cmap=cm.jet, linewidth=0.1)
ax.set_zlim(-10, 10)
fig.colorbar(surf, shrink=0.5, aspect=10)

fx = Rbf(x,y,x_error, function = "cubic" )
print(fx(100,50))
fy = Rbf(x,y,y_error,  function = "cubic" )
print(fy(100,50))


## test

x_test = np.linspace(-250,250,100)
y_test = np.linspace(-40,200,100)
xx, yy  =np.meshgrid(x_test,y_test)
x_test = xx.flatten()
y_test = yy.flatten()
z_test = fy(x_test, y_test).flatten()
ax = fig.add_subplot(2, 2, 4, projection='3d')
surf = ax.plot_trisurf(x_test, y_test, z_test, cmap=cm.jet, linewidth=0.1)

ax.set_zlim(-10.01, 10.01)
fig.colorbar(surf, shrink=0.5, aspect=10)


ax = fig.add_subplot(2, 2, 3, projection='3d')

x_test = np.linspace(-250,250,100)
y_test = np.linspace(-40,200,100)
xx, yy  =np.meshgrid(x_test,y_test)
x_test = xx.flatten()
y_test = yy.flatten()

z_test = fx(x_test, y_test).flatten()

surf = ax.plot_trisurf(x_test, y_test, z_test, cmap=cm.jet, linewidth=0.1)
ax.set_zlim(-10.01, 10.01)
fig.colorbar(surf, shrink=0.5, aspect=10)
plt.show()





np.savetxt("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\CONFIG\\calibrazione_piano.txt", np.asarray([x, y, x_error, y_error]).T,fmt='%10.5f', header="x \t y \t x_err \t y_err")

#res = np.loadtxt("C:\\Users\\alberto.scolari\\output\\calibrazione_piano.txt")

#image = cv2.resize(image, (int(image.shape[1]/7), int(image.shape[0]/7))) 
#cv2.imwrite("C:\\Users\\alberto.scolari\\output\\piano_marcato.png",image)
#cv2.imshow("foto",image)
#cv2.waitKey(0)
