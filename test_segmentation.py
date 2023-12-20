
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import random
import cv2
from Source import VimbaCameraDriver as vcd
from Source import helper
from Source.UTILS.pcd_numpy_utils import *
import Source.REGISTRATION.PC_registration as PC_registration
import Source.REGISTRATION.ICP_utils as ICP_utils 
import alphashape







filepath = "C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\immagine origine.png"

image = o3d.io.read_image(filepath)
o3d.visualization.draw_geometries([image])


images = []
thresh = []
points = []
cam = vcd.Vimba_camera()
hh =  helper.Helperhandler()
reg_hl = PC_registration.PointCloudRegistration()

p = 40
for i in range(10):
    img = cv2.imread(filepath+"\\FIANCO_S0_0"+str(i)+".tif")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    images.append(img)
    
    tr = cv2.threshold(img,p,255,cv2.THRESH_BINARY)[1]
    thresh.append(tr)
    #cv2.imshow("foto",tr)
    #cv2.waitKey(0)
    
    pts = hh.ImageToPoints(tr,cam) 
    print(len(pts))
    bol = pts[:,2]<0.5
    pts = pts[bol]

    #bol = pts[:,1]>50
    #pts = pts[bol]
    #bol = pts[:,1]<80
    #pts = pts[bol]
    pts = np.asarray(random.sample(list(pts), 10000))
    temp = np.asarray(alphashape.alphashape(pts[:,0:2], 0.1).exterior.coords.xy).T
    pts1 = np.zeros((len(temp),3))
    pts1[:,0:2] = temp

    tr = cv2.threshold(img,120,255,cv2.THRESH_BINARY)[1]
    thresh.append(tr)
    #cv2.imshow("foto",tr)
    #cv2.waitKey(0)
    
    pts = hh.ImageToPoints(tr,cam) 
    print(len(pts))
    bol = pts[:,2]>0.5
    pts = pts[bol]

    bol = pts[:,1]>50
    pts = pts[bol]
    bol = pts[:,1]<80
    pts = pts[bol]
    pts = np.asarray(random.sample(list(pts), 5000))
    temp = np.asarray(alphashape.alphashape(pts[:,0:2], 0.1).exterior.coords.xy).T
    pts2 = np.zeros((len(temp),3))
    pts2[:,0:2] = temp
    pts = np.concatenate([pts1,pts2], axis= 0)
    #bol = pts[:,0]>10
    #pts = pts[bol]
    points.append(pts)
    if i!=0:
        o3d.visualization.draw_geometries([NumpyToPCD(points[0]).paint_uniform_color([1,0,0]),NumpyToPCD(points[i]).paint_uniform_color([0,0,1])])
    
        Source_transf, RT_ls, Tls= reg_hl.DenseAlignment(points[i],points[0])
        o3d.visualization.draw_geometries([NumpyToPCD(points[0]).paint_uniform_color([1,0,0]),NumpyToPCD(Source_transf).paint_uniform_color([0,0,1])])
