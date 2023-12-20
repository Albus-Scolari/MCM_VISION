#from STEPAnalyzer import StepAnalyzer
#from utils_STEP import display_shape
#import open3d as o3d
#from Source.UTILS.pcd_numpy_utils import *
#from scipy.spatial import ConvexHull
#from Source import helper
#from Source import VimbaCameraDriver as vcd
#import cv2
#import copy
#from skimage.feature import match_template
#import matplotlib.pyplot as plt
import adsk.core, adsk.fusion, adsk.cam, traceback

app = adsk.core.Application.get()

img1 = cv2.imread("Input1.png")
ref = cv2.imread("Input3Ref.png")
ref =cv2.flip(ref, 0)
ref = cv2.resize(ref, (int(ref.shape[1]/10), int(ref.shape[0]/10)))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
img1 = cv2.resize(img1, (int(img1.shape[1]/10), int(img1.shape[0]/10)))

bol = ref>0
ref[bol] = 255

#img1 = cv2.resize(img1, (int(img1.shape[1]/10), int(img1.shape[0]/10)))

result = match_template(img1, ref)
ij = np.unravel_index(np.argmax(result), result.shape)
x, y = ij[::-1]



fig = plt.figure(figsize=(8, 3))
ax1 = plt.subplot(1, 3, 1)
ax2 = plt.subplot(1, 3, 2)
ax3 = plt.subplot(1, 3, 3, sharex=ax2, sharey=ax2)

ax1.imshow(ref, cmap=plt.cm.gray)
ax1.set_axis_off()
ax1.set_title('template')

ax2.imshow(img1, cmap=plt.cm.gray)
ax2.set_axis_off()
ax2.set_title('image')
# highlight matched region
hcoin, wcoin = ref.shape
rect = plt.Rectangle((x, y), wcoin, hcoin, edgecolor='r', facecolor='none')
ax2.add_patch(rect)

ax3.imshow(result)
ax3.set_axis_off()
ax3.set_title('`match_template`\nresult')
# highlight matched region
ax3.autoscale(False)
ax3.plot(x, y, 'o', markeredgecolor='r', markerfacecolor='none', markersize=10)

plt.show()







#filename = "C:/Users/alberto.scolari/Source/Repos/humans-RME/DATA/STEP_FILE/tendicatenaap214_ok.stp"
#sa = StepAnalyzer()
##shape_ls, colors = sa.STEP_open_with_shapes_colors(filename)
#shape = sa.find_axis_color(filename,[[0,1,0],[1,0,0]])
#display_shape(shape)

##print("OK")
#hh = helper.Helperhandler()
#pcd1 = o3d.io.read_point_cloud("C:/Users/alberto.scolari/points1114.ply")
##o3d.visualization.draw_geometries([pcd1])
##compute concave hull and add
#points1 = PCDToNumpy(pcd1)
##v1 = points1[:,0:2]
###v2 = points2[:,0:2]
###alpha = 0.2
###v1 = alpha_shape(v1, alpha)
###v2 = alpha_shape(v2, alpha*3)
#vt = ConvexHull(v1).vertices
#v1 = v1[vt]
##v2 = ConvexHull(v2)
#print(len(v1))
##print(len(v2))

##z2 = np.zeros(len(v2))
#z1 = np.zeros(len(v1))

##v2  =np.asarray([v2[:,0],v2[:,1],z2]).T
#v1  =np.asarray([v1[:,0],v1[:,1],z1]).T


#img1, points1 = hh.PointCloudToImage(points1)
#img1 = img1.astype(np.uint8)
#cam = vcd.Vimba_camera()
##cv2.imshow("test", img1)
##cv2.waitKey(0)
#v1 = hh.ImageToPoints(img1, cam)* cam.pxmm
#v1 = np.array(v1).reshape((-1,1,2)).astype(np.int32) 
#            ###v2 = np.array(v2).reshape((-1,1,2)).astype(np.int32)
#hull1 = np.squeeze(cv2.convexHull(v1))
##t = copy.deepcopy(hull1[:,0])
##hull1[:,0] =copy.deepcopy(hull1[:,1])
##hull1[:,1] = copy.deepcopy(t)
#img11 = cv2.fillPoly(img1, pts=[hull1], color=(255,255,255))
####invert x y


#cv2.imwrite("C:/Users/alberto.scolari/test.png", img11)
#cv2.waitKey(0)

#o3d.visualization.draw_geometries([NumpyToPCD(v1-np.mean(v1,axis=0)).paint_uniform_color([0,0,1])])