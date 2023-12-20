import numpy as np
import cv2


image = cv2.imread("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\stitched_filt.png")
thrs = 90
t, image = cv2.threshold(image,thrs,255,cv2.THRESH_BINARY)
 
scale_percent = 10 # percent of original size
width = int(image.shape[1] * scale_percent / 100)
height = int(image.shape[0] * scale_percent / 100)
dim = (width, height)
resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
 
cv2.imshow("image",resized)
cv2.waitKey(0) 
arucoDict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_6X6_250)
params = cv2.aruco.DetectorParameters_create()
corners, ids, rejected = cv2.aruco.detectMarkers(image, arucoDict, parameters = params)
print(corners)
#cv2.imwrite("C:\\Users\\alberto.scolari\\source\\repos\\MARCATURA_ROBOT\\DATA\\stitched_filt.png",image)
#arduino = serial.Serial(port="COM6", baudrate=57600,timeout=.1)

#rob.GoToJointsCoord(90,20,30,40,10,40)
#rob.GoToJointsCoord(0,0,0,0,0,0)
#rob.robot.set_cartesian([[-8.86, 518.17, 432.12], [0.013, -0.091, 0.957, 0.273]])
#res = rob.robot.get_cartesian()
#print(res)
#rob.GoToJointsCoord(0,0,0,0,0,0)
#rob.robot.set_cartesian(res)
#res = rob.robot.get_cartesian()

#print(res)