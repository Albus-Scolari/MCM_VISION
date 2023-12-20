
import numpy as np
import copy
import cv2
from itertools import product
from scipy.interpolate import Rbf
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

### test 3d correction
x = np.linspace(0,480, 33)
y = np.linspace(-75,75,11)
z = np.asarray([3,20,40,60])

xyz = np.array(list(product(z,x,y)))
z = xyz[:,0]
x = xyz[:,1]
y = xyz[:,2]

res= []
x_err = []
for i in [0,20,40,60]:
    i = int(i)
    res = np.loadtxt("C:\\Users\\alberto.scolari\\source\\repos\\humans-RME\\DATA\\calibrazione_piano\\" + str(i) + "\\calibrazione_piano.txt").T
    x_err.append(res[2])
 
x_err = np.concatenate(x_err, axis= 0)
fx = Rbf(x,y,z,x_err, function = "cubic" )
print(fx(100,200,25))



#test plot 
x = np.linspace(0,480, 50)
y = np.linspace(-75,75,50)
z = np.linspace(0,60,50)
xx, yy,zz  =np.meshgrid(x,y,z)
x_test = xx.flatten()
y_test = yy.flatten()
z_test = zz.flatten()
c = fx(x_test,y_test,z_test).flatten()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
img = ax.scatter(x_test, y_test, z_test, c=c, cmap=plt.hot())
fig.colorbar(img)
plt.show()
