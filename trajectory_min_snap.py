import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
from mpl_toolkits.mplot3d import Axes3D
#import jdc
from numpy.polynomial.polynomial import polyval

np.set_printoptions(precision=3)
pylab.rcParams['figure.figsize'] = 10, 10

def matrix_generation(ts):
    b =np.array([[1, ts,  ts**2, ts**3,    ts**4,    ts**5,    ts**6,    ts**7],
                 [0, 1 ,2*ts,  3*ts**2,  4*ts**3,  5*ts**4,  6*ts**5,  7*ts**6],
                 [0, 0 ,2,     6*ts,    12*ts**2, 20*ts**3, 30*ts**4, 42*ts**5],
                 [0, 0 ,0,     6,       24*ts,    60*ts**2,120*ts**3,210*ts**4],
                 [0, 0 ,0,     0,       24   ,   120*ts   ,360*ts**2,840*ts**3],
                 [0, 0 ,0,     0,       0    ,   120      ,720*ts  ,2520*ts**2],
                 [0, 0 ,0,     0,       0    ,   0        ,720     ,5040*ts   ],
                 [0, 0 ,0,     0,       0    ,   0        ,0       ,5040      ]])
    
    return b

############################################################
#2D Trajectory
############################################################
def multiple_waypoints(t):
    n= t.shape[0]-1
    
    m= np.zeros((8*n,8*n))
    
    for i in range(n):
    
        if i == 0:
            
            # initial condition of the first curve
            b = matrix_generation(-1.0)
            m[8*i:8*i+4,8*i:8*i+8] = b[:4,:]
            
            # intermidiary condition of the first curve
            b = matrix_generation(1.0)
            m[8*i+4:8*i+7+4,8*i:8*i+8] = b[:-1,:]
            
            # starting condition of the second curve position and derivatives 
            b = matrix_generation(-1.0)
            m[8*i+4+1:8*i+4+7,8*(i+1):8*(i+1)+8] = -b[1:-1,:]
            m[8*i+4+7:8*i+4+8,8*(i+1):8*(i+1)+8] = b[0,:]
            
        elif i!=n-1:
            
            # starting condition of the ith curve position and derivatives 
            b = matrix_generation(1.0)
            m[8*i+4:8*i+7+4,8*i:8*i+8] = b[:-1,:]
            
            # end condition of the ith curve position and derivatives 
            b = matrix_generation(-1.0)
            m[8*i+4+1:8*i+4+7,8*(i+1):8*(i+1)+8] = -b[1:-1,:]
            m[8*i+4+7:8*i+4+8,8*(i+1):8*(i+1)+8] = b[0,:]
        
        if i==n-1: 
            
            # end condition of the final curve position and derivatives (4 boundary conditions) 
            b = matrix_generation(1.0)
            m[8*i+4:8*i+4+4,8*i:8*i+8] = b[:4,:]
            
    return m 

def rhs_generation(x):
    n= x.shape[0]-1
    
    big_x = np.zeros((8*n))
    big_x[:4] = np.array([x[0],0,0,0]).T
    big_x[-4:] = np.array([x[-1],0,0,0]).T
    
    for i in range(1,n):
        big_x[8*(i-1)+4:8*(i-1)+8+4] = np.array([x[i],0,0,0,0,0,0,x[i]]).T
            
    return big_x

t = np.array([0,1,2,3,4,5,6,7,8,9,10])
x = np.array([0,1,2,4,6,7,7.5,6.5,5,4.5,4.0])

m = multiple_waypoints(t)
b = rhs_generation(x)

coeff = coeff = np.linalg.solve(m,b)

c = np.zeros((int(coeff.shape[0]/8),8))

for l in range(t.shape[0]-1):
    t1 = np.linspace(-1.0,1.0,101)
    x1 = np.zeros(t1.shape)
    
    c[l,:] = coeff[8*l:8*l+8]
        
    for i in range(t1.shape[0]):
        x1[i]=polyval(t1[i], c[l,:])
        
    if l ==0:
        xx= x1[:-1]
        tx= (t[l]+t[l+1])/2 + t1*(t[l+1]-t[l])/2
        tt= tx[:-1]
    else:
        xx=np.hstack((xx,x1[:-1]))
        tx= (t[l]+t[l+1] + t1*(t[l+1]-t[l]))/2
        tt=np.hstack((tt,tx[:-1]))

plt.plot(tt,xx)
plt.scatter(t,x,marker='o',color='red')
plt.title('Trajectory').set_fontsize(20)
plt.xlabel('$t$ [$s$]').set_fontsize(20)
plt.ylabel('$x$ [$m$]').set_fontsize(20)
plt.xticks(fontsize = 14)
plt.yticks(fontsize = 14)
plt.show()

###########################################
#3D Trajectory
###########################################
t = np.array([0,1,2,3,4,5])

p = np.array([[0,1,2,2,1.5,1.5,],
              [0,0,0,1,1.5,1.5,],
              [0,1,1,1,1.5,2.0,]])

m = multiple_waypoints(t)

b_x = rhs_generation(p[0,:])
b_y = rhs_generation(p[1,:])
b_z = rhs_generation(p[2,:])

coeff_x  = np.linalg.solve(m,b_x)
coeff_y  = np.linalg.solve(m,b_y)
coeff_z  = np.linalg.solve(m,b_z)

c_x = np.zeros((int(coeff_x.shape[0]/8),8))
c_y = np.zeros((int(coeff_x.shape[0]/8),8))
c_z = np.zeros((int(coeff_x.shape[0]/8),8))

t1 = np.linspace(-1.0,1.0,101)

for l in range(t.shape[0]-1):

    x1 = np.zeros(t1.shape)
    y1 = np.zeros(t1.shape)
    z1 = np.zeros(t1.shape)
    
    c_x[l,:] = coeff_x[8*l:8*l+8]
    c_y[l,:] = coeff_y[8*l:8*l+8]
    c_z[l,:] = coeff_z[8*l:8*l+8]
        
    for i in range(t1.shape[0]):
        x1[i]=polyval(t1[i], c_x[l,:])
        y1[i]=polyval(t1[i], c_y[l,:])
        z1[i]=polyval(t1[i], c_z[l,:])
        
    if l ==0:
        xx= x1[:-1]
        yy= y1[:-1]
        zz= z1[:-1]
        tx= (t[l]+t[l+1])/2 + t1*(t[l+1]-t[l])/2
        tt= tx[:-1]
    else:
        xx=np.hstack((xx,x1[:-1]))
        yy=np.hstack((yy,y1[:-1]))
        zz=np.hstack((zz,z1[:-1]))
        tx= (t[l]+t[l+1] + t1*(t[l+1]-t[l]))/2
        tt=np.hstack((tt,tx[:-1]))

        
fig = plt.figure() 
ax = fig.gca(projection='3d')
ax.plot(xx, yy, zz)
ax.scatter(p[0,:], p[1,:], p[2,:], marker ='o', color='red')

plt.title('Flight path').set_fontsize(20)
ax.set_xlabel('$x$ [$m$]').set_fontsize(20)
ax.set_ylabel('$y$ [$m$]').set_fontsize(20)
ax.set_zlabel('$z$ [$m$]').set_fontsize(20)
plt.legend(['Planned path','waypoints'],fontsize = 14)

plt.show()