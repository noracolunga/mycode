import os
#import tempfile
#os.environ["MPLCONFIGDIR"] = tempfile.gettempdir()
import glob
import cupy as cp
import numpy as np
#import matplotlib.pyplot as plt
import math
import sys
############# parameters 
# 0 = Outside of the black hole in Minkowski Spacetime
# 1 = Outside of the black hole in de Sitter Spacetime
# 2 = Inside of the black hole in Minkowski Spacetime
# 3 = Inside of the black hole in de Sitter Spacetime
spacetime = 3
Nr=101  # number of grid points in r
Ntheta=101 # number of grid points in theta
Nphi=101 # number of grid points in phi
############################################
# outside:
    #outside
if spacetime == 0:
    fileList = glob.glob("out-min*.vts")
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    dt=0.001 # time step size
    Tf=0.1  # final time 
    Ra=2.0; Rb=3.0  # interval ends for r
    rs=1.0 # radius of the blackhole
    t=cp.arange(0, Tf, dt)
    file_name = "out-min{:03d}"
if spacetime == 1:
    fileList = glob.glob("out-deSit*.vts")
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    dt=0.001 # time step size
    Tf=0.1  # final time 
    Ra=2.0; Rb=3.0  # interval ends for r
    rs=1.0 # radius of the blackhole
    t=cp.arange(0, Tf, dt)
    file_name = "out-deSit{:03d}"
    #inside
if spacetime == 2:
    fileList = glob.glob("in-min*.vts")
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    dt=0.001 # time step size
    T0=0.1
    Tf=0.15  # final time 
    Ra=2.0; Rb=3.0  # interval ends for r
    rs=4.0 # radius of the blackhole
    t=cp.arange(T0, Tf, dt)
    file_name = "in-min{:03d}"
    #if Tf>=rs:
    #    print("Tf should be less than rs.")
    #   sys.exit()
if spacetime == 3:
    fileList = glob.glob("in-deSit*.vts")
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)
    dt=0.001 # time step size
    T0=0.1
    Tf=0.2  # final time 
    Ra=2.0; Rb=3.0  # interval ends for r
    rs=4.0 # radius of the blackhole
    t=cp.arange(T0, Tf, dt)
    file_name = "in-deSit{:03d}"
    #if Tf>=rs:
    #    print("Tf should be less than rs.")
    #    sys.exit()
############################################
#computational domain
Ta=math.pi/8; Tb=math.pi/4    # interval ends for theta radians; should be between 0 and pi
Pa=0; Pb=math.pi/4  # interval ends for phi radians
#center of the bump function, inside the domain
r_c=2.5
theta_c=3*math.pi/16
phi_c=math.pi/8
x_c=r_c*np.sin(theta_c)*np.cos(phi_c)
y_c=r_c*np.sin(theta_c)*np.sin(phi_c)
z_c=r_c*np.cos(theta_c)
rbump=0.2
rbump2=rbump**2
r=cp.linspace(Ra, Rb, num=Nr, endpoint=True)
dr=(Rb-Ra)/(Nr-1.0)
dr2=dr**2
theta=cp.linspace(Ta, Tb, num=Ntheta, endpoint=True)
dtheta=(Tb-Ta)/(Ntheta-1.0)
dtheta2=dtheta**2
phi=cp.linspace(Pa, Pb, num=Nphi, endpoint=True)
dphi=(Pb-Pa)/(Nphi-1.0)
dphi2=dphi**2
mu2=9
lamb=1
###############
############### Initialization
Ntime=t.size
Nout=10
# current time step
w1=cp.zeros((Nr, Ntheta, Nphi)) # 3d array 
a=2  
wt1=cp.zeros((Nr, Ntheta, Nphi))
# next time step
w2=cp.zeros((Nr, Ntheta, Nphi))   # 3d array
wt2=cp.zeros((Nr, Ntheta, Nphi))
k1 = cp.zeros_like(w1)
k1t = cp.zeros_like(w1)
k2 = cp.zeros_like(w1)
k2t = cp.zeros_like(w1)
k3 = cp.zeros_like(w1)
k3t = cp.zeros_like(w1)
k4 = cp.zeros_like(w1)
k4t = cp.zeros_like(w1)
block_size_x, block_size_y, block_size_z = 8,8,8
########## GPU stuff ###########
gridx = math.ceil(Nr/block_size_x)
gridy = math.ceil(Ntheta/block_size_y)
gridz = math.ceil(Nphi/block_size_z)
grids = (gridx, gridy, gridz)
blocks = (block_size_x, block_size_y, block_size_x)
x=cp.zeros_like(w1); y=cp.zeros_like(w1); z=cp.zeros_like(w1)