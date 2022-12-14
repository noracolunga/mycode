# Runge-Kutta method
# https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods
# solves ODE y'(t)=f(t,y(t)) with some initial condition y(0)
from parameters import *
import matplotlib.pyplot as plt
from pyevtk.hl import gridToVTK 

if spacetime == 0:
    from kernels_out_min import *
if spacetime == 1:
    from kernels_out_deS import *
if spacetime == 2:
    from kernels_in_min import *
if spacetime == 3:
    from kernels_in_deS import *
#  from paraview import python_view
###############

initial_w(grids, blocks, (Nr, Ntheta, Nphi, r, theta, phi, cp.float64(x_c), cp.float64(y_c), cp.float64(z_c), cp.float64(rbump2), w1))
wt1=a*w1
print(cp.max(cp.absolute(w1)))

sphere(grids, blocks, (Nr, Ntheta, Nphi, r, theta, phi, x, y, z))
#plt.plot(w1[:,50, 50].get())
#plt.show()

for n in range(Ntime):
    fu(grids, blocks, (Nr, Ntheta, Nphi, r, w1, wt1, cp.float64(rs), cp.float64(dr2), cp.float64(dr), cp.float64(dtheta),
        theta, cp.float64(dtheta2), cp.float64(dphi2), cp.float64(mu2), cp.float64(lamb), k1, k1t, cp.float64(t[n])))
    #print(cp.max(cp.absolute(k1)))
    fu(grids, blocks, (Nr, Ntheta, Nphi, r, w1+dt*k1/2, wt1+dt*k1t/2, cp.float64(rs), cp.float64(dr2), cp.float64(dr), cp.float64(dtheta),
        theta, cp.float64(dtheta2), cp.float64(dphi2), cp.float64(mu2), cp.float64(lamb), k2, k2t, cp.float64(t[n]+dt/2.0)))
    #print(cp.max(cp.absolute(k2)))
    fu(grids, blocks, (Nr, Ntheta, Nphi, r, w1+dt*k2/2, wt1+dt*k2t/2, cp.float64(rs), cp.float64(dr2), cp.float64(dr), cp.float64(dtheta),
        theta, cp.float64(dtheta2), cp.float64(dphi2), cp.float64(mu2), cp.float64(lamb), k3, k3t, cp.float64(t[n]+dt/2.0)))
    #print(cp.max(cp.absolute(k3)))
    fu(grids, blocks, (Nr, Ntheta, Nphi, r, w1+dt*k3, wt1+dt*k3t, cp.float64(rs), cp.float64(dr2), cp.float64(dr), cp.float64(dtheta),
        theta, cp.float64(dtheta2), cp.float64(dphi2), cp.float64(mu2), cp.float64(lamb), k4, k4t, cp.float64(t[n]+dt)))
    #print(cp.max(cp.absolute(k4)))
    w1=w1+(1/6)*dt*(k1+2*k2+2*k3+k4)
    wt1=wt1+(1/6)*dt*(k1t+2*k2t+2*k3t+k4t)
    #print(n, cp.max(cp.absolute(w1)))
    if n%Nout == 0:
        gridToVTK(file_name.format(n//Nout), x.get(), y.get(), z.get(), pointData = {"w" : w1.get()})

#for i in range(Nr):
#    for j in range(Ntheta):
#        for k in range(Nphi):
#            x[i,j,k] = r[i]*np.sin(theta[j])*np.cos(phi[k])
#            y[i,j,k] = r[i]*np.sin(theta[j])*np.sin(phi[k]) 
#            z[i,j,k] = r[i]*np.cos(theta[j]) 

print(cp.max(cp.absolute(w1)))

#plt.plot(w1[:,50, 50].get())
#plt.show()

print("Finished")