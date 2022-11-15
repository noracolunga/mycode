
import cupy as cp


initial_w = cp.RawKernel(r'''
extern "C" __global__
void initial_w(const int Nr, const int Ntheta, const int Nphi, double* r, double* theta, double* phi, double x_c, double y_c, double z_c, double rbump2, double* w1){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  int k = blockDim.z*blockIdx.z + threadIdx.z;
  double d2;
  if (i>=0 and i<Nr and j>=0 and j<Ntheta and k>=0 and k<Nphi){
    d2=pow((r[i]*sin(theta[j])*cos(phi[k])-x_c), 2.0)+pow((r[i]*sin(theta[j])*sin(phi[k])-y_c), 2.0)+pow((r[i]*cos(theta[j])-z_c), 2.0);
    if (d2<rbump2){
      w1[i*Ntheta*Nphi+j*Nphi+k]=exp((1.0/rbump2)-1.0/(rbump2-d2));
    }
  }
}
''', 'initial_w')


fu = cp.RawKernel(r'''
extern "C" __global__
void fu(const int Nr, const int Ntheta, const int Nphi, double* r, double* w, double* wt, double rs, double dr2, double dr, double dtheta,
  double* theta, double dtheta2, double dphi2, double mu2, double lamb, double* f, double* ft, double t){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  int k = blockDim.z*blockIdx.z + threadIdx.z;
  if (i>0 and i<(Nr-1) and j>0 and j<(Ntheta-1) and k>0 and k<(Nphi-1)){
    int ijk = i*Ntheta*Nphi+j*Nphi+k;
    f[ijk]=wt[ijk];
    ft[ijk]=pow((1-rs/r[i]),2.0)/dr2*(w[ijk-Ntheta*Nphi]-2*w[ijk]+w[ijk+Ntheta*Nphi])
    +(1-rs/r[i])*2/r[i]*(1-rs/(2*r[i]))/(2*dr)*(w[ijk+Ntheta*Nphi]-w[ijk-Ntheta*Nphi])
    +(1-rs/r[i])/pow(r[i],2.0)/tan(theta[j])/(2*dtheta)*(w[ijk+Nphi]-w[ijk-Nphi])
    +(1-rs/r[i])/pow(r[i],2.0)/dtheta2*(w[ijk-Nphi]-2*w[ijk]+w[ijk+Nphi])
    +(1-rs/r[i])/pow(r[i],2.0)/pow(sin(theta[j]),2.0)/dphi2*(w[ijk-1]-2*w[ijk]+w[ijk+1])+mu2*w[ijk]-lamb*pow(w[ijk], 3.0);
  }
}
''', 'fu')

sphere = cp.RawKernel(r'''
extern "C" __global__
void sphere(const int Nr, const int Ntheta, const int Nphi, double* r, double* theta, double* phi, double* x, double* y, double* z){
  int i = blockDim.x*blockIdx.x + threadIdx.x;
  int j = blockDim.y*blockIdx.y + threadIdx.y;
  int k = blockDim.z*blockIdx.z + threadIdx.z;
  if (i>=0 and i<Nr and j>=0 and j<Ntheta and k>=0 and k<Nphi){
    int ijk = i*Ntheta*Nphi+j*Nphi+k;
    x[ijk] = r[i]*sin(theta[j])*cos(phi[k]);
    y[ijk] = r[i]*sin(theta[j])*sin(phi[k]);
    z[ijk] = r[i]*cos(theta[j]) ;
  }
}
''', 'sphere')