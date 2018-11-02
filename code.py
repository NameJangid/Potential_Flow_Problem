import numpy as np
import scipy.sparse as sparse
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

L = 15.0
H = 10.0
h1 = 2.0
h2 = 4.0
N1 = 301
N2 = 201
Vin = float(input('Input Velocity : '))
Vout = (Vin*H)/(h2-h1)

delta_x = float(L)/float(N1-1)
delta_y = float(H)/float(N2-1)

alpha = ((L/(N1-1))**2)/((H/(N2-1))**2)
asq = alpha**2
asq2 = (-2)*(1+asq)

B = np.zeros(((N1*N2),1), dtype=float)

data = np.zeros(((5*N1*N2),), dtype=float)
rows = np.zeros(((5*N1*N2),), dtype=int)
cols = np.zeros(((5*N1*N2),), dtype=int)

# applying laplacian equation for non-boundary points
for i in range(1, N1-1):
  for j in range(1, N2-1):
    bi = (N2*i)+j
    bi5 = 5*(bi)
    data[bi5]=1; data[bi5+1]=asq; data[bi5+2]=asq2; data[bi5+3]=asq; data[bi5+4]=1;
    rows[bi5]=bi; rows[bi5+1]=bi; rows[bi5+2]=bi; rows[bi5+3]=bi; rows[bi5+4]=bi; 
    cols[bi5]=bi-N2; cols[bi5+1]=bi-1; cols[bi5+2]=bi; cols[bi5+3]=bi+1; cols[bi5+4]=bi+N2;

# making phi=0 for left boundary (Dirichlet Condition)
i = 0
for j in range(0, N2):
  bi = j
  bi5 = 5*bi
  data[bi5]=1
  rows[bi5]=bi
  cols[bi5]=bi

# neumann condition for top boundary
j = N2-1
for i in range(1, N1-1):
  bi = (N2*i)+j
  bi5 = 5*bi
  data[bi5]=1; data[bi5+1]=-1;
  rows[bi5]=bi; rows[bi5+1]=bi;
  cols[bi5]=bi; cols[bi5+1]=bi-1;

# neumann condition for bottom boundary
j = 0
for i in range(1, N1-1):
  bi = (N2*i)+j
  bi5 = 5*bi
  data[bi5]=1; data[bi5+1]=-1;
  rows[bi5]=bi; rows[bi5+1]=bi;
  cols[bi5]=bi; cols[bi5+1]=bi+1;

j_top = N2-int((float(h1)*float(N2))/float(H))
j_bottom = N2-int((float(h2)*float(N2))/float(H))

# neumann condition for right-bottom boundary
i = N1-1
for j in range(0, j_bottom):
  bi = (N2*i)+j
  bi5 = 5*bi
  data[bi5]=1; data[bi5+1]=-1;
  rows[bi5]=bi; rows[bi5+1]=bi;
  cols[bi5]=bi; cols[bi5+1]=bi-N2;

# neumann condition for right-top boundary
i = N1-1
for j in range(j_top, N2):
  bi = (N2*i)+j
  bi5 = 5*bi
  data[bi5]=1; data[bi5+1]=-1;
  rows[bi5]=bi; rows[bi5+1]=bi;
  cols[bi5]=bi; cols[bi5+1]=bi-N2;

# condition for hole at the right boundary
i = N1-1
for j in range(j_bottom, j_top):
  bi = (N2*i)+j
  bi5 = 5*bi
  data[bi5]=1/delta_x; data[bi5+1]=(-1)/delta_x;
  rows[bi5]=bi; rows[bi5+1]=bi;
  cols[bi5]=bi; cols[bi5+1]=bi-N2;
  B[bi][0] = Vout

sm = sparse.csr_matrix((data,(rows,cols)),shape=((N1*N2),(N1*N2)))

del data; del rows; del cols;

sols = spsolve(A=sm,b=B)
velocity = np.zeros(((N1*N2),2),dtype=float)

# velocity part-1
for i in range(1,N1):
  for j in range(1,N2):
    bi = (N2*i)+j
    velocity[bi][0] = (sols[bi]-sols[bi-N2])/delta_x
    velocity[bi][1] = (sols[bi]-sols[bi-1])/delta_y

# velocity bottom
j=0
for i in range(1,N1):
  bi = (N2*i)+j
  velocity[bi][0] = (sols[bi]-sols[bi-N2])/delta_x
  velocity[bi][1] = 0

# velocity left wall
i=0
for j in range(0,N2):
  bi = j
  velocity[bi][0] = Vin
  velocity[bi][1] = 0

sols.tofile('phi_scalar'+str(int(Vin))+'.txt',sep='\n')
np.savetxt(fname='v_vector'+str(int(Vin))+'.txt',X=velocity,fmt='%f',delimiter=',',newline='\n')

xa, ya = np.linspace(0.0,L,N1), np.linspace(0.0,H,N2)
Xa, Ya = np.meshgrid(xa, ya)
phi_plt = np.reshape(sols,(N1,N2))
phi_plt = np.transpose(phi_plt)
fig = plt.figure()
cp = plt.contourf(Xa, Ya, phi_plt, 50)
plt.colorbar(cp)
plt.title('Contour Plot')
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('contour'+str(int(Vin))+'.png', dpi = 1000)

vx = np.reshape(velocity[:,0],(N1,N2));  vx = np.transpose(vx)[0::5,0::5];
vy = np.reshape(velocity[:,1],(N1,N2));  vy = np.transpose(vy)[0::5,0::5];
plt.figure()
Xa = Xa[0::5,0::5];  Ya = Ya[0::5,0::5];
cp2 = plt.quiver(Xa,Ya,vx,vy)
plt.xlabel('x')
plt.ylabel('y')
plt.savefig('quiver'+str(int(Vin))+'.png', dpi = 1000)