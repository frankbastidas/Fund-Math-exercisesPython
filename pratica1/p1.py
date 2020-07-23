#%%

import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(0,10,100)
y=5- 3*x + 1.0*np.random.rand(100)

plt.scatter(x,y)

A=np.array([x,np.ones(len(x))]).T
sol=np.linalg.solve(np.dot(A.T,A),np.dot(A.T,y))

Yfinal=sol[0]*x+sol[1]
plt.scatter(x,Yfinal)
plt.plot(x,Yfinal,color=(1,0,0),linewidth=4)
# %% import data .npy
import numpy as np
import matplotlib.pyplot as plt

dad3=np.load('dados_ex3.npy')

lenDad=np.shape(dad3)



# %% gauss jacobi

import numpy as np
import matplotlib.pyplot as plt

A=np.mat([[5, 3],[4, 6]])
B=np.array([[3,4]]).T

D=np.tril(A)-np.tril(A,-1)

M=A-D

k=0
X0=np.array([[0,0]]).T
X=np.array([[0,0]]).T

Err=10
Respc=np.linalg.inv(D)*(A-D)

itera=30
Xi=np.ones((itera,len(X),1))
ki=np.linspace(0,itera-1,itera)

while k<itera:
    Bn = B - M * X0
    Xi[k] = X
    X = np.linalg.solve(D, Bn)
    Err = np.linalg.norm(X - X0) / np.linalg.norm(X)
    X0 = X
    k = k + 1

#plt.scatter(ki,Xi[:,0])
#plt.scatter(ki,Xi[:,0])
plt.plot(ki,Xi[:,1],color=(1,0,0),linewidth=4)
plt.plot(ki,Xi[:,0],color=(0,1,0),linewidth=4)


# %% Gauss Seidel

import numpy as np
import matplotlib.pyplot as plt

A=np.mat([[5, 3],[4, 6]])
B=np.array([[3,4]]).T

D=np.tril(A,0)

M=A-D

k=0
X0=np.array([[0,0]]).T
X=np.array([[0,0]]).T

Err=10
Respc=np.linalg.inv(D)*(A-D)

itera=30
Xi=np.ones((itera,len(X),1))
ki=np.linspace(0,itera-1,itera)

while k<itera:
    Bn = B - M * X0
    Xi[k] = X
    X = np.linalg.solve(D, Bn)    
    Err = np.linalg.norm(X - X0) / np.linalg.norm(X)
    X0 = X
    k = k + 1

#plt.scatter(ki,Xi[:,0])
#plt.scatter(ki,Xi[:,0])
plt.plot(ki,Xi[:,1],color=(1,0,0),linewidth=4)
plt.plot(ki,Xi[:,0],color=(0,1,0),linewidth=4)



# %% Eigenvalues Eigenvectors

#covariance
import numpy as np
import matplotlib.pyplot as plt

x=np.linspace(-2,2,100)
y=1.2*x+0.5*np.random.normal(0,1,len(x))
ax=plt.axes()
ax.scatter(x,y)
ax.axis('equal')
plt.show()

X=np.vstack((x,y))
cx= np.cov(X)
L,V=np.linalg.eig(cx)
print("Eigenvalues")
print(L)
print("Eigenvectors")
print(V)

ax.arrow(0,0,V[0,1],V[1,1])
ax.arrow(0,0,V[0,0],V[1,0])
plt.show()




# %% SVD 

#covariance
import numpy as np
import matplotlib.pyplot as plt

A=np.matrix([[1,1,0],[0,1,1]])

#eigenvalues AT * A
D,V=np.linalg.eig(A.T*A)

#singular values
sig = np.sqrt(D[0:2])

# U matrix
U=(A*V[:,0:2])/sig

# Builds SVD matrices(U and  V ready)
Sig=np.concatenate((np.diag(sig),np.array([[0],[0]])),axis=1)

#verify A=USVT
Rf=U*Sig*V.T

print(Rf)


# %% Ortogonalization and QR

import numpy as np
import matplotlib.pyplot as plt

m,n=3,5
X=np.random.rand(n,m)
#GS-Process with "reduced" QR decomposition
Q=np.zeros((n,m))
R=np.zeros((m,m))

for i in range(m):
    w=X[:,i]
    for j in range(i):
        tmp=np.dot(Q[:,j],X[:,i])
        w=w-tmp*Q[:,j]  #projection onto vj
        R[j,i]=tmp
    R[i,i] = np.linalg.norm(w)
    w=w/np.linalg.norm(w)
    Q[:,i]=w

print("verify ortonormal")  #should to result 1
print(np.linalg.norm(Q[:,0]))
print(np.dot(Q[:,0],Q[:,0]))

print("verify ortogonal")  #should to result 0
print(np.dot(Q[:,0],Q[:,1]))

print("result of Q^TQ") # product dot in each vector
print(np.matmul(Q.T,Q))

# X=QR

print("original matrix")
print(X)
print("result of QR decomposition")
print(np.matmul(Q,R))


# %%    Gram-Schmidt

import numpy as np
import matplotlib.pyplot as plt

f = lambda x: np.exp(-x) - 2*x - 4 

#buscar valores que pasen el cero, entonces la raiz prodria estar en la mitad

# %% Ejemplo tecnica alternativa (fixed point) despejando 2x de np.exp(-x) - 2*x - 4 
import numpy as np
import matplotlib.pyplot as plt

G = lambda x: (np.exp(-x)-4)/2

x0=-0.75
ap=[x0]

for i in range(50):
    x1=G(x0)
    x0=x1
    ap.append(x1)

plt.figure(figsize=(20,10))
plt.plot(ap,'b-o',markersize=15)




# %% Ejemplo tecnica alternativa2 (fixed point) despejando np.exp(-x) de np.exp(-x) - 2*x - 4 
import numpy as np
import matplotlib.pyplot as plt

G = lambda x: -np.log(2*x + 4)

x0=-0.75
ap=[x0]

for i in range(50):
    x1=G(x0)
    x0=x1
    ap.append(x1)

plt.figure(figsize=(20,10))
plt.plot(ap,'b-o',markersize=15)




# %% Newton-Rapson
import numpy as np
import matplotlib.pyplot as plt

g = lambda x: x - (np.exp(-x) - 2*x - 4) / (-np.exp(-x) - 2)

x0=-0.85
ap=[x0]
for i in range(50):
    x1=g(x0)
    x0=x1
    ap.append(x0)

plt.figure(figsize=(20,10))
plt.plot(ap,'b-o',markersize=15)
plt.savefig('im1.png')


# %% Examen 5

import numpy as np
import matplotlib.pyplot as plt

g = lambda x: x - (x*x-4*x+np.exp(x)-5) / (2*x-4+np.exp(x))

x0=2.5
ap=[x0]
for i in range(30):
    x1=g(x0)
    x0=x1
    ap.append(x0)

plt.figure(figsize=(20,10))
plt.rc('xtick',labelsize=20)
plt.rc('ytick',labelsize=20)
plt.plot(ap,'r-o',markersize=15)
plt.savefig('im1.png')
 


# %%
