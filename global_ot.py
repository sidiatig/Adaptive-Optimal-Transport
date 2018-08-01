import numpy as np
from local_ot import *

def idx_choice(x,y):
    Nx = len(x)
    Ny = len(y)
    #assignement for x:
    idxx = np.zeros(Nx)
    q = int(Nx/Ny)
    r = Nx % Ny
    for i in range(q):
        idxx[i*Ny:(i+1)*Ny] = np.random.choice(np.arange(Ny),Ny,replace=False)
    if r!=0: 
        idxx[-r:] = np.random.choice(np.arange(Ny),r,replace=False)
    #assignement for y:
    idxy = np.zeros(Ny)
    q = int(Ny/Nx)
    r = Ny % Nx
    for i in range(q):
        idxy[i*Nx:(i+1)*Nx] = np.random.choice(np.arange(Nx),Nx,replace=False)
    if r!=0: 
        idxy[-r:] = np.random.choice(np.arange(Nx),r,replace=False)
    return(y[idxx.astype(int)],x[idxy.astype(int)])  


def global_ot(x,y,K=40,tol=1e-2,Maxiter=100):

    Nx = len(x)
    Ny = len(y) 
    a_maps = K*[0]
    b_maps = K*[0]

    # #Initialize intermediary distributions with random selections for x
    # x_K = np.array([y[np.random.randint(0,Ny)] for i in range(Nx)])
    # mu = []
    # mu.append(x)
    # for k in range(1,K):
    #     z = (1-k/K)*x+(k/K)*x_K
    #     mu.append(z)
    # mu.append(y)

    #Initialize intermediary distributions both from x and y
    x_K, y_0 = idx_choice(x,y)
    mu = []
    mu.append(x)
    for k in range(1,K):
        z0k = (1-k/K)*x+(k/K)*x_K
        rdcx = np.random.choice(np.arange(Nx),int(Nx*(1-k/K)),replace=False)
        zx = z0k[rdcx]
        zKk = (1-k/K)*y_0+(k/K)*y
        rdcy = np.random.choice(np.arange(Ny),int(Ny*k/K),replace=False)
        zy = zKk[rdcy]
        z = np.concatenate([zx,zy])
        mu.append(z)
    mu.append(y)


    for niter in range(Maxiter):
        print(niter/Maxiter * 100)#debug
        x_Kold=x_K
        #solve local OT's
        z = x
        for k in range(1,K+1):
    #         #plot before transport
    #         bins = 35
    #         plt.hist(z,density=True, bins=bins, color='red', alpha = 0.5);
    #         plt.hist(mu[k],density=True, bins=bins, color='blue', alpha = 0.5);
#             plt.scatter(z[:,0],z[:,1], color='red', alpha = 0.5);
#             plt.scatter(mu[k][:,0],mu[k][:,1], color='blue', alpha = 0.2);
#             plt.title('Before local OT')
#             plt.draw()
#             plt.show()

            #solve local transport problem
            Tloc,a,b = local_ot(z,mu[k],Niter=200)

            #plot after transport
    #         bins = 35
    #         plt.hist(Tloc,density=True, bins=bins, color='red', alpha = 0.5);
    #         plt.hist(mu[k],density=True, bins=bins, color='blue', alpha = 0.5);
#             plt.scatter(Tloc[:,0],Tloc[:,1], color='red', alpha = 0.5);
#             plt.scatter(mu[k][:,0],mu[k][:,1], color='blue', alpha = 0.2);
#             plt.title('After local OT')
#             plt.draw()
#             plt.show()

            a_maps[k-1] = a
            b_maps[k-1] = b
            z = Tloc
        x_K = z
        if (1/Nx)*la.norm(x_K-x_Kold,2)<tol:
            break
        #update intermediate distributions
        for k in range(1,K):
            mu[k] = (1-k/K)*x+(k/K)*x_K
            
    return(x_K, c_maps, a_maps, b_maps)