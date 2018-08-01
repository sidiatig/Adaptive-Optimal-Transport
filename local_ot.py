import numpy as np
import numpy.linalg as la
from scipy import spatial
from GH import *

def local_ot(x,y,Niter=100):#Updates eta using Dn instead of Gn
    d = 2
    Nx = len(x)
    Ny = len(y)
    da = 10
    db = 14
     
    #local minimax algorithm, rejction using G, update eta with D
#     Niter = 50
    eps = 1e-6
    eta0 = 0.05
    eta = eta0
    etaC = 2.0

    #setting up initial b, and minvar,D
    vx = np.cov(x.T)
    vy = np.cov(y.T)
    mx = np.mean(x,axis=0)
    my = np.mean(y,axis=0)
    # vxy = np.var(np.concatenate([x.T,y.T]))
    D = np.sqrt(np.max([vx[0,0],vx[1,1],vy[0,0],vy[1,1]]))
    # mxy = np.mean(np.concatenate([x,y]))
    #initial a
    a0 = np.zeros(da)
    a0[6] = 1/D
    a0[8] = 1/D
    #initial b
    b0 = np.zeros(db)
    B0 = 0.5*(la.inv(vy)-la.inv(vx))
    b0[3] = B0[0,0]
    b0[4] = B0[0,1]+B0[1,0]
    b0[5] = B0[1,1]
    B1 = la.solve(vx,mx) - la.solve(vy,my)
    b0[1] = B1[0]
    b0[2] = B1[1]
    b0[0] = 0.5*(my.dot(la.solve(vy,my)) - mx.dot(la.solve(vx,mx)))
    b0[7] = 1/D
    b0[11]= 1/D
    b0[8:10] = 0.1*np.random.uniform(low=-1.0,high=1.0,size=2)
    b0[12:] = 0.1*np.random.uniform(low=-1.0,high=1.0,size=2)
    z = np.concatenate([a0,b0],axis=0)
    I = np.eye(da+db)
    minvar0 = 1e-12
    dsq = spatial.distance.cdist(x,y,'sqeuclidean')
    minvar = np.amax(np.amin(dsq,axis=1),axis=0) + minvar0
    delta0 = np.sqrt(minvar)
    dsq2 = spatial.distance.cdist(x,x,'sqeuclidean')
    dsq2 += 1000*np.eye(Nx)
    minvar2 = np.mean(np.amin(dsq2,axis=1),axis=0) + minvar0
    deltainf =  np.sqrt(minvar2)
    deltainf = delta0/20 #debug

    delta = delta0
    #Initial gradient
    Gn,Hn = GH(x,y,a0,b0,delta,D)
    Hxx = Hn[:da,:da]; Hyy = Hn[da:,da:]; Gx = Gn[:da]; Gy = Gn[da:]
    Mn = Gx.T.dot(la.lstsq(Hxx,Gx)[0])
    mn = Gy.T.dot(la.lstsq(Hyy,Gy)[0])

    x0 = x[:,0]
    x1 = x[:,1]
    x0a7 = x0+z[7]
    x0a72 = x0a7**2
    x1a9 = x1+z[9]
    x1a92 = x1a9**2
    argexpT = z[6]**2*x0a72+z[8]**2*x1a92
    expT = np.exp(-0.5*argexpT)
    T = np.zeros((Nx,2))
    T[:,0] = z[0] + (1+z[2])*x0+z[3]*x1+z[5]*z[6]**2*x0a7*expT
    T[:,1] = z[1] + z[3]*x0+(1+z[4])*x1+z[5]*z[8]**2*x1a9*expT

    Dn = abs(Mn + mn)
    ###debug####
#     listG = []
#     listD = []
#     listS = []
#     lists = []
#     listeta = []
    ############

    for n in range(Niter):
        Gnold = Gn
        Dnold = Dn

        delta = (delta0-deltainf)*np.exp(-5*(n+1)/Niter) + deltainf
        ####Debug########
#         listG.append(la.norm(Gn,2))
#         listD.append(Dn)
#         listS.append(Mn)
#         lists.append(mn)
#         listeta.append(eta)
#         #PLOTS FOR DEBUG
#         x0a7 = x0+z[7]
#         x1a9 = x1+z[9]
#         argexpT = z[6]**2*x0a7**2+z[8]**2*x1a9**2
#         expT = np.exp(-0.5*argexpT)
#         T = np.zeros((Nx,2))
#         T[:,0] = z[0] + (1+z[2])*x0+z[3]*x1+z[5]*z[6]**2 * x0a7*expT
#         T[:,1] = z[1] + z[3]*x0+(1+z[4])*x1+z[5]*z[8]**2 * x1a9*expT
#         plt.scatter(T[:,0],T[:,1], color='red', alpha = 0.5);
#         plt.scatter(y[:,0],y[:,1], color='blue', alpha = 0.2);
#         plt.draw()
#         plt.pause(1e-5)
#         print('a',z[:da])
#         print('b',z[da:])
#         print('Step n=',n)
        ##########################################     

        if la.norm(Gn,2)<eps:
            break
        #reject moves if bad direction
        baddir = True
        eta = 2.0*eta
        while(baddir and eta>eta0):
            eta = eta/2.0
            dz = la.solve((1.0/eta)*I+Hn,Gn)
            baddir = dz.dot(Gn)<0.0
        z = z - dz

        Gn,Hn = GH(x,y,z[:da],z[da:],delta,D)
        Hxx = Hn[:da,:da]; Hyy = Hn[da:,da:]; Gx = Gn[:da]; Gy = Gn[da:]
        Mn = Gx.T.dot(la.lstsq(Hxx,Gx)[0])
        mn = Gy.T.dot(la.lstsq(Hyy,Gy)[0])
        Dn = abs(Mn + mn)
        #etamin = eta0/(la.norm(Gn,2)+1)#debug
        if Dn<=Dnold:
            eta = min((etaC+0.1)*eta,1.0/(1+la.norm(Gn,2)))#debug
        else:
            eta = max(eta/etaC, 0.5/(1+la.norm(Gn,2)))
    a,b = z[:da],z[da:]
    x0a7 = x0+z[7]
    x0a72 = x0a7**2
    x1a9 = x1+z[9]
    x1a92 = x1a9**2
    argexpT = z[6]**2*x0a72+z[8]**2*x1a92
    expT = np.exp(-0.5*argexpT)
    T = np.zeros((Nx,2))
    T[:,0] = z[0] + (1+z[2])*x0+z[3]*x1+z[5]*z[6]**2*x0a7*expT
    T[:,1] = z[1] + z[3]*x0+(1+z[4])*x1+z[5]*z[8]**2*x1a9*expT
    return(T,a,b)