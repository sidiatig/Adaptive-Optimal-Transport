import numpy as np
import numpy.linalg as la
from scipy import spatial

def GH(x,y,a,b,eps,D):
    d = 2
    Nx = len(x)
    Ny = len(y)
    da = 10
    db = 14
     
    #Precomputed data, (twisted) gradient and Hessian
    gammaA = 2*1e-1 #alpha regularization parameter 
    gammaB = 2*1e-1 #beta regularization parameter
    
    
    x0 = x[:,0]
    x1 = x[:,1]
    x0a7 = x0+a[7]
    x0a72 = x0a7**2
    x1a9 = x1+a[9]
    x1a92 = x1a9**2
    x0a7x1a9 = x0a7*x1a9
    argexpT = a[6]**2*x0a72+a[8]**2*x1a92
    expT = np.exp(-0.5*argexpT)
    T = np.zeros((Nx,2))
    T[:,0] = a[0] + (1+a[2])*x0+a[3]*x1+a[5]*a[6]**2*x0a7*expT
    T[:,1] = a[1] + a[3]*x0+(1+a[4])*x1+a[5]*a[8]**2*x1a9*expT
    T0 = T[:,0]
    T1 = T[:,1]
    y0 = y[:,0]
    y1 = y[:,1]
    g0y = b[0]+ b[1]*y0+b[2]*y1+0.5*b[3]*y0**2+b[4]*y0*y1+0.5*b[5]*y1**2
    y0b8 = y0+b[8]
    y1b9 = y1+b[9]
    y0b12 = y0+b[12]
    y1b13 = y1+b[13]
    argexpy1 =  b[7]**2*(y0b8**2+y1b9**2)
    expy1 = np.exp(-0.5*argexpy1)
    argexpy2 =  b[11]**2*(y0b12**2+y1b13**2)
    expy2 = np.exp(-0.5*argexpy2)
    g1y = b[6]*expy1
    g2y = b[10]*expy2
    gy = g0y + g1y + g2y
    g0T =b[0]+ b[1]*T0+b[2]*T1+0.5*b[3]*T0**2+b[4]*T0*T1+0.5*b[5]*T1**2
    T0b8 = T0+b[8]
    T1b9 = T1+b[9]
    T0b12 = T0+b[12]
    T1b13 = T1+b[13]
    argexpT1 =  b[7]**2*(T0b8**2+T1b9**2)
    expT1 = np.exp(-0.5*argexpT1)
    argexpT2 =  b[11]**2*(T0b12**2+T1b13**2)
    expT2 = np.exp(-0.5*argexpT2)
    g1T = b[6]*expT1
    g2T = b[10]*expT2
    gT = g0T + g1T + g2T   

    #first derivatives
    gzT = np.zeros((Nx,2))
    gzT[:,0] = b[1]+b[3]*T0+b[4]*T1
    gzT[:,0] += -b[6]*b[7]**2*T0b8*expT1
    gzT[:,0] += -b[10]*b[11]**2*T0b12*expT2
    gzT[:,1] = b[2]+b[4]*T0+b[5]*T1
    gzT[:,1] += -b[6]*b[7]**2*T1b9*expT1
    gzT[:,1] += -b[10]*b[11]**2*T1b13*expT2

    daT = np.zeros((Nx,da,d))
    daT[:,0,:] = np.array([1,0])
    daT[:,1,:] = np.array([0,1])
    daT[:,2,0] = x0
    daT[:,3,0] = x1
    daT[:,3,1] = x0
    daT[:,4,1] = x1
    daT[:,5,0] = expT*a[6]**2*x0a7
    daT[:,5,1] = expT*a[8]**2*x1a9
    daT[:,6,0] = a[5]*a[6]*expT*x0a7*(2-a[6]**2*x0a72)
    daT[:,6,1] = -a[5]*a[8]**2*a[6]*x0a72*x1a9*expT
    daT[:,7,0] = a[5]*a[6]**2*expT*(1-a[6]**2*x0a72)
    daT[:,7,1] = -a[5]*a[8]**2*a[6]**2*x0a7x1a9*expT
    daT[:,8,0] = -a[5]*a[6]**2*a[8]*x1a92*x0a7*expT
    daT[:,8,1] = a[5]*a[8]*x1a9*(2-a[8]**2*x1a92)*expT
    daT[:,9,0] = -a[5]*a[6]**2*a[8]**2*x0a7x1a9*expT
    daT[:,9,1] = a[5]*a[8]**2*(1-a[8]**2*x1a92)*expT

    #gradient w.r.t. a
    penalty_a = np.zeros(da)
    penalty_a[6] = a[6]*eps**2*np.exp(eps**2*a[6]**2) - 0*1.0/(D**2 * a[6]**3)
    penalty_a[8] = a[8]*eps**2*np.exp(eps**2*a[8]**2) - 0*1.0/(D**2 * a[8]**3)
    penalty_a[7] = a[7] * 1.0/(10*D**2)
    penalty_a[9] = a[9] * 1.0/(10*D**2)
    daL = (1.0/Nx) * np.einsum('ijk,ik->j',daT,gzT) + gammaA*(0.00+(1/(2*Nx))*la.norm(gzT))*penalty_a

    expgy = np.exp(gy)
    dbgy = np.zeros((Ny,db))
    dbgT = np.zeros((Nx,db))
    dbgy[:,0] = np.ones(Ny)
    dbgT[:,0] = np.ones(Nx)
    dbgy[:,1] = y0
    dbgT[:,1] = T0
    dbgy[:,2] = y1
    dbgT[:,2] = T1
    dbgy[:,3] = 0.5*y0**2
    dbgT[:,3] = 0.5*T0**2
    dbgy[:,4] = y0*y1
    dbgT[:,4] = T0*T1
    dbgy[:,5] = 0.5*y1**2
    dbgT[:,5] = 0.5*T1**2
    dbgy[:,6] = expy1
    dbgT[:,6] = expT1
    dbgy[:,7] = -b[6]*b[7]*argexpy1*expy1
    dbgT[:,7] = -b[6]*b[7]*argexpT1*expT1
    dbgy[:,8] = -b[6]*b[7]**2*y0b8*expy1
    dbgT[:,8] = -b[6]*b[7]**2*T0b8*expT1
    dbgy[:,9] = -b[6]*b[7]**2*y1b9*expy1
    dbgT[:,9] = -b[6]*b[7]**2*T1b9*expT1
    dbgy[:,10] = expy2
    dbgT[:,10] = expT2
    dbgy[:,11] = -b[10]*b[11]*argexpy2*expy2
    dbgT[:,11] = -b[10]*b[11]*argexpT2*expT2
    dbgy[:,12] = -b[10]*b[11]**2*y0b12*expy2
    dbgT[:,12] = -b[10]*b[11]**2*T0b12*expT2
    dbgy[:,13] = -b[10]*b[11]**2*y1b13*expy2
    dbgT[:,13] = -b[10]*b[11]**2*T1b13*expT2
    #gradient w.r.t. b
    penalty_b = np.zeros(db)
    penalty_b[7] = b[7]*eps**2 * np.exp(eps**2 * b[7]**2) - 1.0/(D**2 * b[7]**3)
    penalty_b[11] = b[11]*eps**2 * np.exp(eps**2 * b[11]**2) - 1.0/(D**2 * b[11]**3)
    penalty_b[8] = b[8] * 1.0/(10*D**2)
    penalty_b[9] = b[9] * 1.0/(10*D**2)
    penalty_b[12] = b[12] * 1.0/(10*D**2)
    penalty_b[13] = b[13] * 1.0/(10*D**2)
    dbL = (1.0/Nx)*np.sum(dbgT,axis=0) - (1.0/Ny)*np.einsum('ik,i->k',dbgy,expgy) - gammaB*penalty_b

    #twisted gradient
    G = np.concatenate([daL,-dbL],axis=0)

    #Second derivatives
    gzzT = np.zeros((Nx,d,d))
    gzzT[:,0,0] = b[3] - b[6]*b[7]**2*(1-b[7]**2*T0b8**2)*expT1-b[10]*b[11]**2*(1-b[11]**2*T0b12**2)*expT2
    gzzT[:,1,0] = b[4] - b[6]*b[7]**2*(1-b[7]**2*T0b8*T1b9)*expT1-b[10]*b[11]**2*(1-b[11]**2*T0b12*T1b13)*expT2
    gzzT[:,0,1] = gzzT[:,1,0]
    gzzT[:,1,1] = b[5] - b[6]*b[7]**2*(1-b[7]**2*T1b9**2)*expT1-b[10]*b[11]**2*(1-b[11]**2*T1b13**2)*expT2

    daaT = np.zeros((Nx,da,da,d))
    daaT[:,5,6,0] = a[6]*x0a7*expT*(2-a[6]**2*x0a72)
    daaT[:,6,5,0] = daaT[:,5,6,0]
    daaT[:,5,6,1] = -a[8]**2*a[6]*x0a72*x1a9*expT
    daaT[:,6,5,1] = daaT[:,5,6,1]
    daaT[:,5,7,0] = a[6]**2*expT*(1-a[6]**2*x0a72)
    daaT[:,7,5,0] = daaT[:,5,7,0]
    daaT[:,5,7,1] = -a[8]**2*a[6]**2*x0a7x1a9*expT
    daaT[:,7,5,1] = daaT[:,5,7,1]
    daaT[:,5,8,0] = -a[6]**2*a[8]*x1a92*x0a7*expT
    daaT[:,8,5,0] = daaT[:,5,8,0]
    daaT[:,5,8,1] = a[8]*x1a9*(2-a[8]**2*x1a92)*expT
    daaT[:,8,5,1] = daaT[:,5,8,1]
    daaT[:,5,9,0] = -a[6]**2*a[8]**2*x0a7x1a9*expT
    daaT[:,9,5,0] = daaT[:,5,9,0]
    daaT[:,5,9,1] = a[8]**2*(1-a[8]**2*x1a92)*expT
    daaT[:,9,5,1] = daaT[:,5,9,1]
    daaT[:,6,6,0] = a[5]*x0a7*expT*(2-3*a[6]**2*x0a72-a[6]**2*(2-a[6]**2*x0a72)*x0a72)
    daaT[:,6,6,1] = -a[5]*a[8]**2*x0a72*(1-a[6]**2*x0a72)*x1a9*expT
    daaT[:,6,7,0] = a[5]*a[6]*expT*(2-5*a[6]**2*x0a72+a[6]**4*x0a7**4)
    daaT[:,6,7,1] = -a[5]*a[8]**2*x0a7x1a9*a[6]*(2-a[6]**2*x0a72)*expT
    daaT[:,7,6,0] = daaT[:,6,7,0]
    daaT[:,7,6,1] = daaT[:,6,7,1]
    daaT[:,6,8,0] = -a[5]*a[6]*x0a7*a[8]*x1a92*(2-a[6]**2*x0a72)*expT
    daaT[:,8,6,0] = daaT[:,6,8,0]
    daaT[:,6,8,1] = -a[5]*a[8]*expT*x1a9*x0a72*a[6]*(2-a[8]**2*x1a92)
    daaT[:,8,6,1] = daaT[:,6,8,1]
    daaT[:,6,9,0] = -a[5]*a[6]*expT*x0a7x1a9*a[8]**2*(2-a[6]**2*x0a72)
    daaT[:,9,6,0] = daaT[:,6,9,0]
    daaT[:,6,9,1] = -a[5]*a[8]**2*a[6]*x0a72*(1-a[8]**2*x1a92)*expT
    daaT[:,9,6,1] = daaT[:,6,9,1]
    daaT[:,7,7,0] = -a[5]*a[6]**2*(2*a[6]**2*x0a7+(1-a[6]**2*x0a72)*a[6]**2*x0a7)*expT
    daaT[:,7,7,1] = -a[5]*a[8]**2*x1a9*a[6]**2*(1-a[6]**2*x0a72)*expT
    daaT[:,7,8,0] = -a[5]*a[6]**2*a[8]*x1a92*(1-a[6]**2*x0a72)*expT
    daaT[:,8,7,0] = daaT[:,7,8,0]
    daaT[:,7,8,1] = -a[5]*a[8]*x0a7x1a9*a[6]**2*(2-a[8]**2*x1a92)*expT
    daaT[:,8,7,1] = daaT[:,7,8,1]
    daaT[:,7,9,0] = -a[5]*a[6]**2*a[8]**2*x1a9*(1-a[6]**2*x0a72)*expT
    daaT[:,9,7,0] = daaT[:,7,9,0]
    daaT[:,7,9,1] = -a[5]*a[8]**2*a[6]**2*x0a7*(1-a[8]**2*x1a92)*expT
    daaT[:,9,7,1] = daaT[:,7,9,1]
    daaT[:,8,8,0] = -a[5]*a[6]*x1a92*x0a7*(1-a[8]**2*x1a92)*expT
    daaT[:,8,8,1] = a[5]*x1a9*(2-3*a[8]**2*x1a92-a[8]**2*(2-a[8]**2*x1a92)*x1a92)*expT
    daaT[:,8,9,0] = -a[5]*a[6]**2*x0a7x1a9*a[8]*(2-a[8]**2*x1a92)*expT
    daaT[:,9,8,0] = daaT[:,8,9,0]
    daaT[:,8,9,1] = a[5]*a[8]*(2-5*a[8]**2*x1a92+a[8]**4*x1a9**4)*expT
    daaT[:,9,8,1] = daaT[:,8,9,1]
    daaT[:,9,9,0] = -a[5]*a[6]**2*x0a7*a[8]**2*(1-a[8]**2*x1a92)*expT
    daaT[:,9,9,1] = -a[5]*a[8]**2*(2*a[8]**2*x1a9+(1-a[8]**2*x1a92)*a[8]**2*x1a9)*expT
    #Hessian w.r.t. aa
    penaltyH_a = np.zeros((da,da))
    penaltyH_a[6,6] = eps**2 * np.exp((a[6]*eps)**2)*(1.0+2.0*a[6]**2*eps**2) + 0*3/(D**2 * a[6]**4)
    penaltyH_a[8,8] = eps**2 * np.exp((a[8]*eps)**2)*(1.0+2.0*a[8]**2*eps**2) + 0*3/(D**2 * a[8]**4)
    penaltyH_a[7,7] = 1*1.0/(10*D**2)
    penaltyH_a[9,9] = 1*1.0/(10*D**2)
    daaL = (1.0/Nx)*(np.einsum('ijkl,il->jk',daaT,gzT) + np.einsum('ijl,ilm,ikm->jk',daT,gzzT,daT)) + gammaA*(0.00+(1/(2*Nx))*la.norm(gzT))*penaltyH_a

    dbzgT = np.zeros((Nx,db,d))
    dbzgT[:,1,0]=np.ones(Nx)
    dbzgT[:,2,1]=np.ones(Nx)
    dbzgT[:,3,0]=T0
    dbzgT[:,4,0]=T1
    dbzgT[:,4,1]=T0
    dbzgT[:,5,1]=T1
    dbzgT[:,6,0]=-b[6]**2*T0b8*expT1
    dbzgT[:,6,1]=-b[6]**2*T1b9*expT1
    dbzgT[:,7,0]=-b[6]*b[7]*expT1*T0b8*(2-argexpT1)
    dbzgT[:,7,1]=-b[6]*b[7]*expT1*T1b9*(2-argexpT1)
    dbzgT[:,8,0]=-b[6]*b[7]**2*expT1*(1-b[7]**2*T0b8**2)
    dbzgT[:,8,1]=b[6]*b[7]**4*expT1*T0b8*T1b9
    dbzgT[:,9,0]=dbzgT[:,8,1]
    dbzgT[:,9,1]=-b[6]*b[7]**2*expT1*(1-b[7]**2*T1b9**2)
    dbzgT[:,10,0]=-b[10]**2*T0b12*expT2
    dbzgT[:,10,1]=-b[10]**2*T1b13*expT2
    dbzgT[:,11,0]=-b[10]*b[11]*expT2*T0b12*(2-argexpT2)
    dbzgT[:,11,1]=-b[10]*b[11]*expT2*T1b13*(2-argexpT2)
    dbzgT[:,12,0]=-b[10]*b[11]**2*expT2*(1-b[11]**2*T0b12**2)
    dbzgT[:,12,1]=b[10]*b[11]**4*expT2*T0b12*T1b13
    dbzgT[:,13,0]=dbzgT[:,12,1]
    dbzgT[:,13,1]=-b[10]*b[11]**2*expT2*(1-b[11]**2*T1b13**2)
    # #Hessian w.r.t. ab
    dabL = (1.0/Nx)*np.einsum('ikl,ijl->jk',dbzgT,daT)

    dbbgy = np.zeros((Ny,db,db))
    dbbgT = np.zeros((Nx,db,db))
    dbbgy[:,6,7] = -b[7]*(y0b8**2+y1b9**2)*expy1
    dbbgy[:,7,6] = dbbgy[:,6,7]
    dbbgy[:,6,8] = -b[7]**2*y0b8*expy1
    dbbgy[:,8,6] = dbbgy[:,6,8]
    dbbgy[:,6,9] = -b[7]**2*y1b9*expy1
    dbbgy[:,9,6] = dbbgy[:,6,9]
    dbbgy[:,7,7] = -b[6]*(y0b8**2+y1b9**2)*expy1*(1-argexpy1)
    dbbgy[:,7,8] = -b[6]*b[7]*expy1*y0b8*(2-argexpy1)
    dbbgy[:,8,7] = dbbgy[:,7,8]
    dbbgy[:,7,9] = -b[6]*b[7]*expy1*y1b9*(2-argexpy1)
    dbbgy[:,9,7] = dbbgy[:,7,9]
    dbbgy[:,8,8] = -b[6]*b[7]**2*expy1*(1-y0b8**2*b[7]**2)
    dbbgy[:,8,9] = b[6]*b[7]**4*y0b8*y1b9*expy1
    dbbgy[:,9,8] = dbbgy[:,8,9] 
    dbbgy[:,9,9] = -b[6]*b[7]**2*expy1*(1-y1b9**2*b[7]**2)
    dbbgy[:,10,11] = -b[11]*(y0b12**2+y1b13**2)*expy2
    dbbgy[:,11,10] = dbbgy[:,10,11]
    dbbgy[:,10,12] = -b[11]**2*y0b12*expy2
    dbbgy[:,12,10] = dbbgy[:,10,12]
    dbbgy[:,10,13] = -b[11]**2*y1b13*expy2
    dbbgy[:,13,10] = dbbgy[:,10,13]
    dbbgy[:,11,11] = -b[10]*(y0b12**2+y1b13**2)*expy2*(1-argexpy2)
    dbbgy[:,11,12] = -b[10]*b[11]*expy2*y0b12*(2-argexpy2)
    dbbgy[:,12,11] = dbbgy[:,11,12]
    dbbgy[:,11,13] = -b[10]*b[11]*expy2*y1b13*(2-argexpy2)
    dbbgy[:,13,11] = dbbgy[:,11,13]
    dbbgy[:,12,12] = -b[10]*b[11]**2*expy2*(1-y0b12**2*b[11]**2)
    dbbgy[:,12,13] = b[10]*b[11]**4*y0b12*y1b13*expy2
    dbbgy[:,13,12] = dbbgy[:,12,13] 
    dbbgy[:,13,13] = -b[10]*b[11]**2*expy2*(1-y1b13**2*b[11]**2)
    dbbgT[:,6,7] = -b[7]*(T0b8**2+T1b9**2)*expT1
    dbbgT[:,7,6] = dbbgT[:,6,7]
    dbbgT[:,6,8] = -b[7]**2*T0b8*expT1
    dbbgT[:,8,6] = dbbgT[:,6,8]
    dbbgT[:,6,9] = -b[7]**2*T1b9*expT1
    dbbgT[:,9,6] = dbbgT[:,6,9]
    dbbgT[:,7,7] = -b[6]*(T0b8**2+T1b9**2)*expT1*(1-argexpT1)
    dbbgT[:,7,8] = -b[6]*b[7]*expT1*T0b8*(2-argexpT1)
    dbbgT[:,8,7] = dbbgT[:,7,8]
    dbbgT[:,7,9] = -b[6]*b[7]*expT1*T1b9*(2-argexpT1)
    dbbgT[:,9,7] = dbbgT[:,7,9]
    dbbgT[:,8,8] = -b[6]*b[7]**2*expT1*(1-T0b8**2*b[7]**2)
    dbbgT[:,8,9] = b[6]*b[7]**4*T0b8*T1b9*expT1
    dbbgT[:,9,8] = dbbgT[:,8,9] 
    dbbgT[:,9,9] = -b[6]*b[7]**2*expT1*(1-T1b9**2*b[7]**2)
    dbbgT[:,10,11] = -b[11]*(T0b12**2+T1b13**2)*expT2
    dbbgT[:,11,10] = dbbgT[:,10,11]
    dbbgT[:,10,12] = -b[11]**2*T0b12*expT2
    dbbgT[:,12,10] = dbbgT[:,10,12]
    dbbgT[:,10,13] = -b[11]**2*T1b13*expT2
    dbbgT[:,13,10] = dbbgT[:,10,13]
    dbbgT[:,11,11] = -b[10]*(T0b12**2+T1b13**2)*expT2*(1-argexpT2)
    dbbgT[:,11,12] = -b[10]*b[11]*expT2*T0b12*(2-argexpT2)
    dbbgT[:,12,11] = dbbgT[:,11,12]
    dbbgT[:,11,13] = -b[10]*b[11]*expT2*T1b13*(2-argexpT2)
    dbbgT[:,13,11] = dbbgT[:,11,13]
    dbbgT[:,12,12] = -b[10]*b[11]**2*expT2*(1-T0b12**2*b[11]**2)
    dbbgT[:,12,13] = b[10]*b[11]**4*T0b12*T1b13*expT2
    dbbgT[:,13,12] = dbbgT[:,12,13] 
    dbbgT[:,13,13] = -b[10]*b[11]**2*expT2*(1-T1b13**2*b[11]**2)

    dbgdbg = np.einsum('jn,jm->jnm',dbgy,dbgy)
    #Hessian w.r.t. bb
    penaltyH_b = np.zeros((db,db))
    # penaltyH_b[5,5] = 1*1.0/(10*D**2)+1*6*eps**2 *(b[5]-b[8])**(-4)
    # penaltyH_b[8,8] = 1*1.0/(10*D**2)+1*6*eps**2 *(b[5]-b[8])**(-4)
    # penaltyH_b[5,8] = -1*6*eps**2 *(b[5]-b[8])**(-4)
    # penaltyH_b[8,5] = -1*6*eps**2 *(b[5]-b[8])**(-4)
    penaltyH_b[7,7] = eps**2 * np.exp((b[7]*eps)**2)*(1.0+2.0*b[7]**2*eps**2) + 3/(D**2 * b[7]**4)
    penaltyH_b[8,8] = 1*1.0/(10*D**2)
    penaltyH_b[9,9] = 1*1.0/(10*D**2)
    penaltyH_b[11,11] = eps**2 * np.exp((b[11]*eps)**2)*(1.0+2.0*b[11]**2*eps**2) + 3/(D**2 * b[11]**4)
    penaltyH_b[12,12] = 1*1.0/(10*D**2)
    penaltyH_b[13,13] = 1*1.0/(10*D**2)
    dbbL = (1.0/Nx)*np.sum(dbbgT,axis=0) - (1.0/Ny)*np.einsum('jnm,j->nm',(dbbgy+dbgdbg),expgy) - gammaB*penaltyH_b

    # #Twisted Hessian
    H = np.zeros((da+db,da+db))
    H[:da,:da] = daaL
    H[:da,da:] = dabL
    H[da:,:da] = -dabL.T
    H[da:,da:] = -dbbL
    return(G,H)