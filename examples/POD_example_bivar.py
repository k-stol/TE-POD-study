##################################################################
# Compute transfter entropy (TE) between two time coeff.s of POD modes
# Daniele Massaro
##################################################################
#
import sys
import os
import numpy as np
import math as mt
import matplotlib.pyplot as plt
import scipy.io
from idtxl.data import Data
from idtxl.bivariate_te import BivariateTE
sys.path.append('./')
import math as mt


def test(a,b,tausMIN,tausMAX,tautMAX):
    """
    Compute TE between source and target:
        - a = source
        - b = target
        - tausMIN = min time lag for the source
        - tausMAX = max time lag for the source
        - tautMAX = max time lag for the target
    Args:
       `a,b`: numpy array of n samples
       `tausMIN,tausMAX,tautMAX`: int, time lag for source and target
    Returns:
        te_bivar: int, TE value
        corr1: numpy matrix

    """    
    # (1) Initialisation
    n = np.size(a)
    x=np.ones((n,1))
    y=np.zeros((n,1))
    
    
    x[:,0] = a
    y[:,0] = b

    # (2) Pearson's correlation between samples of x and y
    corr1=np.corrcoef(x[:,0],y[:,0])
    print("Pearson's correlation between Xt-Yt: ",corr1[0,1])
    corr2=np.corrcoef(x[:-1,0],y[1:,0])
    print("Pearson's correlation between Xt-1-Yt: ",corr2[0,1])
    corr3=np.corrcoef(x[:-2,0],y[2:,0])
    print("Pearson's correlation between Xt-2-Yt: ",corr3[0,1])
    corr4=np.corrcoef(x[1:,0],y[:-1,0])
    print("Pearson's correlation between Xt-Yt-1: ",corr4[0,1])
    
    # (3) construct a IDTxl Data object
    data=Data(np.hstack((x,y)),
          dim_order='sp',
          normalise=False)

    # (4) estimate TE
    settings = {
        'cmi_estimator': 'JidtKraskovCMI',
        'n_perm_max_stat': 100,
        'n_perm_min_stat': 100,
        'n_perm_max_seq': 100,
        'n_perm_omnibus': 100,
        'max_lag_sources': tausMAX, #int(taus+5),
        'min_lag_sources': tausMIN, #int(taus),
        'max_lag_target':  tautMAX,
        'verbose': True,
        'write_ckp': True}
    nw_bivar = BivariateTE()
    results = nw_bivar.analyse_single_target(
                   settings, data, target=1, sources=0)      #indices refer to the order in Data stacking
    te_bivar = results.get_single_target(1, fdr=False)#['te']#[0]  #first arg is the index of target in Data

    print('te_bivar:',te_bivar)
    
    return te_bivar,corr1

if __name__ == "__main__":    

    #-------- SETTINGS ----------------
    #Directory with the data 
    dataDir ="TEST_CASE/"


    # POD-mode 0 is the mean
    # POD-mode source
    mode_s = 1
    # POD-mode target
    mode_t = 4
    
    min_lag_source = 1
    
    max_lag_target = 20 
    
    max_lag_source = 20 #[1,2,4,6,8,10,15,20] 
    #----------------------------------
    
    mat1 = scipy.io.loadmat(dataDir+'A2coef.mat')
    # mat1 = np.load(dataDir+'vel').item()
    tmcoeff = mat1['A2']
    
    s_ = tmcoeff[:,mode_s]
    t_ = tmcoeff[:,mode_t]
    
    a,corr = test(s_,t_,min_lag_source,max_lag_source,max_lag_target)
    b = a.omnibus_te  
    CR_v = corr[0,1]
    TE_v = b    

    

