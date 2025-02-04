################################################################################
# MIT License
#
# Copyright (c) 2021 University of Amsterdam
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning for Medical Imaging Amsterdam UMC Oliver Gurney-Champion | Spring 2023
# Date modified: Jan 2023
################################################################################

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

def ivim(bvalues, Dt, Fp, Ds, S0):
    # regular IVIM function
    return (S0 * (Fp * np.exp(-bvalues * Ds) + (1 - Fp) * np.exp(-bvalues * Dt)))

def sim_signal(SNR=(20,40), bvalues=[0, 1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 40, 50, 75, 100, 150, 300, 500, 700, 850, 1000], sims=100000, Dmin=0.5 / 1000, Dmax=3.0 / 1000, fmin=0.0, fmax=0.7, Dsmin=0.005, Dsmax=0.1,fp2min=0, fp2max=0.3,Ds2min=0.2,Ds2max=0.5,
               rician=False,seed=123):
    """
    This simulates IVIM curves. Data is simulated by randomly selecting a value of D, f and D* from within the
    predefined range.

    input:
    :param SNR: SNR of the simulated data. If SNR is set to 0, no noise is added
    :param bvalues: 1D Array of b-values used
    :param sims: number of simulations to be performed (need a large amount for training)

    optional:
    :param Dmin: minimal simulated D. Default = 0.0005
    :param Dmax: maximal simulated D. Default = 0.002
    :param fmin: minimal simulated f. Default = 0.1
    :param Dmax: minimal simulated f. Default = 0.5
    :param Dsmin: minimal simulated D*. Default = 0.05
    :param Dsmax: minimal simulated D*. Default = 0.2
    :param rician: boolean giving whether Rician noise is used; default = False

    :return data_sim: 2D array with noisy IVIM signal (x-axis is sims long, y-axis is len(b-values) long)
    :return D: 1D array with the used D for simulations, sims long
    :return f: 1D array with the used f for simulations, sims long
    :return Ds: 1D array with the used D* for simulations, sims long
    """

    # randomly select parameters from predefined range
    rg = np.random.RandomState(seed)
    test = rg.uniform(0, 1, (sims, 1))
    D = Dmin + (test * (Dmax - Dmin))
    test = rg.uniform(0, 1, (sims, 1))
    f = fmin + (test * (fmax - fmin))
    test = rg.uniform(0, 1, (sims, 1))
    Ds = Dsmin + (test * (Dsmax - Dsmin))

    # initialise data array
    data_sim = np.zeros([len(D), len(bvalues)])
    bvalues = np.array(bvalues)

    if type(SNR) == tuple:
        test = rg.uniform(0, 1, (sims, 1))
        SNR = np.exp(np.log(SNR[1]) + (test * (np.log(SNR[0]) - np.log(SNR[1]))))
        addnoise = True
    elif SNR == 0:
        addnoise = False
    else:
        SNR = SNR * np.ones_like(Ds)
        addnoise = True

    # loop over array to fill with simulated IVIM data
    data_sim = ivim(bvalues, D, f, Ds, 1)
    # if SNR is set to zero, don't add noise
    if addnoise:
        # fill arrays
        noise_real = rg.normal(0, 1, (sims, len(bvalues)))
        noise_imag = rg.normal(0, 1, (sims, len(bvalues)))
        noise_real = noise_real / SNR
        noise_imag = noise_imag / SNR
        if rician:
            data_sim = np.sqrt(np.power(data_sim + noise_real, 2) + np.power(noise_imag, 2))
        else:
            # or add Gaussian noise
            data_sim = data_sim + noise_imag
    else:
        data_sim = data_sim

    # normalise signal
    S0_noisy = np.mean(data_sim[:, bvalues == 0], axis=1)
    data_sim = data_sim / S0_noisy[:, None]
    return np.squeeze(data_sim).astype('float32'), np.squeeze(D).astype('float32'), np.squeeze(f).astype('float32'), np.squeeze(Ds).astype('float32')


def load_real_data(eval=True):

    ### folder patient data
    folder = 'data'

    ### load patient data
    print('Load patient data \n')

    # load and init b-values
    text_file = np.genfromtxt('{folder}/bvalues.bval'.format(folder=folder))
    bvalues = np.array(text_file)

    # load nifti
    data = nib.load('{folder}/data.nii.gz'.format(folder=folder))
    datas = data.get_fdata()

    # reshape image for fitting
    sx, sy, sz, n_b_values = datas.shape
    ub = np.unique(bvalues)
    selsb = np.array(ub) == 0
    datan=np.zeros([sx,sy,sz,len(ub)])
    #only load first b-value image per b-value
    for b in ub:
        datan[:,:,:,ub==b]=np.reshape(datas[:,:,:,bvalues==b][:,:,:,0],[sx,sy,sz,1])
    datas=datan
    if eval:
        data_slice=datas[:,:,8,:]
        X_dw = np.reshape(data_slice, (sx * sy, len(ub)))
    else:
        datas=np.delete(datas,8,2)
        X_dw = np.reshape(datas, (sx * sy * (sz-1), len(ub)))

    ### select only relevant values, delete background and noise, and normalise data
    S0 = np.nanmean(X_dw[:, selsb], axis=1)
    S0[S0 != S0] = 0
    S0 = np.squeeze(S0)
    valid_id = (S0 > (0.5 * np.median(S0[S0 > 0])))
    datatot = X_dw[valid_id, :]
    # normalise data
    S0 = np.nanmean(datatot[:, selsb], axis=1).astype('<f')
    datatot = datatot / S0[:, None]
    print('Patient data loaded\n')
    if eval:
        return datatot.astype('float32'), valid_id, ub
    else:
        return datatot.astype('float32'), ub


def plot_example(input, valid_id,vmax=0.7):
    ### folder patient data
    folder = 'data'

    ### load patient data
    print('Load patient data \n')

    # load and init b-values
    data = nib.load('{folder}/data.nii.gz'.format(folder=folder))
    datas = data.get_fdata()
    # reshape image for fitting
    sx, sy, sz, n_b_values = datas.shape
    img = np.zeros([sx * sy])
    img[valid_id] = input
    img = np.reshape(img, [sx, sy])
    plt.imshow(img,vmin=0, vmax=vmax,cmap='gray')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    pass

def plot_ref():
    folder = 'data'
    dataD = nib.load('{folder}/D.nii.gz'.format(folder=folder))
    dataf = nib.load('{folder}/f.nii.gz'.format(folder=folder))
    dataDp = nib.load('{folder}/Dp.nii.gz'.format(folder=folder))

    plt.imshow(dataD.get_fdata()[:,:,8],vmax=0.003,cmap='gray')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.imshow(dataf.get_fdata()[:,:,8],vmax=0.7,cmap='gray')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    plt.imshow(dataDp.get_fdata()[:,:,8],vmax=0.1,cmap='gray')
    plt.ion()
    plt.show()
    plt.pause(0.001)
    pass

def error_metrics(par_pred,par_ref):
    """
    Computes the random and systematic errors from the prediction.
    Args:
      par_pred: 1D float array of size [n], predictions of the model
      par_ref: 1D float array of size [n]. Ground truth reference for each sample
    Returns:
      CV: random error (coefficient of variation)
      Sys: systematic error
    """

    CV = np.std(par_pred-par_ref)
    sys = np.mean(par_pred-par_ref)

    return CV, sys