# -*- coding: utf-8 -*-
__authors__ = 'Francesco Ciucci, Ting Hei Wan, Adeleke Maradesa, Baptiste Py'
__date__ = '28th June 2024'

"""
This file stores all the functions that are shared by all three DRT methods, i.e., simple, Bayesian, and Bayesian Hilbert Transform.
References: 
    [1] T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta. 184 (2015) 483-499.
    [2] M. Saccoccio, T. H. Wan, C. Chen,F. Ciucci, Optimal regularization in distribution of relaxation times applied to electrochemical impedance spectroscopy: Ridge and lasso regression methods - A theoretical and experimental study, Electrochimica Acta. 147 (2014) 470-482.
    [3] J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
    [4] A. Maradesa, B. Py, T.H. Wan, M.B. Effat, F. Ciucci, Selecting the regularization parameter in the distribution of relaxation times, Journal of the Electrochemical Society. 170 (2023) 030502.
"""

# Maths and data related packages
import numpy as np
import sys
from numpy import log, log10, sqrt
import pandas as pd
from math import pi
from scipy.optimize import differential_evolution, minimize
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import importlib
from .qp_solver import solve_nonnegative_qp
try:
    from cvxopt import matrix, solvers
except Exception:
    def matrix(x):
        return np.asarray(x, dtype=float)
    class _FallbackSolvers:
        @staticmethod
        def qp(H, c, G=None, h=None):
            x = solve_nonnegative_qp(np.asarray(H, dtype=float), np.asarray(c, dtype=float).reshape(-1))
            return {'x': x.reshape(-1, 1), 'status': 'optimal'}
    solvers = _FallbackSolvers()

# pyDRTtools related package
#
from . import peak_analysis as peaks
from .importer import read_eis_file
# from . import parameter_selection as param
from . import basics
from . import nearest_PD as nPD
import importlib
importlib.reload(basics)
importlib.reload(peaks)
# importlib.reload(param)
#from . import deep_learning as deep
from . import BHT 
from . import HMC
import time 

class EIS_object(object):
    
    # The EIS_object class stores the input data and the DRT result.
      
    def __init__(self, freq, Z_prime, Z_double_prime):
        
        """
        This is EIS_object class 
        Inputs:
            freq: frequency of the EIS measurement
            Z_prime: real part of the impedance
            Z_double_prime: imaginery part of the impedance
        """
        # define an EIS_object
        self.freq = freq
        self.Z_prime = Z_prime
        self.Z_double_prime = Z_double_prime
        self.Z_exp = Z_prime + 1j*Z_double_prime
        
        # keep a copy of the original data
        self.freq_0 = freq
        self.Z_prime_0 = Z_prime
        self.Z_double_prime_0 = Z_double_prime
        self.Z_exp_0 = Z_prime + 1j*Z_double_prime

        self.tau = 1/freq  # original pyDRTtools collocation points
        self._reset_tau_fine()
        self.method = 'none'
        self.original_input_name = None
        self.source_path = None

    def _reset_tau_fine(self):
        """Set the physical relaxation-time range using tau = 1/(2*pi*f)."""
        f_max = float(np.max(self.freq_0))
        f_min = float(np.min(self.freq_0))
        taumin_base = log10(1/(2*pi*f_max))
        taumax_base = log10(1/(2*pi*f_min))
        base_margin = 0.5
        target_log10_tau = 1/np.log(10.0)
        taumin = min(taumin_base-base_margin, target_log10_tau-0.1)
        taumax = max(taumax_base+base_margin, target_log10_tau+0.1)
        self.tau_fine = np.logspace(taumin, taumax, 10*self.freq_0.shape[0])
    
    @classmethod
    def from_file(cls, filename):
        data = read_eis_file(filename)
        obj = cls(data[:, 0], data[:, 1], data[:, 2])
        from pathlib import Path
        path = Path(filename)
        obj.original_input_name = path.stem
        obj.source_path = str(path)
        return obj

    def plot_DRT(self): # plot the DRT result
        
        basics.pretty_plot(4,4)
        plt.rc('font', family='serif', size=15)
        plt.rc('xtick', labelsize=15)
        plt.rc('ytick', labelsize=15)
        plt.rc('text', usetex=True)    
        
        if self.method == 'simple':    
            plt.plot(self.out_tau_vec, self.gamma, 'k')
            y_min = 0
            y_max = max(self.gamma)
            
        elif self.method == 'credit':
            plt.fill_between(self.out_tau_vec, self.lower_bound, self.upper_bound,  facecolor='lightgrey')
            plt.plot(self.out_tau_vec, self.gamma, color='black', label='MAP')
            plt.plot(self.out_tau_vec, self.mean, color='blue', label='mean')
            plt.plot(self.out_tau_vec, self.lower_bound, color='black', linewidth=1)
            plt.plot(self.out_tau_vec, self.upper_bound, color='black', linewidth=1)
            plt.legend(frameon=False, fontsize = 15)
            y_min = 0
            y_max = max(self.upper_bound)
            
        elif self.method == 'BHT':    
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_re, 'b', linewidth=1)
            plt.semilogx(self.out_tau_vec, self.mu_gamma_fine_im, 'k', linewidth=1)
            y_min = min(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
            y_max = max(np.concatenate((self.mu_gamma_fine_re, self.mu_gamma_fine_im)))
        
        else:
            return
        
        plt.xscale('log')
        plt.xlim(self.out_tau_vec.min(), self.out_tau_vec.max())
        plt.ylim(y_min, y_max*1.1)
        plt.xlabel(r'$f/{\rm Hz}$', fontsize=20)
        plt.ylabel(r'$\gamma(\tau)/\Omega$', fontsize=20)
    
        plt.show()

# 

def simple_run(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', reg_param = 1E-3, shape_control = 'FWHM Coefficient', coeff = 0.5):
    
    
    """
    This function enables to compute the DRT using ridge regression (also known as Tikhonov regression)
    References:
        T. H. Wan, M. Saccoccio, C. Chen, F. Ciucci, Influence of the discretization methods on the distribution of relaxation times deconvolution: Implementing radial basis functions with DRTtools, Electrochimica Acta 184 (2015) 483-499.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type 
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
    """
    
    # Step 1.1: define the optimization bounds
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    ###
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    # Step 1.2: compute epsilon
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)
    
    # Step 1.3: compute A matrix
    ## assemble_A_re(freq_vec, tau_vec, epsilon, rbf_type)
    entry.A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type)
    entry.A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type)
    
    # Step 1.4: compute M matrix  assemble_M_1(tau_vec, epsilon, rbf_type)
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type)
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type)
    
    # Step 2: conduct ridge regularization
    if data_used == 'Combined Re-Im Data': # select both parts of the impedance for the simple run
 
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 1 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:,N_RL:] = entry.A_re_temp
            entry.A_re[:,0] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:,N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            # initial guess for the hyperparameter
            log_lambda_0 = log(reg_param) # initial guess for lambda
            #
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler, it will be used if needed
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
            Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
           
        elif induct_used == 1: # considering the inductance
            N_RL = 2
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            entry.A_re[:,1] = 1
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq

            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_combined,c_combined = basics.quad_format_combined(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraint # N_RL
            ## bound matrix
            G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_combined), matrix(c_combined),G,h)
            x = np.array(sol['x']).flatten()

            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im

            # only consider std of residuals in both parts
            sigma_re_im = np.std(np.concatenate([entry.res_re,entry.res_im]))
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
            Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_re.T@inv_V@entry.b_re + entry.A_im.T@inv_V@entry.b_im
            
    elif data_used == 'Im Data': # select the imaginary part of the impedance for the simple run
        
        if induct_used == 0 or induct_used == 2: # without considering the inductance
            N_RL = 0 # N_RL length of resistance plus inductance
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 
                
            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            # enforce positivity constraints
            ## bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im
            
            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
            
            Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_im.T@inv_V@entry.b_im

            
        elif induct_used == 1: # considering the inductance
            N_RL = 1
            entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_re[:, N_RL:] = entry.A_re_temp
            
            entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
            entry.A_im[:, N_RL:] = entry.A_im_temp
            entry.A_im[:,0] = 2*pi*entry.freq
            
            entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
            entry.M[N_RL:,N_RL:] = entry.M_temp
            
            # optimally select the regularization level
            log_lambda_0 = log(reg_param) # initial guess for lambda
            if cv_type=='custom':
                entry.lambda_value = reg_param
            else:
                entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 

            print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
            
            # recover the DRT using cvxopt
            
            H_im, c_im = basics.quad_format_separate(entry.A_im, entry.b_im, entry.M, entry.lambda_value)
            #
            # enforce positivity constraints
            # bound matrix
            G = matrix(-np.identity(entry.b_im.shape[0]+N_RL))
            h = matrix(np.zeros(entry.b_im.shape[0]+N_RL))
            # Formulate the quadratic programming problem
            ##
            # Solve the quadratic programming problem
            sol = solvers.qp(matrix(H_im), matrix(c_im),G,h)
            x = np.array(sol['x']).flatten()

            # prepare for HMC sampler
            entry.mu_Z_re = entry.A_re@x
            entry.mu_Z_im = entry.A_im@x
            entry.res_re = entry.mu_Z_re-entry.b_re
            entry.res_im = entry.mu_Z_im-entry.b_im
            
            # only consider std of residuals in the imaginary part
            sigma_re_im = np.std(entry.res_im)
            inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
            
            Sigma_inv = (entry.A_im.T@inv_V@entry.A_im) + (entry.lambda_value/sigma_re_im**2)*entry.M
            mu_numerator = entry.A_im.T@inv_V@entry.b_im

    elif data_used == 'Re Data': # select the real part of the impedance for the simple run
        N_RL = 1
        entry.A_re = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_re[:, N_RL:] = entry.A_re_temp
        entry.A_re[:,0] = 1
        
        entry.A_im = np.zeros((N_freqs, N_taus+N_RL))
        entry.A_im[:, N_RL:] = entry.A_im_temp

        entry.M = np.zeros((N_taus+N_RL, N_taus+N_RL))
        entry.M[N_RL:,N_RL:] = entry.M_temp
        
        # optimally select the regularization level
        log_lambda_0 = log(reg_param) # initial guess for lambda
        if cv_type=='custom':
            entry.lambda_value = reg_param
        else:
            entry.lambda_value = basics.optimal_lambda(entry.A_re, entry.A_im, entry.b_re, entry.b_im, entry.M, data_used, induct_used, log_lambda_0, cv_type) 

        print('The value of the regularization parameter is', entry.lambda_value) # to check the value of lambda
        
        # recover the DRT using cvxopt 
        H_re,c_re = basics.quad_format_separate(entry.A_re, entry.b_re, entry.M, entry.lambda_value)
    
        # enforce positivity constraints
        # ## bound matrix
        G = matrix(-np.identity(entry.b_re.shape[0]+N_RL))
        h = matrix(np.zeros(entry.b_re.shape[0]+N_RL))
        # Formulate the quadratic programming problem
        ###
        # Solve the quadratic programming problem
        sol = solvers.qp(matrix(H_re), matrix(c_re),G,h)
        x = np.array(sol['x']).flatten()

        # prepare for HMC sampler
        entry.mu_Z_re = entry.A_re@x
        entry.mu_Z_im = entry.A_im@x       
        entry.res_re = entry.mu_Z_re-entry.b_re
        entry.res_im = entry.mu_Z_im-entry.b_im
        
        # only consider std of residuals in the real part
        sigma_re_im = np.std(entry.res_re)
        inv_V = 1/sigma_re_im**2*np.eye(N_freqs)
        
        Sigma_inv = (entry.A_re.T@inv_V@entry.A_re) + (entry.lambda_value/sigma_re_im**2)*entry.M
        mu_numerator = entry.A_re.T@inv_V@entry.b_re

    entry.Sigma_inv = (Sigma_inv+Sigma_inv.T)/2
    
    # test if the covariance matrix is positive definite
    if (nPD.is_PD(entry.Sigma_inv)==False):
        entry.Sigma_inv = nPD.nearest_PD(entry.Sigma_inv) # if not, use the nearest positive definite matrix
    
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    entry.mu = np.linalg.solve(L_Sigma_inv, mu_numerator)
    entry.mu = np.linalg.solve(L_Sigma_inv.T, entry.mu)
    # entry.mu = np.linalg.solve(entry.Sigma_inv, mu_numerator)
    
    # Step 3: obtaining the result of inductance, resistance, and gamma  
    if N_RL == 0: 
        entry.L, entry.R = 0, 0        
    elif N_RL == 1 and data_used == 'Im Data':
        entry.L, entry.R = x[0], 0    
    elif N_RL == 1 and data_used != 'Im Data':
        entry.L, entry.R = 0, x[0]
    elif N_RL == 2:
        entry.L, entry.R = x[0:2]
        
    entry.x = x[N_RL:]
    entry.out_tau_vec, entry.gamma = basics.x_to_gamma(x[N_RL:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.N_RL = N_RL 
    entry.method = 'simple'
    
    return entry


def Bayesian_run(entry, rbf_type = 'Gaussian', data_used = 'Combined Re-Im Data', induct_used = 1, der_used = '1st order', cv_type = 'GCV', reg_param = 1E-3, shape_control = 'FWHM Coefficient', coeff = 0.5, NMC_sample = 2000):
    
    """
    This function enables to recover the DRT with its uncertainty in a Bayesian framework. 
    References:
        F. Ciucci, C. Chen, Analysis of electrochemical impedance spectroscopy data using the distribution of relaxation times: A Bayesian and hierarchical Bayesian approach, Electrochimica Acta 167 (2015) 439-454.
        M. B. Effat, F. Ciucci, Bayesian and hierarchical Bayesian based regularization for deconvolving the distribution of relaxation times from electrochemical impedance spectroscopy data, Electrochimica Acta 247 (2017) 1117-1129.
    Inputs:
        entry: an EIS spectrum
        rbf_type: discretization function
        data_used: part of the EIS spectrum used for regularization
        induct_used: treatment of the inductance part
        der_used: order of the derivative considered for the M matrix
        cv_type: regularization method used to select the regularization parameter for ridge regression
        reg_param: regularization parameter applied when "custom" is used for cv_type  
        shape_control: option for controlling the shape of the radial basis function (RBF) 
        coeff: magnitude of the shape control
        NMC_sample: number of samples for the HMC sampler
    """
    
    NMC_sample = int(NMC_sample)
    if NMC_sample < 502:
        raise ValueError("Number of Samples must be at least 502 because the first 501 samples are discarded as burn-in.")

    simple_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used = induct_used, 
               der_used=der_used, cv_type=cv_type, reg_param=reg_param, shape_control = shape_control, coeff=coeff) 

    # using HMC sampler to sample the truncated Gaussian distribution
    
    # object_A.plot_DRT()
    entry.mu = entry.mu[entry.N_RL:] # reshape to avoid error as
    entry.Sigma_inv = entry.Sigma_inv[entry.N_RL:,entry.N_RL:]
    
    # Step 1: Cholesky Transform instead of direct inverse 
    L_Sigma_inv = np.linalg.cholesky(entry.Sigma_inv)
    L_Sigma_agm = np.linalg.inv(L_Sigma_inv)
    entry.Sigma = L_Sigma_agm.T@L_Sigma_agm
    
    # Step 2: set up the boundary constraints
    F = np.eye(entry.x.shape[0])
    g = np.finfo(float).eps*np.ones(entry.mu.shape[0])
    initial_X = entry.x
    
    # Step 3: use generate_tmg from HMC_exact.py to sample the truncated Gaussian distribution
    entry.Xs = HMC.generate_tmg(F, g, entry.Sigma, entry.mu, initial_X, cov=True, L=NMC_sample)
    entry.lower_bound = np.quantile(entry.Xs[:,501:],.005,axis=1)
    entry.upper_bound = np.quantile(entry.Xs[:,501:],.995,axis=1)
    entry.mean = np.mean(entry.Xs[:,501:],axis=1)    
    
    # Step 4: map array to gamma
    entry.out_tau_vec,entry.lower_bound = basics.x_to_gamma(entry.lower_bound, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.out_tau_vec,entry.upper_bound = basics.x_to_gamma(entry.upper_bound, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    entry.out_tau_vec,entry.mean = basics.x_to_gamma(entry.mean, entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    
    entry.method = 'credit'

    return entry

# Hilbert run

def BHT_run(entry, rbf_type = 'Gaussian', der_used = '1st order', shape_control = 'FWHM Coefficient', coeff = 0.5):
    
    """
    This function enables to assess the compliance of an EIS spectrum to the Kramers-Kronig relations.
    References: 
       J. Liu, T. H. Wan, F. Ciucci, A Bayesian view on the Hilbert transform and the Kramers-Kronig transform of electrochemical impedance data: Probabilistic estimates and quality scores, Electrochimica Acta. 357 (2020) 136864.
       F. Ciucci, The Gaussian process hilbert transform (GP-HT): Testing the consistency of electrochemical impedance spectroscopy data, Journal of the Electrochemical Society. 167-12 (2020) 126503.
    Inputs:
       entry: an EIS spectrum
       rbf_type: discretization function
       der_used: order of the derivative considered for the M matrix
       shape_control: option for controlling the shape of the radial basis function (RBF) 
       coeff: magnitude of the shape control           
    """   
    
    omega_vec = 2*pi*entry.freq
    N_freqs = entry.freq.shape[0]
    N_taus = entry.tau.shape[0]
    
    # Step 1: construct the A matrix
    entry.epsilon = basics.compute_epsilon(entry.freq, coeff, rbf_type, shape_control)
    
    A_re_temp = basics.assemble_A_re(entry.freq, entry.tau, entry.epsilon, rbf_type)
    A_im_temp = basics.assemble_A_im(entry.freq, entry.tau, entry.epsilon, rbf_type)
    
    # add resistance column and inductance column to A_re and A_im
    entry.A_re = np.append(np.ones([N_freqs,1]), A_re_temp, axis=1)
    entry.A_im = np.append(omega_vec.reshape(N_freqs,1), A_im_temp, axis=1)
    entry.A_H_re = A_re_temp
    entry.A_H_im = A_im_temp  
    entry.b_re = entry.Z_exp.real
    entry.b_im = entry.Z_exp.imag
    
    # Step 2: construct the M matrix
    if der_used == '1st order':
        entry.M_temp = basics.assemble_M_1(entry.tau, entry.epsilon, rbf_type)
    
    elif der_used == '2nd order':
        entry.M_temp = basics.assemble_M_2(entry.tau, entry.epsilon, rbf_type)
    entry.M = np.zeros((N_taus+1, N_taus+1))
    entry.M[1:,1:] = entry.M_temp 
    
    # Step 3: perform Hilbert transform estimation (try for "max_attempts" until no error occur for the HT_single_est)
    max_attempts = 20 # number of attempt
    for attempt in range(max_attempts):
        try:
            # generate a random initial guess for theta_0 in the range
            theta_0 = 10**(6 * np.random.rand(3) - 3)
            # perform the Hilbert transform estimation for the real part of the impedance data
            out_dict_real = BHT.HT_single_est(theta_0, entry.Z_exp.real, entry.A_re, entry.A_H_im, entry.M, N_freqs, N_taus)
            # update theta_0 based on the result of the real part estimation
            theta_0 = out_dict_real['theta']
            # perform the Hilbert transform estimation for the imaginary part of the impedance data
            out_dict_imag = BHT.HT_single_est(theta_0, entry.Z_exp.imag, entry.A_im, entry.A_H_re, entry.M, N_freqs, N_taus)

            # exit the loop if both estimations are successful
            break  
        except Exception as e:
            # print the error message and the attempt number if an error occurs
            print(f'Error Occurred: {e}. Attempt {attempt + 1}/{max_attempts}. Trying another initial condition.')
    else:
        raise RuntimeError(f"Failed to execute successfully after {max_attempts} attempts.")
    
    # Step 4: score the EIS
    entry.out_scores = BHT.EIS_score(theta_0, entry.freq, entry.Z_exp, out_dict_real, out_dict_imag, N_MC_samples=10000)
    
    # Step 5: display the bands and the Hilbert fitting of the real and the imaginary parts
    
    # Step 5.1: Real part
    
    # Step 5.1.1: Bayesian regression
    entry.mu_Z_re = out_dict_real.get('mu_Z')
    entry.cov_Z_re = np.diag(out_dict_real.get('Sigma_Z'))

    entry.mu_R_inf = out_dict_real.get('mu_gamma')[0]
    entry.cov_R_inf = np.diag(out_dict_real.get('Sigma_gamma'))[0]

    # Step 5.1.2: DRT part
    entry.mu_Z_DRT_re = out_dict_real.get('mu_Z_DRT')
    entry.cov_Z_DRT_re = np.diag(out_dict_real.get('Sigma_Z_DRT'))

    # Step 5.1.3: HT prediction
    entry.mu_Z_H_im = out_dict_real.get('mu_Z_H')
    entry.cov_Z_H_im = np.diag(out_dict_real.get('Sigma_Z_H'))

    # Step 5.1.4: sigma_n estimation
    entry.sigma_n_re = out_dict_real.get('theta')[0]

    # Step 5.1.5: mu_gamma estimation
    entry.mu_gamma_re = out_dict_real.get('mu_gamma')
    entry.out_tau_vec,entry.mu_gamma_fine_re = basics.x_to_gamma(entry.mu_gamma_re[1:],entry.tau_fine,entry.tau, entry.epsilon, rbf_type)
    
    # Step 5.2: Imaginary part
    
    # Step 5.2.1: Bayesian regression
    entry.mu_Z_im = out_dict_imag.get('mu_Z')
    entry.cov_Z_im = np.diag(out_dict_imag.get('Sigma_Z'))

    entry.mu_L_0 = out_dict_imag.get('mu_gamma')[0]
    entry.cov_L_0 = np.diag(out_dict_imag.get('Sigma_gamma'))[0]

    # Step 5.2.2: DRT part
    entry.mu_Z_DRT_im = out_dict_imag.get('mu_Z_DRT')
    entry.cov_Z_DRT_im = np.diag(out_dict_imag.get('Sigma_Z_DRT'))
    
    # Step 5.2.3: HT prediction
    entry.mu_Z_H_re = out_dict_imag.get('mu_Z_H')
    entry.cov_Z_H_re = np.diag(out_dict_imag.get('Sigma_Z_H'))

    # Step 5.2.4: sigma_n estimation
    entry.sigma_n_im = out_dict_imag.get('theta')[0]

    # Step 5.2.5: mu_gamma estimation
    entry.mu_gamma_im = out_dict_imag.get('mu_gamma')
    entry.out_tau_vec,entry.mu_gamma_fine_im = basics.x_to_gamma(entry.mu_gamma_im[1:], entry.tau_fine, entry.tau, entry.epsilon, rbf_type)
    
    # Step 6: plot the fits
    entry.mu_Z_H_re_agm = entry.mu_R_inf + entry.mu_Z_H_re
    entry.band_re_agm = sqrt(entry.cov_R_inf + entry.cov_Z_H_re + entry.sigma_n_im**2)

    entry.mu_Z_H_im_agm = omega_vec*entry.mu_L_0 + entry.mu_Z_H_im
    entry.band_im_agm = sqrt((omega_vec**2)*entry.cov_L_0 + entry.cov_Z_H_im + entry.sigma_n_re**2)

    # Step 7: residuals of the Hilbert DRT
    entry.res_H_re = entry.mu_Z_H_re_agm-entry.b_re
    entry.res_H_im = entry.mu_Z_H_im_agm-entry.b_im
    
    entry.method = 'BHT'    
    
    return entry

# For Peak Analysis

def _matlab_gauss_fct(tau_vec, p_ref, p):
    """Evaluate the Gaussian parameterization used for peak deconvolution."""
    tau_vec = np.asarray(tau_vec, dtype=float).reshape(-1)
    p_ref = np.asarray(p_ref, dtype=float).reshape(-1)
    p = np.asarray(p, dtype=float).reshape(-1)
    gamma_out = np.zeros_like(tau_vec)
    for i in range(0, p.size, 3):
        R0 = p_ref[i] * p[i]
        mu_log_tau = p_ref[i+1] * p[i+1]
        sigma = p_ref[i+2] * p[i+2]
        sigma = max(abs(float(sigma)), np.finfo(float).eps)
        gamma_out += R0 * np.exp(-(np.log(tau_vec)-mu_log_tau)**2/(2*sigma**2))
    return gamma_out


def peak_analysis(entry, rbf_type='Gaussian', data_used='Combined Re-Im Data', induct_used=1,
                  der_used='2nd order', cv_type='custom', reg_param=1E-3,
                  shape_control='FWHM Coefficient', coeff=0.5, peak_method='separate', N_peaks=1):
    """Perform DRT peak deconvolution using the specified number of peaks.

    The number of fitted peaks is defined explicitly by ``N_peaks`` rather than
    being reduced by an automatic peak-detection threshold.
    """
    if entry is None or entry.method in {'credit', 'BHT'}:
        return entry

    if entry.method == 'none':
        entry = simple_run(entry, rbf_type=rbf_type, data_used=data_used, induct_used=induct_used,
                           der_used=der_used, cv_type=cv_type, reg_param=reg_param,
                           shape_control=shape_control, coeff=coeff)

    N_peaks = max(1, int(round(abs(float(N_peaks)))))
    entry.N_peaks = N_peaks

    gamma = np.asarray(entry.gamma, dtype=float).reshape(-1)
    tau_fine = np.asarray(entry.out_tau_vec, dtype=float).reshape(-1)
    if gamma.size != tau_fine.size or gamma.size == 0:
        raise ValueError('DRT result is unavailable for peak analysis.')

    idx0 = int(np.argmax(gamma))
    peak_value_0 = float(gamma[idx0])
    log_tau_mu_0 = float(np.log(tau_fine[idx0]))
    if abs(np.exp(log_tau_mu_0)-1) < np.finfo(float).eps:
        log_tau_mu_0 = float(np.log(tau_fine[idx0] + np.finfo(float).eps))

    sigma_0 = float(np.mean(np.diff(np.log(1.0/np.asarray(entry.freq, dtype=float)))))
    if not np.isfinite(sigma_0) or sigma_0 == 0:
        sigma_0 = 1.0
    sigma_0 = abs(sigma_0)

    p_ref = np.array([peak_value_0, log_tau_mu_0, sigma_0], dtype=float)
    p_init = np.ones_like(p_ref)
    log_tau_min = float(np.min(np.log(tau_fine)))
    log_tau_max = float(np.max(np.log(tau_fine)))
    lb = np.array([0.0, log_tau_min, 0.0], dtype=float)
    ub = np.array([np.inf, log_tau_max, np.inf], dtype=float)
    p_fit = p_init.copy()

    for n in range(1, N_peaks+1):
        def objective(p):
            r = _matlab_gauss_fct(tau_fine, p_ref, p) - gamma
            return float(r @ r)

        bounds = list(zip(lb, ub))
        result = minimize(
            objective, p_init, method='SLSQP', bounds=bounds,
            options={'ftol': 1e-15, 'maxiter': 100000, 'disp': False}
        )
        # MATLAB fmincon normally returns its best iterate even when tolerance
        # termination is imperfect; preserve that behavior.
        p_fit = np.asarray(result.x, dtype=float)

        residual = gamma - _matlab_gauss_fct(tau_fine, p_ref, p_fit)
        idx_temp = int(np.argmax(residual))
        peak_value_temp = float(gamma[idx_temp])
        log_tau_mu_temp = float(np.log(tau_fine[idx_temp]))
        if abs(np.exp(log_tau_mu_temp)-1) < np.finfo(float).eps:
            log_tau_mu_temp = float(np.log(tau_fine[idx_temp] + np.finfo(float).eps))
        sigma_temp = sigma_0

        p_ref_temp = np.array([peak_value_temp, log_tau_mu_temp, sigma_temp], dtype=float)
        lb_temp = np.array([0.0, log_tau_min, 0.0], dtype=float)
        ub_temp = np.array([np.inf, log_tau_max, np.inf], dtype=float)

        if n != N_peaks:
            p_ref = np.concatenate([p_ref*p_fit, p_ref_temp])
            p_init = np.ones_like(p_ref)
            lb = np.concatenate([lb, lb_temp])
            ub = np.concatenate([ub, ub_temp])

    actual = p_ref * p_fit
    entry.p_result = actual.reshape((N_peaks, 3)).T
    entry.gamma_fit_tot = _matlab_gauss_fct(tau_fine, p_ref, p_fit)

    gamma_gauss_mat = np.zeros((tau_fine.size, N_peaks), dtype=float)
    for i in range(N_peaks):
        R0 = entry.p_result[0, i]
        mu_log_tau = entry.p_result[1, i]
        sigma = max(abs(float(entry.p_result[2, i])), np.finfo(float).eps)
        gamma_gauss_mat[:, i] = R0*np.exp(-(np.log(tau_fine)-mu_log_tau)**2/(2*sigma**2))

    entry.gamma_gauss_mat = gamma_gauss_mat
    entry.freq_fine = 1.0/tau_fine
    entry.g_gauss_mat = gamma_gauss_mat * entry.freq_fine[:, None]
    entry.out_gamma_fit = [gamma_gauss_mat[:, i] for i in range(N_peaks)]
    entry.Gaussian = gamma_gauss_mat.T
    entry.num_vectors = N_peaks
    entry.column_headings = [f'Gaussian_{i+1}' for i in range(N_peaks)]
    entry.df = pd.DataFrame(gamma_gauss_mat, columns=entry.column_headings)
    entry.method = 'peak'
    return entry
