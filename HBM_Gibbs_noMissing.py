# Python function for HBM of Gaussian data using Gibbs sampling
# Assume no missing data in input
#
# === Input
# - x_S: [num. of observed data x N] x M (list)
#
# === Output
# - mu_s: M x N x num. of steps
# - C_s: M x N x N x num. of steps
# - mu_0: N x num. of steps
# - C_0: N x N x num. of steps
# - Sig_0: N x N x num. of steps
# - nu_0: num. of steps x 1
# - a_k: N x num. of steps
#
# === Constants
# - N: num. of props
# - M: num. of sites
# - Ns: list of num. of data in each site [len = M]
#
# === Indices
# - t: index for steps
# - m: index for sites
# - x: dummy in list-loop
#

from types import SimpleNamespace
import pandas as pd
import numpy as np
from numpy.random import normal as nrnd
from scipy.stats import invwishart, multivariate_normal, norm, invgamma, wishart
from scipy.special import logsumexp, multigammaln
from scipy.linalg import cholesky as chol
from copy import deepcopy


def hbm_gibbs_nomissing(x_S, Nt=21000):
    # Initialize constants
    N = x_S[0].shape[1]
    M = len(x_S)
    Ns = [x.shape[0] for x in x_S]

    # Initialize hyper-hyperparameters for uniform hyperpriors
    nu_0_grid = np.arange(N, 1001)
    mu_mu0 = np.repeat(0, N)
    C_mu0 = np.diag(np.repeat(1e+4, N))
    alpha_0 = 0.5
    beta_0 = 1e-4
    nu_C0 = N+1
    lambda_Sig0 = N+2
    Psi_Sig0 = np.diag(np.repeat(1e+4, N))

    # Pre-calculate re-usable constants
    inv_Psi_Sig0 = np.linalg.inv(Psi_Sig0)
    inv_C_mu0 = np.linalg.inv(C_mu0)
    inv_Cmu0Mumu0 = np.linalg.solve(C_mu0, mu_mu0)
    lnGamma_nu0_M = M*multigammaln(nu_0_grid/2, N)
    MNln2 = M*N*np.log(2)

    # Initialize outputs
    mu_0 = np.empty((N, Nt))
    C_0 = np.empty((N, N, Nt))
    mu_s = np.empty((M, N, Nt))
    C_s = np.empty((M, N, N, Nt))
    nu_0 = np.empty(Nt)
    Sig_0 = np.empty((N, N, Nt))
    a_k = np.empty((N, Nt))

    # Initialize mu hyperparameters for t=0
    mu_0[:, 0] = multivariate_normal.rvs(mean=mu_mu0, cov=C_mu0)
    a_k[:, 0] = invgamma.rvs(alpha_0, scale=beta_0, size=N)
    Sig_C0 = np.diag(4./a_k[:, 0])
    C_0[:, :, 0] = invwishart.rvs(df=nu_C0, scale=Sig_C0)
    # Initialize C hyperparameters for t=0
    nu_0[0] = np.random.choice(nu_0_grid)
    Sig_0[:, :, 0] = wishart.rvs(lambda_Sig0, scale=Psi_Sig0)
    # Initialize mu & C for t=0
    mu_s[:, :, 0] = multivariate_normal.rvs(mean=mu_0[:, 0], cov=C_0[:, :, 0], size=M)
    C_s[:, :, :, 0] = invwishart.rvs(df=nu_0[0], scale=Sig_0[:, :, 0], size=M)

    # Initialize re-usable constants that will be updated in loop
    inv_C0 = np.linalg.inv(C_0[:, :, 0])
    inv_C0mu0 = np.matmul(inv_C0, mu_0[:, 0])
    inv_Ci = [np.linalg.inv(x) for x in C_s[:, :, :, 0]]

    # Main loop of Gibbs sampling
    for t in range(1, Nt):

        # update mu_s for all sites
        tmp_C = [np.linalg.inv(inv_C0 + Ns[m] * x) for m, x in enumerate(inv_Ci)]
        tmp_mu = np.array(
            [np.matmul(tmp_C[m], inv_C0mu0 + np.matmul(x, np.sum(x_S[m], axis=0))) for m, x in enumerate(inv_Ci)])
        mu_s[:, :, t] = mvn_rvs(tmp_mu, tmp_C)

        # update C_s for all sites
        tmp_mu = [x - np.tile(mu_s[m, :, t], (Ns[m], 1)) for m, x in enumerate(x_S)]
        C_s[:, :, :, t] = np.array(
            [invwishart.rvs(df=nu_0[t - 1] + Ns[m], scale=Sig_0[:, :, t - 1] + np.matmul(x.T, x)) for m, x in
             enumerate(tmp_mu)])
        inv_Ci = [np.linalg.inv(x) for x in C_s[:, :, :, t]]

        # update mu_0
        tmp_C = np.linalg.inv(inv_C_mu0 + M * inv_C0)
        tmp_mu = np.matmul(tmp_C, inv_Cmu0Mumu0 + np.matmul(inv_C0, np.sum(mu_s[:, :, t], axis=0)))
        mu_0[:, t] = multivariate_normal.rvs(mean=tmp_mu, cov=tmp_C)

        # update C_0
        tmp_mu = mu_s[:, :, t] - np.tile(mu_0[:, t], (M, 1))
        C_0[:, :, t] = invwishart.rvs(df=nu_C0 + M, scale=Sig_C0 + np.matmul(tmp_mu.T, tmp_mu))
        inv_C0 = np.linalg.inv(C_0[:, :, t])
        inv_C0mu0 = np.matmul(inv_C0, mu_0[:, t])

        # update Sig_0
        tmp_mu = np.linalg.inv(inv_Psi_Sig0 + np.array(inv_Ci).sum(axis=0))
        Sig_0[:, :, t] = wishart.rvs(M * nu_0[t - 1] + lambda_Sig0, scale=tmp_mu)

        # update nu_0
        tmp_det = [np.linalg.slogdet(x) for x in C_s[:, :, :, t]]
        tmp_det_Sig0 = np.linalg.slogdet(Sig_0[:, :, t])
        if np.prod([x[0] for x in tmp_det]) * tmp_det_Sig0[0] < 0:
            print(f't = {t}: Error on determinant calc.!')
        tmp_logw = nu_0_grid / 2 * (M * tmp_det_Sig0[1] - sum([x[1] for x in tmp_det]) - MNln2) - lnGamma_nu0_M
        tmp_w = np.exp(tmp_logw - logsumexp(tmp_logw))
        nu_0[t] = np.random.choice(nu_0_grid, p=tmp_w)

        # update a_k, Sig_C0
        a_k[:, t] = invgamma.rvs(alpha_0 + (N + 1) / 2, scale=beta_0 + 2 / np.diag(C_0[:, :, t]))
        Sig_C0 = np.diag(4. / a_k[:, t])

    return mu_s, C_s, mu_0, C_0, Sig_0, nu_0, a_k


def mvn_rvs(mean, cov):
    """
    Function to sample from multiple Gaussian distributions
    Parameters
    ----------
    mean : numpy.array
        N_samples x N_dim
    cov : numpy.array or list
        (N_samples x N_dim x N_dim) or list[numpy.array(N_dim x N_dim)]
    return : numpy.array
        N_samples x N_dim
    """
    Z = nrnd(size=(mean.shape[0], mean.shape[1]))
    L = [chol(x) for x in cov]

    return np.array([Z[i].dot(L[i]) for i, z in enumerate(Z)]) + mean


class HBM_prd():

    def __init__(self, HBM_samples):
        """
        SMC iqspr runner (assume data type of samples = list or np.array).
        Parameters
        ----------
        HBM_samples : dict
            Dictionary of HBM samples
        """
        self._s_hbm = SimpleNamespace(**HBM_samples)
        self._gen_mu = None
        self._gen_cov = None

    def gen_gauss_samples(self, N_burn=0, del_t=1, N_i=1, rand_seed=None):
        """
        N_burn : int
            Num. of burn-in samples considered (>= 0)
        del_t : int
            Pick one sample for every del_t samples (> 0)
        N_i : int
            Num. of samples per hyperparameter sample (> 0)
        rand_seed : int
            random seed
        """

        s = self._s_hbm
        Nt = s.mu_0.shape[1]
        N_psi = len(list(range(N_burn, Nt, del_t)))
        Ndim = s.mu_0.shape[0]

        if rand_seed is not None:
            np.random.seed(rand_seed)
        rand_seeds = np.random.randint(0, 2 ** 32 - 1, size=N_psi)

        col_mu = [f'mu_{x}' for x in range(Ndim)]
        col_sig = [f'sig_{x}' for x in range(Ndim)]
        col_rho = [f'rho_{y}{x}' for y in range(1, Ndim) for x in range(y)]

        if N_i == 1:
            data_Mu = np.zeros([Ndim, N_psi])
            data_Sigma = np.zeros([Ndim, Ndim, N_psi])

            for ii, iS in enumerate(range(N_burn, Nt, del_t)):
                data_Mu[:, ii] = multivariate_normal.rvs(mean=s.mu_0[:, iS], cov=s.C_0[:, :, iS],
                                                         random_state=rand_seeds[ii]).T
                data_Sigma[:, :, ii] = invwishart.rvs(df=s.nu_0[iS], scale=s.Sig_0[:, :, iS],
                                                      random_state=rand_seeds[ii]).T
        else:
            data_Mu = np.zeros([Ndim, N_i, N_psi])
            data_Sigma = np.zeros([Ndim, Ndim, N_i, N_psi])

            for ii, iS in enumerate(range(N_burn, Nt, del_t)):
                data_Mu[:, :, ii] = multivariate_normal.rvs(mean=s.mu_0[:, iS], cov=s.C_0[:, :, iS], size=N_i,
                                                            random_state=rand_seeds[ii]).T
                data_Sigma[:, :, :, ii] = invwishart.rvs(df=s.nu_0[iS], scale=s.Sig_0[:, :, iS], size=N_i,
                                                         random_state=rand_seeds[ii]).T

        mu = data_Mu[:Ndim, ].reshape(Ndim, -1)
        cov = data_Sigma[:Ndim, :Ndim, ].reshape(Ndim, Ndim, -1)
        sig = np.sqrt(np.diagonal(cov).T)
        rho = np.vstack([cov[y, x, :] / sig[y, :] / sig[x, :] for y in range(1, Ndim) for x in range(y)])

        self._gen_mu = mu
        self._gen_cov = cov

        return pd.DataFrame(np.concatenate([mu, sig, rho]).T, columns=col_mu + col_sig + col_rho)

    def gen_x_samples(self, x_S, N_burn=10, del_t=1, Nt=11, rand_seed=None, N_burn_h=0, del_t_h=1):
        """
        x_S : list of numpy.array
            List of data (Ns x M)
        N_burn : int
            Num. of burn-in samples considered (>= 0)
        del_t : int
            Pick one sample for every del_t samples (> 0)
        Nt : int
            Total num. of samples to be generated (> 0)
        rand_seed : int
            random seed
        N_burn_h : int
            Num. of burn-in samples used for hyperparameters (>=0)
        del_t_h : int
            Pick one sample for every del_t samples used for hyperparameters (> 0)
        """

        s = self._s_hbm
        M = len(x_S)
        N = x_S[0].shape[1]
        Ns = [x.shape[0] for x in x_S]

        Nt_h = s.mu_0.shape[1]
        N_psi = len(list(range(N_burn_h, Nt_h, del_t_h)))

        mu_s = [[] for _ in range(N_psi)]
        C_s = [[] for _ in range(N_psi)]

        if rand_seed is not None:
            np.random.seed(rand_seed)
            
        for ii, iS in enumerate(range(N_burn_h, Nt_h, del_t_h)):
            mu_s_ = np.empty((M, N, Nt))
            C_s_ = np.empty((M, N, N, Nt))

            # initialize mu_s and C_s
            mu_s_[:, :, 0] = multivariate_normal.rvs(mean=s.mu_0[:, iS], cov=s.C_0[:, :, iS], size=M)
            C_s_[:, :, :, 0] = invwishart.rvs(df=s.nu_0[iS], scale=s.Sig_0[:, :, iS], size=M)

            # Initialize re-usable constants that will be updated in loop
            inv_C0 = np.linalg.inv(s.C_0[:, :, iS])
            inv_C0mu0 = np.matmul(inv_C0, s.mu_0[:, iS])
            inv_Ci = [np.linalg.inv(x) for x in C_s_[:, :, :, 0]]

            for t in range(1, Nt):
                # update mu_s for all sites
                tmp_C = [np.linalg.inv(inv_C0 + Ns[m] * x) for m, x in enumerate(inv_Ci)]
                tmp_mu = np.array(
                    [np.matmul(tmp_C[m], inv_C0mu0 + np.matmul(x, np.sum(x_S[m], axis=0))) for m, x in
                     enumerate(inv_Ci)])
                mu_s_[:, :, t] = mvn_rvs(tmp_mu, tmp_C)

                # update C_s for all sites
                tmp_mu = [x - np.tile(mu_s_[m, :, t], (Ns[m], 1)) for m, x in enumerate(x_S)]
                C_s_[:, :, :, t] = np.array(
                    [invwishart.rvs(df=s.nu_0[iS] + Ns[m], scale=s.Sig_0[:, :, iS] + np.matmul(x.T, x)) for m, x in
                     enumerate(tmp_mu)])
                inv_Ci = [np.linalg.inv(x) for x in C_s_[:, :, :, t]]

            mu_s[ii] = deepcopy(mu_s_[:, :, range(N_burn, Nt, del_t)])
            C_s[ii] = deepcopy(C_s_[:, :, :, range(N_burn, Nt, del_t)])

        mu_s = np.concatenate(mu_s, axis=2)
        C_s = np.concatenate(C_s, axis=3)

        return [mvn_rvs(mu_s[i, :, :].T, C_s[i, :, :, :].T) for i in range(mu_s.shape[0])]

    def gen_missing_x(self, x_S, N_burn=10, del_t=1, Nt=11, rand_seed=None, N_burn_h=0, del_t_h=1):
        """
        x_S : list of numpy.array
            List of data (Ns x M)
        N_burn : int
            Num. of burn-in samples considered (>= 0)
        del_t : int
            Pick one sample for every del_t samples (> 0)
        Nt : int
            Total num. of samples to be generated (> 0)
        rand_seed : int
            random seed
        N_burn_h : int
            Num. of burn-in samples used for hyperparameters (>=0)
        del_t_h : int
            Pick one sample for every del_t samples used for hyperparameters (> 0)
        """

        s = self._s_hbm
        M = len(x_S)
        N = x_S[0].shape[1]
        Ns = [x.shape[0] for x in x_S]

        Nt_h = s.mu_0.shape[1]
        # N_psi = len(list(range(N_burn_h, Nt_h, del_t_h)))

        x_S_out = []

        if rand_seed is not None:
            np.random.seed(rand_seed)

        for ii, iS in enumerate(range(N_burn_h, Nt_h, del_t_h)):
            mu_s_ = np.empty((M, N, Nt))
            C_s_ = np.empty((M, N, N, Nt))

            # initialize mu_s_, C_s_, and x_S_
            mu_s_[:, :, 0] = multivariate_normal.rvs(mean=s.mu_0[:, iS], cov=s.C_0[:, :, iS], size=M)
            C_s_[:, :, :, 0] = invwishart.rvs(df=s.nu_0[iS], scale=s.Sig_0[:, :, iS], size=M)
            x_S_ = [mvn_rvs_filling(x_S, mu_s_[:, :, 0], C_s_[:, :, :, 0])]

            # Initialize re-usable constants that will be updated in loop
            inv_C0 = np.linalg.inv(s.C_0[:, :, iS])
            inv_C0mu0 = np.matmul(inv_C0, s.mu_0[:, iS])
            inv_Ci = [np.linalg.inv(x) for x in C_s_[:, :, :, 0]]

            for t in range(1, Nt):
                # update mu_s for all sites
                tmp_C = [np.linalg.inv(inv_C0 + ns * x) for ns, x in zip(Ns, inv_Ci)]
                tmp_mu = np.array(
                    [np.matmul(c_m, inv_C0mu0 + np.matmul(c_inv, np.sum(x_m, axis=0))) for c_m, x_m, c_inv in
                     zip(tmp_C, x_S_[-1], inv_Ci)])
                mu_s_[:, :, t] = mvn_rvs(tmp_mu, tmp_C)

                # update C_s for all sites
                tmp_mu = [x_m - np.tile(m_m[:, t], (ns, 1)) for ns, m_m, x_m in zip(Ns, mu_s_, x_S_[-1])]
                C_s_[:, :, :, t] = np.array(
                    [invwishart.rvs(df=s.nu_0[iS] + ns, scale=s.Sig_0[:, :, iS] + np.matmul(x.T, x)) for ns, x in
                     zip(Ns, tmp_mu)])
                inv_Ci = [np.linalg.inv(x) for x in C_s_[:, :, :, t]]

                # update x_S_ for all sites
                x_S_.append(mvn_rvs_filling(x_S, mu_s_[:, :, t], C_s_[:, :, :, t]))

            x_S_out += [deepcopy(x_S_[x]) for x in range(N_burn, Nt, del_t)]

        return x_S_out


def mvn_rvs_filling(x_S, mean, cov):
    """
    Function to sample from multiple Gaussian distributions for missing data
    (assume N_data = num. of data in each site is the same for all sites)

    Parameters
    ----------
    x_S : list
        List (len=N_samples) of site data with same missing spots denoted as np.nan in numpy.array(N_data x N_dim)
    mean : numpy.array
        N_samples x N_dim
    cov : numpy.array or list
        (N_samples x N_dim x N_dim) or list[numpy.array(N_dim x N_dim)]
    return : list
        List (len=N_samples) of numpy.array(N_data x N_dim)
    """

    # extract different missing data patterns (exclude no missing case)
    # assume missing patterns in all data are the same, so take the first one for reference
    x_S_ = deepcopy(x_S)
    tmp_df = pd.DataFrame(x_S_[0]).isna()
    tmp_group = tmp_df.groupby(by=tmp_df.columns.to_list()).groups
    try:
        del tmp_group[tuple(False for _ in range(tmp_df.shape[1]))]
    except:
        pass

    # sample missing x samples for each missing pattern and assemble data
    for key, val in tmp_group.items():
        n = len(val)

        if all(key):  # all missing is equivalent to normal sampling
            tmp_mu = [np.tile(m_u, (n, 1)) for m_u in mean]
            tmp_C = cov
        else:
            X_o = [x[np.ix_(val, np.logical_not(key))] for x in x_S_]
            Mu_o = mean[:, np.logical_not(key)]
            Mu_u = mean[:, key]
            Sig_o = cov[np.ix_(np.arange(len(cov)), np.logical_not(key), np.logical_not(key))]
            Sig_u = cov[np.ix_(np.arange(len(cov)), key, key)]
            Sig_uo = cov[np.ix_(np.arange(len(cov)), key, np.logical_not(key))]

            tmp_inv = [np.matmul(c_uo, np.linalg.inv(c_o)) for c_uo, c_o in zip(Sig_uo, Sig_o)]
            tmp_C = [c_u - np.matmul(c_inv, c_uo.T) for c_u, c_inv, c_uo in zip(Sig_u, tmp_inv, Sig_uo)]
            tmp_mu = [np.tile(m_u, (n, 1)) + np.matmul(c_inv, (x_o - np.tile(m_o, (n, 1))).T).T for m_u, c_inv, x_o, m_o
                      in zip(Mu_u, tmp_inv, X_o, Mu_o)]

        Z = nrnd(size=(len(tmp_mu), n, tmp_mu[0].shape[1]))
        L = [chol(x) for x in tmp_C]
        for x, l, z, m_u in zip(x_S_, L, Z, tmp_mu):
            x[np.ix_(val, key)] = z.dot(l) + m_u

    return x_S_

