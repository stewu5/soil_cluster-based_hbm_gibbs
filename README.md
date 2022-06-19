# soil_cluster-based_hbm_gibbs

**This repository includes two jupyter notebooks that contain codes of a 2D and 3D synthetic test for a hierarchical Bayesian model (HBM) based on Gaussian likelihoods. Gibbs sampling is used as the basic sampling scheme for the HBM designed in Ching et al. 2021, which we called HBM-Gibbs. We tested the ability of a cluster-based HBM algorithm combined with HBM-Gibbs to capture the multimodal hyperparameter distribution efficiently. The codes generate results in the manuscript titled "Quasi-site-specific soil property prediction using a cluster-based Hierarchical Bayesian Model", which is submitted to the Structural Safety for review (last update: 2022.06.20).**

## Publication

Corresponding paper (under review):
1. Wu, S., Ching, J., and Phoon, K.-K. (2022). "Quasi-site-specific soil property prediction using a cluster-based Hierarchical Bayesian Model." Journal of Engineering Mechanics, accepted.

Two related papers are:
1. Ching, J., Wu, S., and Phoon, K.-K. (2021). "Constructing quasi-site-specific multivariate probability distribution using hierarchical bayesian model." Journal of Engineering Mechanics, 147(10), 04021069.
2. Wu, S., Angelikopoulos, P., Beck, J. L., and Koumoutsakos, P. (2018). "Hierarchical Stochastic Model in Bayesian Inference for Engineering Applications: Theoretical Implications and Efficient Approximation." ASCE-ASME J Risk and Uncert in Engrg Sys Part B Mech Engrg, 5(1), 011006.

## Dependence

The codes in the two jupyter notebooks were tested under the following environment.

| Package        | Version   |
| -------------- | --------- |
| `Python`       | 3.7.9     |
| `scipy`        | 1.6.0     |
| `pandas`       | 1.2.1     |
| `jupyter`      | 1.0.0     |
| `seaborn`      | 0.11.1    |
| `matplotlib`   | 3.3.2     |
| `numpy`        | 1.20.0    |
| `pickleshare`  | 0.7.5     |
| `scikit-learn` | 0.23.2    |
| `ipython`      | 7.20.0    |

## Copyright and license

Released under the `MIT license`.
