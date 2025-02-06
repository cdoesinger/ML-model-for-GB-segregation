import numpy as np

import scipy.special

import sklearn
import sklearn.gaussian_process
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
import sklearn.metrics, sklearn.model_selection


def calculate_rad_dist(XYZ: np.ndarray, n_neighbours: int = 15) -> np.ndarray:
    """
    Calculate the radial distances from the XYZ coordinates of the atoms in the LAE.
    input:
        XYZ: ndarray of shape (n_Data, >3*n_neighbours [X1,Y1,Z1, X2,...]) as is read in from segregation_data.csv
        n_neighbours: number of neighbours to be considered
    returns:
        R_nn: ndarray of shape(m_Data, n_neighbours) contains to distances to the n_neighbours nearest neighbours
    """

    CartCoord = np.zeros((XYZ.shape[0], n_neighbours, 3))
    for i_neigh in range(n_neighbours):
        CartCoord[:, i_neigh, 0] = XYZ[:, 3*i_neigh+0] # x
        CartCoord[:, i_neigh, 1] = XYZ[:, 3*i_neigh+1] # y
        CartCoord[:, i_neigh, 2] = XYZ[:, 3*i_neigh+2] # z
    R_nn = np.sqrt(np.sum(CartCoord**2, axis=2))
    return R_nn

def calculate_Spher_Coord(XYZ: np.ndarray) -> np.ndarray:
    """
    Calculate the spherical coordinated (r, theta, phi) of the atoms in the LAE.
    input:
        XYZ: ndarray of shape (n_Data, >3*n_neighbours [X1,Y1,Z1, X2,...]) as is read in from segregation_data.csv, Cartesian Coordinated of the atoms.
    returns:
        SpherCoord: ndarray of shape(m_Data, n_Atom, 3) contains the spherical coordinated of the n_Atom atoms in the LAE.
            SpherCoord[:,:,0]: R, distance to center (segregation site)
            SpherCoord[:,:,1]: theta, polar angle [, pi] {north-south}
            SpherCoord[:,:,2]: phi, azimuth} [0, 2pi] {east-west}

    """

    CartCoord = np.zeros((XYZ.shape[0], int(XYZ.shape[1]/3), 3))
    for i_neigh in range(int(XYZ.shape[1]/3)):
        CartCoord[:, i_neigh, 0] = XYZ[:, 3*i_neigh+0] # x
        CartCoord[:, i_neigh, 1] = XYZ[:, 3*i_neigh+1] # y
        CartCoord[:, i_neigh, 2] = XYZ[:, 3*i_neigh+2] # z
    SpherCoord = np.zeros(CartCoord.shape)
    
    # radius
    SpherCoord[:, :, 0] = np.sqrt(np.sum(CartCoord**2, axis=2))

    # theta: arctan( Z/(X^2+Y^2)^(0.5) ) [this gives angle in the interval [pi/2 (x=0,y=0,z=1); -pi/2 (x=0,y=0,z=-1)] ]
    SpherCoord[:, :, 1] = np.pi/2 - np.arctan2( CartCoord[:, :, 2], np.sqrt(CartCoord[:, :, 0]**2+CartCoord[:, :, 1]**2) )

    # phi: arctan( y, x )
    SpherCoord[:, :, 2] = np.arctan2( CartCoord[:, :, 1], CartCoord[:, :, 0] )

    return SpherCoord

def calculate_Steinhardt_Par(XYZ: np.ndarray, l: list[int], r_cut: float = np.infty, n_neighbours: int = -1 ) -> np.ndarray:
    """
    Calculate the Steinhardt Parameters from the XYZ coordinates of the atoms in the LAE.
    input:
        XYZ: ndarray of shape (n_Data, >3*n_neighbours [X1,Y1,Z1, X2,...]) as is read in from segregation_data.csv
        l: list of angular momentum numbers for which to calculate the Steinhardt Parameters, can be an integer, list of int or ndarray of int
        r_cut: radial cutoff for the calculation of the Steinhardt Parameters.
        n_neighbours: Ignored of if r_cut<inf and r_cut>0. Number of neighbours to be considered, a value less or equal 0 means all neighbours available are included. 
    returns:
        QL: ndarray of shape(m_Data, len(l)) contains the Steinhardt Parameters for all datapoints.
    """
    if isinstance(l, int) or isinstance(l, float):
        l_arr = np.array([int(l)])
    elif isinstance(l, list):
        l_arr = np.array(l)
    elif isinstance(l, np.ndarray):
        l_arr = l.reshape(-1,)
    else:
        print("l has wrong type!")
        return np.nan
    
    SpherCoord = calculate_Spher_Coord(XYZ)
    if n_neighbours>0:
        SpherCoord[:,n_neighbours:, ]
    QL = np.zeros((SpherCoord.shape[0], l_arr.size), dtype=float)

    for i_l in range(len(l_arr)):
        for m in range(-l_arr[i_l], l_arr[i_l]+1, 1):
            Y_lm = scipy.special.sph_harm(m, l_arr[i_l], SpherCoord[:,:,2], SpherCoord[:,:,1])
            if r_cut>0. and np.isfinite(r_cut) :
                Y_lm[SpherCoord[:,:,0]>r_cut] = 0. + 0.j
                n_neigh = np.sum(SpherCoord[:,:,0]<=r_cut, axis=1)
            if np.isinf(r_cut) or r_cut<0:
                Y_lm[:,n_neighbours:] = 0. + 0.j
                n_neigh = np.ones((Y_lm.shape[0],))*n_neighbours
            QL[:,i_l] += np.abs( np.sum(Y_lm, axis=1)/n_neigh )**2
        QL[:, i_l] *= 4*np.pi/(2*l_arr[i_l]+1)
    QL = np.sqrt(QL)

    return QL

def train_GPR(x_data: np.ndarray, y_data: np.ndarray) -> sklearn.gaussian_process.GaussianProcessRegressor:
    """
    Train the GPR (Matern Kernel) by optimizing the likelihood.
    The parameters alpha (uncertainty of the training data, diagonal of the Kernel-matrix) and the Matern-nu are varied "by hand".
    The optimization of both alpha and nu are restricted to reduce computational costs for the example script.
    """

    gpr_opt_like = -np.inf
    for exp_alpha in np.arange(-6, -4.1, 2):
        for nu in [1.5, 2.5]:
            kernel = ConstantKernel(1, constant_value_bounds=[1E-2, 2]) * \
                     Matern(length_scale=np.ones(x_data.shape[1]), length_scale_bounds=[1E-2, 2*1E+2], nu=nu) + \
                     WhiteKernel(1E-4,(1E-8,1E-1))
            gpr    = sklearn.gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=10**exp_alpha).fit(x_data, y_data)
            if gpr.log_marginal_likelihood_value_ > gpr_opt_like:
                gpr_opt = gpr
                alpha_opt    = exp_alpha
                nu_opt       = nu
                gpr_opt_like = gpr.log_marginal_likelihood_value_
            break
    return gpr_opt
