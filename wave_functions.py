"""
This module contains utilities to work with wave data.
    
Author: Kelby Kramer, Massachusetts Institute of Technology and Woods Hole Oceanographic Institution, 2024
"""

# load modules
import numpy as np


###################################################################################################
# WAVE PARAMETER FUNCTIONS
###################################################################################################

def disp_iterative_c(T0,h_target):
    """
    Iteratively solve the dispersion relationship for wave speed, wave number, and wave length.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    T0 : float
        Wave period [s].
    h_target : float
        Water depth [m].
    
    Returns:
    -----------
    c : float
        Wave speed [m/s].
    k : float
        Wave number [1/m].
    L : float
        Wave length [m].
    """
    tol = 1e-9
    g = 9.81
    omega = 2 * np.pi / T0
    # Assume deep water for first guess
    k = (omega**2) / g
    k_old = k * 100
    k_new = k
    count = 0
    while np.any((abs(k_old - k_new) / k_new) > tol):
    # while (abs(k_old - k_new) / k_new) > tol:
        k_old = k
        c = np.sqrt((g / k) * np.tanh(k * h_target))
        k = omega / c
        k_new = k
        count += 1
        if count > 100:
            print('Max Iterations Reached:',count)
            break
    L = (2*np.pi)/k
    return c, k, L


def get_n(L,h):
    """
    Takes the wave velocity and gets the group velocity coefficient.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    L : float
        Wave length [m].
    h : float
        Water depth [m].

    Returns:
    -----------
    n : float
        Group velocity coefficient.
    """
    k = (2 * np.pi) / L
    G = (2 * k * h) / np.sinh(2 * k * h)
    n = (1 + G) / 2
    return n


def get_Cg(c,L,h):
    """
    Calculates the group velocity of a wave.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    c : float
        Wave speed [m/s].
    L : float
        Wave length [m].
    h : float
        Water depth [m].

    Returns:
    -----------
    Cg : float
        Group velocity [m/s].
    """
    n = get_n(L,h)
    Cg = c*n
    return Cg


def get_Ks(Cg0,Cg):
    """
    Calculate the shoaling coefficient Ks.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    Cg0 : float
        Deep water group velocity [m/s].
    Cg : float
        Group velocity [m/s].

    Returns:
    -----------
    Ks : float
        Shoaling coefficient.
    """
    Ks = Cg0 / Cg
    return Ks


def get_Kr(c0,c,theta0):
    """
    Gets the refraction coefficient and wave angle.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    c0 : float
        Deep water wave speed [m/s].
    c : float
        Wave speed [m/s].
    theta0 : float
        Deep water wave angle [radians].

    Returns:
    -----------
    Kr : float
        Refraction coefficient.
    theta : float
        Wave angle [radians].
    """
    theta = np.arcsin((c/c0)*np.sin(theta0))
    Kr = np.cos(theta0)/np.cos(theta)
    return Kr, theta


def get_H_shoal(H0,L0,h0,c0,L,h,c,theta0):
    """
    Gets the shoaling wave height and wave angle.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    H0 : float
        Deep water wave height [m].
    L0 : float
        Deep water wave length [m].
    h0 : float
        Deep water water depth [m].
    c0 : float
        Deep water wave speed [m/s].
    L : float
        Wave length [m].
    h : float
        Water depth [m].
    c : float
        Wave speed [m/s].
    theta0 : float
        Deep water wave angle [radians].

    Returns:
    -----------
    H : float
        Shoaling wave height [m].
    theta : float
        Wave angle [radians].
    """
    Cg0 = get_Cg(c0,L0,h0)
    Cg = get_Cg(c,L,h)
    Ks = get_Ks(Cg0,Cg)
    Kr, theta = get_Kr(c0,c,theta0)
    H = H0 * np.sqrt(Ks*Kr)
    return H, theta

def get_break_h_and_H(Hs, hs, thetas, gamma):
    """
    Get the breaking wave height and depth.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    Hs : float
        Significant wave height [m].
    hs : float
        Water depth [m].
    thetas : float
        Wave angle [radians].
    gamma : float
        Breaking wave height to water depth ratio.

    Returns:
    -----------
    hb : float
        Breaking wave height [m].
    Hb : float
        Breaking wave height [m].
    thetab : float
        Breaking wave angle [radians].
    break_index : int
        Index of breaking wave.
    """
    hs = np.abs(hs)
    if Hs.ndim == 1:  # If input arrays are 1D
        break_index = np.argmax(Hs >= gamma * hs)
        hb = hs[break_index]
        Hb = Hs[break_index]
        thetab = thetas[break_index]
        # end_depth = hs[0]
    elif Hs.ndim == 2:  # For 2D arrays
        break_index = np.argmax(Hs >= gamma * hs, axis=-1)
        # break_index = np.argmin(Hs <= gamma * hs, axis=-1)
        idx = np.indices(break_index.shape)
        hb = hs[idx[0], break_index]
        Hb = Hs[idx[0], break_index]
        thetab = thetas[idx[0], break_index]
        # end_depth = hs[0,0] 
    else:  # For multi-dimensional arrays
        break_index = np.argmax(Hs >= gamma * hs, axis=-1)
        idx = np.indices(break_index.shape) 
        hb = hs[idx[0], idx[1], break_index]
        Hb = Hs[idx[0], idx[1], break_index]
        thetab = thetas[idx[0], idx[1], break_index]
        # end_depth = hs[0,0,0]
    # replace any values where array = end_depth with nan
    # hb = np.where(hb >= end_depth, np.nan, hb)
    # Hb = np.where(hb >= end_depth, np.nan, Hb)
    # thetab = np.where(hb >= end_depth, np.nan, thetab)

    return hb, Hb, thetab, break_index


def get_shoaling_with_breaking(Hs,hs,gamma,break_index):
    """
    Combine shoaling and breaking wave height.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    Hs : float
        Significant wave height [m].
    hs : float
        Water depth [m].
    gamma : float
        Breaking wave height to water depth ratio.
    break_index : int
        Index of breaking wave.

    Returns:
    -----------
    Hs_all : float
        Wave height with breaking and shoaling [m].
    """
    if Hs.ndim == 1:
        Hs_broken = hs*gamma
        Hs_all = np.append(Hs[0:break_index],Hs_broken[break_index:])
    elif Hs.ndim == 2:
        Hs_broken = hs*gamma
        # print('Hs_broken.shape:',Hs_broken.shape)
        Hs_all = np.zeros(Hs.shape)
        for i in range(Hs.shape[1]):
            Hs_all[:,i] = np.append(Hs[0:break_index[i],i],Hs_broken[break_index[i]:,i])
    elif Hs.ndim == 3:
        Hs_broken = hs*gamma
        Hs_all = np.zeros(Hs.shape)
        for i in range(Hs.shape[0]):
            for j in range(Hs.shape[1]):
                Hs_all[i,j,:] = np.append(Hs[i,j,0:break_index[i,j]],Hs_broken[i,j,break_index[i,j]:])
    return Hs_all


def get_E(H):
    """
    Calculate wave energy.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    H : float
        Wave height [m].

    Returns:
    -----------
    E : float
        Wave energy [J/m^2].
    """
    roe = 1025
    g = 9.81
    E = 1/8 * roe * g * H**2
    return E


def get_G(k,h):
    """
    Calculate the radiation stress coefficient G.
        KK MIT/WHOI 2024

    Arguments:
    ----------
    k : float
        Wave number [1/m].
    h : float
        Water depth [m].

    Returns:
    -----------
    G : float
        Radiation stress coefficient.
    """
    G = (2*k*h)/(np.sinh(2*k*h))
    return G
    

def get_Sxx_and_Sxy(H,k,h,theta):
    """
    Calculate the radiation stress components Sxx and Sxy.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    H : float
        Wave height [m].
    k : float
        Wave number [1/m].
    h : float
        Water depth [m].
    theta : float
        Wave angle [radians].

    Returns:
    -----------
    Sxx : float
        Radiation stress component.
    Sxy : float
        Radiation stress component.
    """
    E = get_E(H)
    G = get_G(k,h)
    Sxx = 1/2 * E * ((1+G)*np.cos(theta)**2 + G)
    Sxy = 1/2 * E * (1+G) * np.sin(theta) * np.cos(theta)
    return Sxx, Sxy

def deep_water_period(water_depth):
    """
    Determine what wave period is the maximum for the buoy to be considered in deep water.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    water_depth : float
        Water depth [m].

    Returns:
    -----------
    deep_water_period : float
        Wave period [s].
    """
    deep_water_period = 2*water_depth
    return deep_water_period

def get_theta_relative_to_shore(shore_normal_heading,Dp):
    """
    Get the wave angle relative to the shore.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    shore_normal_heading : float
        Shore normal heading [degrees].
    Dp : np.array
        Wave direction [degrees].

    Returns:
    -----------
    theta0 : np.array
        Wave angle relative to the shore [degrees].
    """
    theta0 = Dp - shore_normal_heading
    # Replace with nan if the magnitude of theta0 is greater than 90 degrees
    theta0 = np.where(np.abs(theta0) > 90, np.nan, theta0)
    return theta0

def create_depth_array(buoy_depth, h_final, dh):
    """
    Create an array of depths to calcualte wave parameters
    KK MIT/WHOI 2024

    Arguments:
    ----------
    buoy_depth : float
        Buoy depth [m].
    h_final : float
        Final depth [m].
    dh : float
        Depth increment [m].
    
    Returns:
    -----------
    hs : np.array
        Depth array [m].
    """
    hs = np.arange(buoy_depth,h_final,-dh)
    return hs

def get_ws(S,d,Cd):
    """
    Get fall velocity of sand particles.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    S : float
        Specific gravity of sand.
    d : float
        Diameter of sand particle [m].
    Cd : float
        Drag coefficient.

    Returns:
    -----------
    ws : float
        Fall velocity of sand particles [m/s].
    """
    g = 9.81
    ws = np.sqrt((4*g*d*(S-1))/(3*Cd)) # Coastal Dynamics 6.4
    return ws

def get_ws_stokes(S,d,nu=1e-6):
    """
    Get the fall velocity of san particles in the stokes range
    KK MIT/WHOI 2024

    Arguments:
    ----------
    S : float
        Specific gravity of sand.
    d : float
        Diameter of sand particle [m].
    nu : float
        Kinematic viscosity of water [m^2/s].

    Returns:
    -----------
    ws : float
        Fall velocity of sand particles [m/s].
    """
    g = 9.81
    ws = ((S-1)*g*(d**2))/(18*nu) # Coastal Dynamics 6.7
    return ws

def get_DFV_Omega(Hb,T,ws):
    """
    Get the dimensionless fall velocity.
    KK MIT/WHOI 2024

    Arguments:
    ----------
    Hbs : float
        Breaking wave height [m].
    T0s : float
        Wave period [s].
    ws : float
        Fall velocity of sand particles [m/s].

    Returns:
    -----------
    Omega : float
        Dimensionless fall velocity.
    """
    Omega = Hb/(ws*T)
    return Omega