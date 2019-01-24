
# coding: utf-8

# # LSST Kilonova Simulations

# In[3]:


import sys
sys.path.append('/Users/kristophermortensen/NUREU17/LSST/TargetOfOpportunityStrategy')
from Kilonova_Class import kilonova
import numpy as np
import astropy as astro
import random
import pandas
import sncosmo
import os
import scipy
import astropy.units as u
import statsmodels.api as sm
from astropy.cosmology import WMAP9 as cosmo
from astropy.cosmology import z_at_value
from astropy.table import Table, Column, vstack
from astropy.io import ascii
from astropy.time import Time
from scipy.constants import h,k,c,Stefan_Boltzmann
from scipy.interpolate import interp1d, RegularGridInterpolator
from scipy.integrate import quad
from operator import itemgetter, attrgetter
import warnings
warnings.filterwarnings('ignore')


# In[4]:


# GW Sources
Villar = ascii.read("Villar Data.csv")

# - trigger
trigger = Time('2017-08-17 12:41:04')
trigger = trigger.mjd


# In[5]:


#
# Bandpasses
#

def bandpasses(folder):
    bandpasses = list_bandpass(folder)
    bandpasses1 = map_angstrom(bandpasses)
    bandpasses2 = map_transmission(bandpasses1)
    bandpasses3 = map_order(bandpasses2)
    bandpasses_new = clean_bandpasses(bandpasses3)
    return bandpasses_new

def list_bandpass(folder):
    band_passes = {}
    
    if folder == 'LSST Filters':
        x = 0
    else:
        x = 1
    for file in os.listdir(folder)[x:]:
        key = str(file)
        value = ascii.read(folder+"/"+file)
        band_passes[key[:-4]] = value
    
    return band_passes


def map_angstrom(bandpasses):
    bandpasses_new = {}
    for key, table in bandpasses.items():
        
        if table.colnames[0] == 'wavelength(A)':
            table['wavelength(A)'] = table['wavelength(A)']
            table.rename_column('wavelength(A)', 'wavelength')
            bandpasses_new[key] = table
            
        elif table.colnames[0] == 'wavelength(nm)':
            table['wavelength(nm)'] = 10*table['wavelength(nm)']
            table.rename_column('wavelength(nm)', 'wavelength')
            bandpasses_new[key] = table
            
        elif table.colnames[0] == 'wavelength(mu)':
            table['wavelength(mu)'] = 10000*table['wavelength(mu)']
            table.rename_column('wavelength(mu)', 'wavelength')
            bandpasses_new[key] = table
            
        else:
            table.rename_column(table.colnames[0], 'wavelength')
            bandpasses_new[key] = table
    
    return bandpasses_new


def map_transmission(bandpasses):
    bandpasses_new = {}
    for key, table in bandpasses.items():
        
        if table.colnames[1] == 'transmission':
            bandpasses_new[key] = table
        
        elif table.colnames[1] == 'transmission(%)':
            table['transmission(%)'] = 0.01*table['transmission(%)']
            table.rename_column('transmission(%)', 'transmission')
            bandpasses_new[key] = table
    
    return bandpasses_new


def map_order(bandpasses):
    bandpasses_new = {}
    for key, table in bandpasses.items():
        x = []
        y_wave = []
        y_trans = []
        for i in range(len(table)):
            x.append([table['wavelength'][i], table['transmission'][i]])
            
        y = sorted(x, key=itemgetter(0))
        
        for i in range(len(y)):
            y_wave.append(y[i][0])
            y_trans.append(y[i][1])
            
        table_new = Table((y_wave,y_trans), names=('wavelength', 'transmission'))
        bandpasses_new[key] = table_new
    return bandpasses_new

def clean_bandpasses(bandpasses):
    bandpasses_new = {}
    for key, table in bandpasses.items():
        bandpasses_new[key] = clean_up(table)
    return bandpasses_new
        

def clean_up(table):
    df = table.to_pandas()
    df = df.groupby('wavelength', as_index=False)['transmission'].mean()
    new_table = Table.from_pandas(df)
    return new_table

def interp_bandpasses(bandpasses):
    approx = 'cubic'
    interpolations = {}
    
    for key,table in bandpasses.items():
        interp = interp1d(table['wavelength'], table['transmission'], bounds_error=False, fill_value=0)
        interpolations[key] = interp
    return interpolations




#
# Lambda Effective (Average)
#

def lambda_effectives(bandpasses):
    interps = interp_bandpasses(bandpasses)
    bandpasses_new = {}
    for key, table in bandpasses.items():
        global interp
        interp = interps[key]
        bandpasses_new[key] = calc_lambda(table, interp)
    if 'MASTER R' and 'MASTER B' in list(bandpasses_new.keys()):
        bandpasses_new['MASTER W'] = 0.8*bandpasses_new['MASTER R']+0.2*bandpasses_new['MASTER B'] 
    return bandpasses_new


def calc_lambda(table, interp):
    lambda_eff = np.divide(quad(f, min(wavelength_new(table)), max(wavelength_new(table)))[0],
                           quad(interp, min(wavelength_new(table)), max(wavelength_new(table)))[0])
    return lambda_eff

def f(x):
    return x*interp(x)


def wavelength_new(table):
    set_length=10000
    day_new = np.linspace(min(table['wavelength']),
                          max(table['wavelength']),set_length)
    return day_new


# In[6]:


#
# LSST Simulations
#

bands = ['u','g','r','i','z','y']
LSST_bandpasses = bandpasses('LSST Filters')
LSST_transmissions = interp_bandpasses(LSST_bandpasses)
LSST_lambdas = lambda_effectives(LSST_bandpasses)



def LSST_magdict(sed_list, times, gwsource, simulated):
    mag_list = {band: [LSST_mags(sed_data, time, band, gwsource, simulated) for sed_data, time in zip(sed_list, times)] for band in bands}
    return Table([times,mag_list['u'],mag_list['g'],mag_list['r'],mag_list['i'],mag_list['z'],mag_list['y']],
                 names=('day','u','g','r','i','z','y'))
    
def LSST_mags(sed_data, time, band, gwsource, simulated):
    mag = 0
    sed_minlam = min(sed_data['lambda'])
    sed_maxlam = max(sed_data['lambda'])
    global sed_interp
    sed_interp = flux_interp(sed_data, band, simulated)
    lam_eff = LSST_lambdas['LSST '+band]
    LSST_lams = LSST_bandpasses['LSST '+band]
    global LSST_trans
    LSST_trans = LSST_transmissions['LSST '+band]
    LSST_minlam = min(LSST_lams['wavelength'])
    LSST_maxlam = max(LSST_lams['wavelength'])
    min_time = max(sed_minlam, LSST_minlam)
    max_time = min(sed_maxlam, LSST_maxlam)
    X = quad(flux_integral, min_time, max_time)[0]
    Y = quad(LSST_trans, min_time, max_time)[0]
    
    if simulated == True:
        flux = X/Y
        mag = convert_to_mag(flux, lam_eff)
    else:
        upperlimits = upperlimit(gwsource[np.where(gwsource['band']==band)], 'magnitude')
        times = reset_days(upperlimits['reg days'], trigger)
        if time < max(times):
            flux = X/Y
            mag = convert_to_mag(flux, lam_eff)
        else:
            mag = np.inf
    return mag

def flux_interp(sed_data, band, simulated):
    if simulated == True:
        approx = 'cubic'
    else:
        if band in ['u','g','i','r','y']:
            approx = 'slinear'
        else:
            approx = 'cubic'
    
    interp = interp1d(sed_data['lambda'], sed_data['flux']*1e-7, bounds_error=False, fill_value='extrapolate', kind=approx)
    return interp

def flux_integral(lam):
    return LSST_trans(lam)*sed_interp(lam)

def eflux_integral(lam):
    return LSST_trans(lam)*sed_einterp(lam)

def convert_to_mag(flux, lam_eff):
    c = astro.constants.c.to('Angstrom/s').value
    f_nu = (lam_eff**2/c)*flux
    mag = -2.5*np.log10(f_nu)-48.6
    return mag


# In[7]:


#
# Villar Light Curve Models
#


# Constants

#Note: These dictionaries have lists that contain [mean, upperbound, lowerbound].
#Temperature (Temp) in Kelvin. Mass of ejecta (m_ej) in solar masses. Velocity of ejecta (v_ej) in fractions of c.
Temp = {'blue': [674.058, 416.996, 486.067], 'purple': [1307.972, 34.040, 42.067], 'red': [3745.062, 75.034, 75.337]}
m_ej = {'blue': [0.020, 0.001, 0.001], 'purple': [0.047, 0.001, 0.002], 'red': [0.011, 0.002, 0.001]}
v_ej = {'blue': [0.266, 0.008, 0.008], 'purple': [0.152, 0.005, 0.005], 'red': [0.137, 0.025, 0.021]}
Opacity = {'blue': 0.5, 'purple': 3, 'red':10}
Blue = np.asarray([m_ej['blue'][0], v_ej['blue'][0], Temp['blue'][0], Opacity['blue']])
Purple = np.asarray([m_ej['purple'][0], v_ej['purple'][0], Temp['purple'][0], Opacity['purple']])
Red = np.asarray([m_ej['red'][0], v_ej['red'][0], Temp['red'][0], Opacity['red']])
ThreeComp = np.asarray([Blue, Purple, Red]).flatten()
beta = 13.4
t_0 = 1.3
sigma = 0.11
M_sol_g = astro.units.solMass.to('g')
e_th_table = ascii.read('e_th_table.csv')
sig_sb = astro.constants.sigma_sb.cgs.value
c_cm = astro.constants.c.cgs.value


# In[8]:


#
# Villar SED Creation
#

def BB_seds(times, param_list, dist = 40.7):
    '''
    Inputs:
    -> times - list of times measured in days
    
    -> param_list - a list containing lists of the following parameters: 
    
                    [EJECTA_MASS (M_sol), EJECTA_VELOCITY (c), CRITICAL_TEMPERATURE (K), OPACITY (cm/g)] (1)
                    Note: For multiple components, repeat (1) for each component and combine to a 1-D array
                    
    -> dist - gives the distance of the kilonova in Mpc

    Outputs:
    -> sed_table - sed tables which are used in to plot the magnitudes
    '''
    N = len(param_list)
    m_ejectas = [param_list[i] for i in range(0,N,4)]
    v_ejectas = [param_list[i] for i in range(1,N,4)]
    temps = [param_list[i] for i in range(2,N,4)]
    opacs = [param_list[i] for i in range(3,N,4)]
    tr_list = np.dstack([SED_sims(times, m_ej, v_ej, temp, opac) for m_ej, v_ej, temp, opac in zip(m_ejectas, v_ejectas, temps, opacs)])
    temp_list = tr_list[0]
    radii_list = tr_list[1]
    BBcolor = [BB_combinterp(temps, radii, dist) for temps, radii in zip(temp_list, radii_list)]
    sed_table = [Table([BBcolor[i][0], BBcolor[i][1], np.zeros(len(BBcolor[i][0]))],
                       names=('lambda', 'flux', 'e_flux')) for i in range(len(BBcolor))]
    return sed_table


def SED_sims(times, M_ej, V_ej, T_c, opac):
    '''
    Inputs:
    -> times - times; floats measured in seconds
    -> M_ej - ejecta mass; float measured in solar masses
    -> V_ej - ejecta velocity; float measured in fraction of c
    -> T_c - critical temperature for color component; float measured in K
    -> opac - color's opacity; float measured in cm^2/g
    
    
    Outputs:
    -> temperature - calculated temperature at time t; float measure in K
    -> radius - calculated radius at time t; float measured in cm
    '''
    t = times*86400
    
    Lbols = Lbol(t, M_ej, V_ej, opac)
    temps = T_photo(t, Lbols, V_ej, T_c)
    Temperatures = np.fromiter([max(temp, T_c) for temp in temps], float)
    Radii = R_photo(t, Lbols, V_ej, temps, T_c)
    return Temperatures, Radii



def BB_combinterp(temps, radii, dist = 40.7):
    steps = 1000
    min_lambda, max_lambda = LSST_range(LSST_bandpasses)
    lambdas = np.linspace(min_lambda, max_lambda, steps)
    fluxes = np.asarray([blackbody_function(lambdas, temp, radius, dist) for temp, radius in zip(temps, radii)])
    flux = fluxes.sum(axis=0)
    curve = np.array([lambdas, flux])
    return curve


def blackbody_function(lam, T, R, dist = 40.7):
    """ Blackbody as a function of wavelength (angstrom) and temperature (K).

    returns units of erg/s/cm^2/cm/Steradian
    """
    conversion = 3.085677581e+24 #Mpc to cm
    d = dist*conversion
    z = z_at_value(cosmo.luminosity_distance, dist*u.Mpc)
    
    lam = 1e-10*(lam*(1+z)) # angstroms to meters (adjusted redshift)
    flux = ((2*h*c**2) / (lam**5))*(1/(np.exp((h*c)/(lam*k*T)) - 1))*np.pi*(R/d)**2
    return flux


def LSST_range(bandpasses):
    '''
    Inputs:
    -> bandpasses - dictionary containing the transmission curves of the six LSST filters
    
    Outputs:
    -> min_lambda, max_lambda - floats measured in Angstroms; the two lambdas give the range of wavelngths for BB_interp
    '''
    lam_list = []
    for key,table in bandpasses.items():
        lam_list.append(min(table['wavelength']))
        lam_list.append(max(table['wavelength']))
    return min(lam_list), max(lam_list)


def R_photo(times, lbols, v_ej, temps, T_c):
    '''
    Inputs:
    -> t - times; floats measured in seconds
    -> lbols - bolometric luminosities
    -> temp - calculated temperature at time t; float measured in K
    -> T_c - critical temperature for color component; float measured in K
    
    Outputs:
    -> radius - calculated radius at time t; float measured in cm
    '''
    radii = np.fromiter([(v_ej*c_cm)*t if temp > T_c else                          0 if np.isnan(lbol) else ((lbol)/(4*np.pi*sig_sb*T_c**4))**0.5 for t,lbol,temp in zip(times, lbols, temps)],
                        float)
    return radii

def T_photo(times, lbols, v_ej, T_c):
    '''
    Inputs:
    -> t - time; float measured in seconds
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    -> opacity - color's opacity; float measured in cm^2/g
    -> T_c - critical temperature for color component; float measured in K
    
    Outputs:
    -> temp - calculated temperature at time t; float measured in K
    '''
    temps = np.fromiter([T_c if np.isnan(lbol) else ((lbol)/(4*np.pi*sig_sb*((v_ej*c_cm)*t)**2))**0.25 for t,lbol in zip(times, lbols)],
                        float)
    
    return temps

def Lbol(t, m_ej, v_ej, opacity):
    '''
    Inputs:
    -> t - time; float measured in seconds
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    -> opacity - color's opacity; float measured in cm^2/g
    
    Outputs:
    -> lbol - calculated bolometric luminosity at time t; float measured in erg/s
    '''
    T_d = t_d(m_ej, v_ej, opacity)
    lbol = (2*np.exp(-1.*((t/T_d)**2))/T_d)*Lbol_integral(t, m_ej, v_ej, T_d)
    return lbol

def t_d(m_ej, v_ej, opacity):
    '''
    Inputs:
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    -> opacity - color's opacity; float measured in cm^2/g
    
    Outputs:
    -> t_d - calculated value of t_d (see Villar paper); float measured in seconds
    '''
    return np.sqrt((2*(opacity)*(m_ej*M_sol_g))/(beta*(v_ej*c_cm)*c_cm))

def Lbol_integral(times, m_ej, v_ej, t_d):
    '''
    Inputs:
    -> t - time; float measured in seconds
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    -> t_d - calculated value of t_d (see Villar paper); float measured in seconds
    
    Outputs:
    -> lbol_int - the integral part of Lbol() (see Villar paper); no useful output on its own, but measured in erg/s
    '''
    
    return np.fromiter([quad(L_integrand, 0, t, args=(m_ej, v_ej, t_d))[0] for t in times], float)

def L_integrand(t, m_ej, v_ej, t_d):
    '''
    Inputs:
    -> t - time; float measured in seconds
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    -> t_d - calculated value of t_d (see Villar paper); float measured in seconds
    
    Outputs:
    -> l_integrand - the integrand of Lbol_integral() (see Villar paper); no useful output on its own, but measured in erg/s.
    '''
    return L_in(t, m_ej)*e_th(t, m_ej, v_ej)*np.exp((t/t_d)**2)*(t/t_d)

def L_in(t, m_ej):
    '''
    Inputs:
    -> t - time; float measured in seconds
    -> m_ej - ejecta mass; float measured in solar masses
    
    Outputs:
    -> l_in - function used in Lbol() (see Villar paper); units of erg/s
    '''
    return 4e18*(m_ej*M_sol_g)*(0.5-(1/np.pi)*np.arctan((t-t_0)/sigma))**1.3

def e_th(time, m_ej, v_ej):
    '''
    Inputs:
    -> t - time; float measured in seconds
        Note: the time for e_th(t must be measured in days)
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    
    Outputs:
    -> e_th - function used in Lbol() (see Villar paper); it is a float X such that 0 < X < 1.
    '''
    t = time/86400
    a,b,d = calc_coeffs(m_ej, v_ej)
    return 0.36*(np.exp(-a*t)+(np.log(1+2*b*t**d)/(2*b*t**d)))

def calc_coeffs(m_ej, v_ej):
    '''
    Inputs:
    -> m_ej - ejecta mass; float measured in solar masses
    -> v_ej - ejecta velocity; float measured in fraction of c
    
    Outputs:
    -> coeffs_list - array containing lists of the coefficiencts a, b, and d. 
                     Note: Each coefficient has three different values [i_rand, i_rad, i_tor] for i in {a,b,d}; rand, rad, and tor are
                            different measurements for a, b, and d depending on the data points used (see Barnes et al. 2016).
    '''
    
    a_interps = e_th_interps('a')
    b_interps = e_th_interps('b')
    d_interps = e_th_interps('d')
    
    a_list = a_interps([m_ej, v_ej])[0]
    b_list = b_interps([m_ej, v_ej])[0]
    d_list = d_interps([m_ej, v_ej])[0]
        
    return [a_list, b_list, d_list]

def e_th_interps(coeff):
    '''
    Input:
    -> coeff - string representing the coefficient a, b, and d
    
    Output:
    e_th_interps - a 2D interpolation for a coefficient; interpolations are dependent on ejecta mass and velocity
                   Note: there are three interpolations for one component: random (rand), radial (rad), and toroidal (tor);
                           see Barnes et al. 2016 for more explanation.
    '''
    function = 'linear'
    v_ej = np.asarray([0.1, 0.2, 0.3])
    m_ej = np.asarray([1.e-3, 5.e-3, 1.e-2, 5.e-2])
    
    if coeff == 'a':
        coeffs = np.asarray([[2.01, 4.52, 8.16], [0.81, 1.9, 3.2], [
                              0.56, 1.31, 2.19], [.27, .55, .95]])
    elif coeff == 'b':
        coeffs = np.asarray([[0.28, 0.62, 1.19], [0.19, 0.28, 0.45], [
                              0.17, 0.21, 0.31], [0.10, 0.13, 0.15]])
    else:
        coeffs = np.asarray([[1.12, 1.39, 1.52], [0.86, 1.21, 1.39], [
                              0.74, 1.13, 1.32], [0.6, 0.9, 1.13]])
    
    interp = RegularGridInterpolator((m_ej, v_ej), coeffs, method=function)
    
    return interp


# In[9]:


def kilonova_simulation(times, params, dist, model = True):
    seds = BB_seds(times, params, dist)
    mag_table = LSST_magdict(seds, times, Villar, True)
    if model == True:
        light_curves = mag_table
    else:
        model_lc = check_mags(mag_table)
        sigmas = calc_sigmas(model_lc)
        random_obs = rand_mags(model_lc, sigmas)
        light_curves = [random_obs, sigmas]
    return light_curves

def check_mags(mag_table):
    new_mags = {band: [np.inf if mag > kilonova.m_5(band, kilonova.parameters)                      else mag for mag in mag_table[band]] for band in bands}
    return Table([mag_table['day'],new_mags['u'],new_mags['g'],new_mags['r'],new_mags['i'],new_mags['z'],new_mags['y']],
                 names=('day','u','g','r','i','z','y'))

def calc_sigmas(model_lc):
    sigmas = {band: [np.inf if mag > kilonova.m_5(band, kilonova.parameters)                      else kilonova.sigma_1(mag, band) for mag in model_lc[band]] for band in bands}
    return Table([model_lc['day'],sigmas['u'],sigmas['g'],sigmas['r'],sigmas['i'],sigmas['z'],sigmas['y']],
                 names=('day','u','g','r','i','z','y'))

def rand_mags(obs_lc, sigmas):
    #np.random.normal(obs_lc[band], sigmas[band])
    obs = {band: obs_lc[band] for band in bands}
    return Table([obs_lc['day'],obs['u'],obs['g'],obs['r'],obs['i'],obs['z'],obs['y']],
                 names=('day','u','g','r','i','z','y'))

