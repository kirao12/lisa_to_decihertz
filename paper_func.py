import h5py as h5
import numpy as np
from scipy.integrate import quad

import astropy.units as u
import astropy.constants as c
import astropy.coordinates as coords
from astropy import constants as const

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from importlib import reload

import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'

import cogsworth

from gala import dynamics as gd

from legwork import evol, utils, source

import os
import argparse

import time

import matplotlib.ticker as mticker
import matplotlib.patheffects as pe

from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from legwork.source import Source
import legwork.psd as psd
import astropy.units as u
import legwork.strain as strain
import legwork

from astropy.visualization import quantity_support


def load_and_filter_lisa_decigo(
    harmonics_files,
    population_files,
    lisa_key="max_harm_info_lisa",
    decigo_key="max_harm_info_decigo",
    population_key="final_bpp",
    snr_cut=7,
):
    """
    Parameters
    ----------
    harmonics_files : dict
        Dictionary mapping merger_type -> filepath
        e.g. {
            "BBH": "/path/bhbh_harmonics.h5",
            "BHNS": "/path/bhns_harmonics.h5",
            "BNS": "/path/nsns_harmonics.h5",
        }

    population_files : dict
        Dictionary mapping merger_type -> filepath
        e.g. {
            "BBH": "/path/BHBH-f.h5",
            "BHNS": "/path/BHNS-f.h5",
            "BNS": "/path/NSNS-f.h5",
        }

    lisa_key : str
        HDF5 key for LISA harmonics

    decigo_key : str
        HDF5 key for DECIGO harmonics

    population_key : str
        HDF5 key for population data

    snr_cut : float
        SNR threshold

    Returns
    -------
    df_filtered_decigo : pd.DataFrame
    df_filtered_lisa : pd.DataFrame
    decigo : pd.DataFrame
    lisa : pd.DataFrame
    df : pd.DataFrame
    """
    
    start = time.time()

    lisa_list = []
    decigo_list = []
    pop_list = []

    for merger_type, harm_path in harmonics_files.items():
        lisa_df = pd.read_hdf(harm_path, key=lisa_key)
        decigo_df = pd.read_hdf(harm_path, key=decigo_key)

        lisa_df["merger_type"] = merger_type
        decigo_df["merger_type"] = merger_type

        lisa_list.append(lisa_df)
        decigo_list.append(decigo_df)

        pop_df = pd.read_hdf(population_files[merger_type], key=population_key)
        pop_df["merger_type"] = merger_type
        pop_list.append(pop_df)
    
    print("finished loading files")

    lisa = pd.concat(lisa_list, ignore_index=True).dropna()
    decigo = pd.concat(decigo_list, ignore_index=True).dropna()
    df = pd.concat(pop_list, ignore_index=True)

    lisa = lisa[lisa["max_snr"] > snr_cut]
    decigo = decigo[decigo["max_snr"] > snr_cut]

    df_filtered_lisa = df.loc[df.index.intersection(lisa.index)].copy()
    df_filtered_decigo = df.loc[df.index.intersection(decigo.index)].copy()
    
    print("finished filtering dataframes")

    lisa["ecc"] = df_filtered_lisa["ecc"]
    decigo["ecc"] = df_filtered_decigo["ecc"]

    lisa["bin_num"] = lisa.index
    decigo["bin_num"] = decigo.index

    df_filtered_lisa["bin_num"] = lisa["bin_num"]
    df_filtered_decigo["bin_num"] = decigo["bin_num"]
    df['bin_num'] = df.index
    
    print("loading and filtering complete, it took {:1.1f} seconds".format(time.time() - start))

    return df_filtered_decigo, df_filtered_lisa, decigo, lisa, df


def sensitivity_curve(
    filepath,
    frequency_col,
    strain_col,
    detector=None
):
    """
    Read a sensitivity curve and return a PSD(f) callable.

    Parameters
    ----------
    filepath : str
        path to sensitivity curve file
    detector : str, optional
        name of detector (for future use)
    frequency_col : str
        column name for frequency [Hz]
    strain_col : str
        column name for characteristic strain
        
    Returns
    --------
    f : astropy quantity
        frquency array with units [Hz]
    h : astropy quantity (?)
        characteristic strain array
    """

    sc = pd.read_csv(filepath, sep='\s+')

    f = sc[frequency_col].values
    h = sc[strain_col].values

    # PSD = h^2 / f
    psd_interp = interp1d(
        f,
        (h**2) / f,
        bounds_error=False,
        fill_value=np.inf,
    )

    def psd(f):
        """
        Power spectral density.

        Parameters
        ----------
        f : astropy quantity
            frequency array with units

        Returns
        -------
        PSD : astropy Quantity
            PSD with units of 1/Hz
        """
        return psd_interp(f.to(u.Hz).value) / u.Hz

    return f, h, psd

def chirp_mass(m_1, m_2):
    """Computes chirp mass of binaries (from LEGWORK)

    Parameters
    ----------
    m_1 : `float/array`
        Primary mass

    m_2 : `float/array`
        Secondary mass

    Returns
    -------
    m_c : `float/array`
        Chirp mass
    """
    m_c = (m_1 * m_2)**(3/5) / (m_1 + m_2)**(1/5)

    # simplify units if present
    if isinstance(m_c, u.quantity.Quantity):
        m_c = m_c.to(u.Msun)

    return m_c


NtargetMW_BBH = 732066.62
NtargetMW_BHNS = 330892.60
NtargetMW_BNS = 95336.13

def N_detect(population, run, merger_type, detection_key, NtargetMW):
    """Computes N_detect for each realization (run)
    
    Parameters
    ----------
    population : pd.Dataframe
        dataframe of full population of binaries
    run : pd.Dataframe[column]
        column of dataframe showing each realization (run) for each system
    merger_type : str
        e.g. merger_type={BBH, BHNS, BNS}
    detection_key : pd.Dataframe[column]
        e.g. theta_target_decigo
    NtargetMW : float
    
    Returns
    -------
    N_detect : array
    """
    df = population.loc[population.run==run]  # get the specific MW run
    df = df.loc[df.merger_type==merger_type]   # get the specific merger type
    full_weights_sum = df['full_weights_sum'].iloc[0]   # get the denominator of f_detect

    # check to make sure that full_weights_sum is the same for all rows
    assert (df['full_weights_sum'] == full_weights_sum).all(), "full_weights_sum isn't all the same value!"

    # calculate numerator of f_detect
    fdet_numerator = np.sum(df[detection_key]*df['mixture_weight']*df['MW_weights'])

    # calculate f_detect
    fdet = fdet_numerator / full_weights_sum

    # get N_detect
    N_detect = fdet * NtargetMW
    return N_detect


def percentile_stats(df, types, value_col="N_detect"):
    """Computes the median, 95%, and 5% statistics for N_detect by merger type
    
    Parameters
    ----------
    df: pd.DataFrame
        that includes N_detect and merger_type
    types: array
        e.g. types = ["BBH", "BHNS", "BNS"]
    
    Returns:
    --------
    medians: list
    yerr: array

    """
    medians = []
    lower_errors = []
    upper_errors = []

    for mtype in types:
        subset = df[df["merger_type"] == mtype]

        low = np.percentile(subset[value_col], 5)
        med = np.percentile(subset[value_col], 50)
        high = np.percentile(subset[value_col], 95)

        medians.append(med)
        lower_errors.append(med - low)
        upper_errors.append(high - med)

    yerr = np.array([lower_errors, upper_errors])

    return medians, yerr


def sample_binaries(df, types, ecc_min=0.00, n_samples=5,
                    ecc_col="ecc", random_state=None):
    """
    Randomly sample binaries by merger type with eccentricity cut.
    
    Parameters:
    -----------
    df: pd.Dataframe
        dataframe including eccentricity data
        
    types: array
        e.g. types = ["BBH", "BHNS", "BNS"]
    
    ecc_min: float
    
    n_samples: integer

    Returns:
    ---------
        Dictionary of sampled dataframes keyed by merger type.
        
    """
    samples = {}

    for mtype in types:
        subset = df[
            (df["merger_type"] == mtype) &
            (df[ecc_col] >= ecc_min)
        ]

        if len(subset) == 0:
            print(f"No systems found for {mtype} with e >= {ecc_min}")
            samples[mtype] = None
            continue

        n_draw = min(n_samples, len(subset))

        samples[mtype] = subset.sample(
            n=n_draw,
            random_state=random_state
        )

    return samples
