import sys
import os
import numpy as np
import scipy.io
from idtxl.data import Data
from idtxl.bivariate_te import BivariateTE
import warnings
import multiprocessing as mp
import time
import logging

### SETTINGS ------------------------------------------------------------------------------------------

# Data directory
DATA_DIR = "./data/"

# Number of POD modes
# Mode 0 is the mean, so we should not include it in the analysis.
# For this script, we assume 1-indexed modes from tmcoeff 
# (as in, tmcoeff[:, 0] is the mean mode, so we start from 1).
NUM_POD = 20

# TE parameters
MIN_LAG_S = 1   # Minimum lag for the source variable
MAX_LAG_S = 50  # Maximum lag for the source variable
MAX_LAG_T = 50  # Maximum lag for the target variable
N_PERM = 100    # Number of permutations for significance testing
ALPHA = 0.05    # Significance level for tests

# TE settings dictionary
SETTINGS = {
    'cmi_estimator': 'JidtKraskovCMI',

    'max_lag_sources': MAX_LAG_S,       # Maximum history of source process
    'min_lag_sources': MIN_LAG_S,       # Minimum history of source process
    'max_lag_target': MAX_LAG_T,        # Maximum history of target process

    'n_perm_max_stat': N_PERM,          # Permutations for max_stat part of analysis
    'n_perm_min_stat': N_PERM,          # Permutations for min_stat part of analysis
    'n_perm_max_seq': N_PERM,           # Permutations for max_seq part of analysis
    'n_perm_omnibus': N_PERM,           # Permutations for omnibus significance test

    'alpha_max_stat': ALPHA,            # Significance level for max_stat test
    'alpha_min_stat': ALPHA,            # Significance level for min_stat test
    'alpha_max_seq': ALPHA,             # Significance level for max_seq test
    'alpha_omnibus': ALPHA,             # Significance level for omnibus test

    'fdr_correction': False,            # Set to True to apply FDR correction

    'verbose': False,                   # IDTxl's internal verbosity
    'write_ckp': False,                 # Checkpoint writing
    
    # 'reuse_missing_data_ests': True, # In case we get missing data estimates
}

# Placeholders for mp
mp_tmcoeff = None
mp_bivar_func = None

# Parallel processing setup
num_proc = mp.cpu_count()

# Configure logging
LOG_FILENAME = 'bte_study_' + str(NUM_POD) + '_progress.log'
LOG_LEVEL = logging.INFO


### EXECUTION ------------------------------------------------------------------------------------------

# Bivarate Transfer Entropy function
def bivar_te(source, target, settings_dict):
    """Compute Bivariate Transfer Entropy between source and target time series.

    Args:
        source (np.ndarray): Source time series data.
        target (np.ndarray): Target time series data.
        settings_dict (dict): Dictionary containing settings for the IDTxl TE computation.

    Returns:
        tuple: A tuple containing the TE value and Pearson correlation coefficient.
    """
    
    
    with warnings.catch_warnings():
        # Supress the stupid fluff that IDTxl prints
        warnings.filterwarnings('ignore', category=UserWarning, message="Number of replications is not sufficient")
    
    n = np.size(source)
    if n != np.size(target):
        print("Error: source and target must have the same number of samples.")
        return np.nan, np.nan
    
    # Reshape to (n, 1)
    source_reshaped = source.reshape(-1, 1)
    target_reshaped = target.reshape(-1, 1)
    
    # Pearson correlation
    corr = np.corrcoef(source_reshaped[:, 0], target_reshaped[:, 0])[0, 1]
    
    # Data preparation for IDTxl
    # Data stacked as (samples, processes)
    data_idt = Data(np.hstack((source_reshaped, target_reshaped)),
                dim_order='sp',  # samples, processes
                normalise=False) # We do it in the preprocessing step
    
    # TE settings
    settings = settings_dict.copy()
    
    # TE computation
    try:
        bte = BivariateTE()
        # For bivariate TE with data [source, target]
        # source is process 0, target is process 1
        results_bte = bte.analyse_single_target(
            settings=settings,
            data=data_idt,
            target=1,
            sources=[0]
        )
        
        # Get the TE result for the target (index 1)
        # fdr=False means we are not applying False Discovery Rate correction here
        # to filter results based on p-values. We get the raw calculation.
        single_process_obj = results_bte.get_single_target(target_id=1, fdr=False)
        
        if single_process_obj is not None and hasattr(single_process_obj, 'omnibus_te'):
            te_value = single_process_obj.omnibus_te
            # p_value = single_process_te_obj.p_value_omnibus_te
            # print(f"  TE value: {te_value:.4f}, p-value: {p_value if p_value is not None else 'N/A'}")
            return te_value, corr
        else:
            # This case might occur if no significant interaction is found or an issue arises
            print(f"No significant TE found or result object is None. source: {source_reshaped}, target: {target_reshaped}")
            return np.nan, corr
    
    # except Exception as e:
    #     print(f"Error during TE computation: {e}")
    #     return np.nan, corr
    
    except Exception as e:
        # print(f"Worker Error (target {}): MTE computation failed: {e}")
        print(f"Worker Error (target {target}): MTE computation failed: {e}")
        return np.nan, corr
    
# Worker function for parallel processing
def te_worker(args_tuple):
    target_idx, source_idx, settings_dict = args_tuple
    
    if mp_tmcoeff is None or mp_bivar_func is None:
        raise RuntimeError("Global data/function for worker not initialized.")
    
    source_ts = mp_tmcoeff[:, source_idx]
    target_ts = mp_tmcoeff[:, target_idx]
    
    te_val, corr_val = bivar_te(
        source=source_ts,
        target=target_ts,
        settings_dict=settings_dict
    )
    
    return target_idx, source_idx, te_val, corr_val

# Function to set up a logger
def setup_logger(log_filename, level=logging.INFO):
    """Configures a logger to write to a file."""
    logger = logging.getLogger("TE_Study_Logger") # Get a specific logger instance
    if logger.hasHandlers(): # Prevent adding multiple handlers if called again
        logger.handlers.clear()
        
    logger.setLevel(level)
    
    # File Handler
    fh = logging.FileHandler(log_filename, mode='w') # 'w' to overwrite, 'a' to append
    fh.setLevel(level)
    
    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    fh.setFormatter(formatter)
    
    logger.addHandler(fh)
    return logger

# Main function to run the TE study in parallel
def te_study_parallel():
    global mp_tmcoeff, mp_bivar_func

    logger = setup_logger(LOG_FILENAME, LOG_LEVEL)
    logger.info("Starting Bivariate Transfer Entropy study script.")
    
    logger.info(f"Settings: Data directory: {DATA_DIR}, Modes to analyse: {NUM_POD}")
    logger.info(f"Lag settings: min_src={MIN_LAG_S}, max_src={MAX_LAG_S}, max_tgt={MAX_LAG_T}")
    logger.info(f"Number of worker processes: {num_proc}")
    
    ## Load data
    try:
        data = scipy.io.loadmat(os.path.join(DATA_DIR, 'A2coef.mat'))
        tmcoeff_full = data['A2'].astype(np.float64)   # Expected shape: (n_samples, n_modes)
        logger.info(f"Successfully loaded 'A2coef.mat'. Shape: {tmcoeff_full.shape}, Dtype: {tmcoeff_full.dtype}")
    except Exception as e:
        logger.exception(f"Error loading .mat file: {e}")
        return None, None, 0
    
    num_modes = tmcoeff_full.shape[1]
    actual_modes = min(num_modes, NUM_POD)
    
    if actual_modes < 2:
        logger.error("Less than 2 POD modes found. Cannot perform TE analysis.")
        return None, None, actual_modes

    logger.info(f"Found {num_modes} POD modes, using {actual_modes} modes for analysis.")
    
    tmcoeff_slice = tmcoeff_full[:, 1:actual_modes+1] # We exclude the mean mode (mode 0) here
    
    # Normalize the data
    mean_vals = np.mean(tmcoeff_slice, axis=0)
    std_devs = np.std(tmcoeff_slice, axis=0)
    # Avoid division by zero if a mode is constant
    std_devs[std_devs == 0] = 1.0 

    mp_tmcoeff = (tmcoeff_slice - mean_vals) / std_devs
    logger.info("Data normalised (Z-score per mode).")
    
    mp_bivar_func = bivar_te
    
    logger.info(f"Manually normalized mp_tmcoeff dtype: {mp_tmcoeff.dtype}")
    if np.any(np.isnan(mp_tmcoeff)):
        logger.warning("CRITICAL: Manually normalized mp_tmcoeff CONTAINS NaNs!")
        logger.warning(f"NaN counts per column: {np.sum(np.isnan(mp_tmcoeff), axis=0)}")
    if np.any(np.isinf(mp_tmcoeff)):
        logger.warning("CRITICAL: Manually normalized mp_tmcoeff CONTAINS Infs!")
        logger.warning(f"Inf counts per column: {np.sum(np.isinf(mp_tmcoeff), axis=0)}")

    logger.info(f"Manually normalized mp_tmcoeff min/max values per column (ignoring NaNs if any):")
    logger.info(f"Mins: {np.nanmin(mp_tmcoeff, axis=0)}")
    logger.info(f"Maxs: {np.nanmax(mp_tmcoeff, axis=0)}")
    
    tasks = []
    for source_idx in range(actual_modes):
        for target_idx in range(actual_modes):
            if target_idx == source_idx:
                continue # Skip TE from a mode to itself
            task_args = (
                target_idx, source_idx, SETTINGS
            )
            tasks.append(task_args)
            
    if not tasks:
        logger.warning("No tasks to process (e.g., only 1 mode selected).")
        # Still initialize matrices in case
        te_matrix = np.full((actual_modes, actual_modes), 0.0 if actual_modes > 0 else np.nan)
        corr_matrix = np.full((actual_modes, actual_modes), 1.0 if actual_modes > 0 else np.nan)
        if actual_modes == 1: # Fill diagonal for single mode case
             te_matrix[0,0] = 0.0
             corr_matrix[0,0] = 1.0
        return te_matrix, corr_matrix
    
    logger.info(f"Starting Bivariate Transfer Entropy calculations for {len(tasks)} pairs using {num_proc} processes...")
    start_time = time.time()

    chunksize = 1 # Maybe increase later if cpu utilization is low
    
    results_list = []
    with mp.Pool(processes=num_proc) as pool:
        async_result = pool.map_async(te_worker, tasks, chunksize=chunksize)
        
        # Progress tracking
        total_tasks = len(tasks)
        # Estimate number of tasks per chunk (can be off)
        num_tasks_chunk = (total_tasks + chunksize -1) // chunksize 
        
        while not async_result.ready():
            # _number_left refers to chunks, not individual tasks, if chunksize > 1
            num_left = async_result._number_left
            completed = num_tasks_chunk - num_left
            
            percent = int((completed / num_tasks_chunk) * 100) if num_tasks_chunk > 0 else 0

            logger.info(f"Progress: {percent}% (approx. {completed}/{num_tasks_chunk} chunks processed)")
            time.sleep(30)
            
        results_list = async_result.get()
            
    logger.info(f"Progress: 100% ({total_tasks}/{total_tasks} pairs processed)")
    logger.info(f"Parallel calculations finished. Time taken: {time.time() - start_time:.2f} seconds")
    
    # Assemble results into matrices
    te_matrix_bivar = np.full((actual_modes, actual_modes), np.nan)
    corr_matrix = np.full((actual_modes, actual_modes), np.nan)

    # Fill diagonal for self-loops (TE=0, Corr=1)
    for i in range(actual_modes):
        te_matrix_bivar[i, i] = 0.0
        corr_matrix[i, i] = 1.0
        
    for res_target_idx, res_source_idx, te_val, corr_val in results_list:
        te_matrix_bivar[res_target_idx, res_source_idx] = te_val
        corr_matrix[res_target_idx, res_source_idx] = corr_val
        
    # Save and log the results
    te_matrix_filename = "bivariate_te_matrix.csv"
    corr_matrix_filename = "bte_pearson_correlation_matrix.csv"
    np.savetxt(te_matrix_filename, te_matrix_bivar, delimiter=",")
    logger.info(f"Bivariate TE matrix saved to {te_matrix_filename}")
    np.savetxt(corr_matrix_filename, corr_matrix, delimiter=",")
    logger.info(f"Pearson correlation matrix saved to {corr_matrix_filename}")
    
    logger.info(f"TE Matrix (Target x Source) sample (first 3x3 or less):\n{te_matrix_bivar[:3,:3]}")
    logger.info(f"Correlation Matrix (Target x Source) sample (first 3x3 or less):\n{corr_matrix[:3,:3]}")
        
    return te_matrix_bivar, corr_matrix, actual_modes

if __name__ == "__main__":
    # Run the TE study in parallel
    te_matrix, corr_matrix, actual_modes = te_study_parallel()
    
    if te_matrix is not None and corr_matrix is not None:
        print(f"TE Matrix shape: {te_matrix.shape}, Correlation Matrix shape: {corr_matrix.shape}")
        print(f"Actual modes used for analysis: {actual_modes}")
    else:
        print("TE study did not complete successfully.")