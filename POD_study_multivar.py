import sys
import os
import numpy as np
import scipy.io
from idtxl.data import Data
from idtxl.multivariate_te import MultivariateTE
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
NUM_POD = 4

# MTE parameters
MIN_LAG_S = 1   # Minimum lag for the source variable
MAX_LAG_S = 20  # Maximum lag for the source variable
MAX_LAG_T = 20  # Maximum lag for the target variable
N_PERM = 30     # Number of permutations for significance testing
ALPHA = 0.05    # Significance level for tests

# MTE settings dictionary
SETTINGS = {
    'cmi_estimator': 'JidtKraskovCMI',      # Method for estimating conditional mutual information

    'max_lag_sources': MAX_LAG_S,           # Maximum history of each source process to consider
    'min_lag_sources': MIN_LAG_S,           # Minimum history of each source process to consider
    'max_lag_target': MAX_LAG_T,            # Maximum history of the target process to consider
    
    # Alpha levels for various significance tests in the MTE algorithm
    # 'alpha_sig_te': ALPHA,                  # Significance level for the TE value of a selected source
    # 'alpha_relu_source': ALPHA,             # Significance level for a source to be selected in the greedy search
    # 'alpha_max_prediction': ALPHA,          # Significance level for prediction improvement when adding a source
    # 'alpha_min_prediction': ALPHA,          # Significance level for the omnibus prediction (overall model fit)
    
    'fdr_correction': False,                # Whether to apply False Discovery Rate correction to p-values

    # Permutation test settings
    'n_perm_max_stat': N_PERM,              # Permutations for some internal max-statistic tests
    'n_perm_min_stat': N_PERM,              # Permutations for some internal min-statistic tests
    'n_perm_max_seq': N_PERM,               # Permutations related to sequential testing or embedding
    'n_perm_omnibus': N_PERM,               # Permutations for the omnibus TE significance test
    'n_perm_redundancy': N_PERM,            # Permutations for redundancy calculations

    # Parameters for the greedy source selection process
    'max_num_selected_sources': None,       # Maximum number of influential sources to select (None for no limit)
    'min_num_samples_ratio': 2,             # Minimum number of samples required per estimated parameter
    
    # General behavior
    'verbose': False,                       # IDTxl's internal verbosity (True for detailed output to SLURM logs)
    'write_ckp': False,                     # Whether to write checkpoint files during analysis
}


# Placeholders for mp
mp_tmcoeff = None
mp_multivar_func = None
global_settings = None # To store SETTINGS for workers

# Parallel processing setup
num_proc = mp.cpu_count()

# Configure logging
LOG_FILENAME = 'mte_study_' + str(NUM_POD) + '_progress.log'
LOG_LEVEL = logging.INFO


### EXECUTION ------------------------------------------------------------------------------------------

# Multivariate Transfer Entropy function
def multivar_te(target_idx, all_modes, settings_dict):
    """Compute Multivariate Transfer Entropy for a target time series, given

    Args:
        target_idx (int): Index of the target time series in the all_modes array.
        all_modes (np.ndarray): Array containing all modes of the time series data.
        settings_dict (dict): Dictionary containing settings for the IDTxl MTE computation.

    Returns:
        tuple: A tuple containing the TE value and Pearson correlation coefficient.
    """
    
    with warnings.catch_warnings():
        # Supress the stupid fluff that IDTxl prints
        warnings.filterwarnings('ignore', category=UserWarning, message="Number of replications is not sufficient")
    
    # Data preparation for IDTxl
    # Data stacked as (samples, processes)
    data_idt = Data(all_modes, dim_order='sp', normalise=False)
    
    # TE settings
    settings = settings_dict
    
    # TE computation
    try:
        mte = MultivariateTE()
        # For multivariate TE with data stacked as [source, target]
        results_mte = mte.analyse_network(
            settings=settings_dict,
            data=data_idt,
            targets=[target_idx]
        )

        num_processes = all_modes.shape[1]  # Number of processes (modes)
        source_te_map = {src_idx_map: 0.0 for src_idx_map in range(num_processes)}
        
        # Get FDR correction boolean
        use_fdr = settings.get('fdr_correction', False)
        
        single_target_results = results_mte.get_single_target(target=target_idx, fdr=use_fdr)

        selected_vars = single_target_results.get('selected_vars_sources')
        selected_te_values = single_target_results.get('selected_sources_te')

        for i in range(len(selected_vars)):
            source_process_index = selected_vars[i][0] # This is the actual index of the source process
            te_val = selected_te_values[i]

            # If multiple lags from the same source are selected,
            # store the maximum TE contribution from that source process.
            if 0 <= source_process_index < num_processes:
                source_te_map[source_process_index] = max(source_te_map[source_process_index], te_val)
            else:
                # This case should ideally not happen if indices are consistent
                print(f"Worker Warning (target {target_idx}): Encountered an out-of-bounds "
                        f"source_process_index {source_process_index} from IDTxl results.")

        return source_te_map
    
    except Exception as e:
        print(f"Worker Error (target {target_idx}): MTE computation failed: {e}")
        return {}
    
# Worker function for parallel processing
def mte_worker(target_idx):
    global mp_tmcoeff, mp_multivar_func, global_settings    # This should make the workers fork the global variables

    if mp_tmcoeff is None or mp_multivar_func is None or global_settings is None:
        raise RuntimeError("Global data, MTE function, or MTE settings for worker not initialized.")
    
    source_te_map = mp_multivar_func( # This will call calc_mte
        target_idx=target_idx,
        all_modes=mp_tmcoeff,
        settings_dict=global_settings
    )
    
    return target_idx, source_te_map

# Function to set up a logger
def setup_logger(log_filename, level=logging.INFO):
    """Configures a logger to write to a file."""
    logger = logging.getLogger("MTE_Study_Logger") # Get a specific logger instance
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
def mte_study_parallel():
    global mp_tmcoeff, mp_multivar_func, global_settings

    logger = setup_logger(LOG_FILENAME, LOG_LEVEL)
    logger.info("Starting Multivariate Transfer Entropy (MTE) study script.")
    
    logger.info(f"Settings: Data directory: {DATA_DIR}, Modes to analyse (target for MTE): {NUM_POD}")
    logger.info(f"Lag settings from MTE_SETTINGS: min_src={SETTINGS['min_lag_sources']}, max_src={SETTINGS['max_lag_sources']}, max_tgt={SETTINGS['max_lag_target']}")
    logger.info(f"Number of worker processes: {num_proc}")
    
    ## Load data
    try:
        data = scipy.io.loadmat(os.path.join(DATA_DIR, 'A2coef.mat'))
        tmcoeff_full = data['A2'].astype(np.float64)   # Expected shape: (n_samples, n_modes)
        logger.info(f"Successfully loaded 'A2coef.mat'. Shape: {tmcoeff_full.shape}, Dtype: {tmcoeff_full.dtype}")
    except Exception as e:
        logger.exception(f"Error loading .mat file: {e}")
        return None, None, 0
    
    num_modes = tmcoeff_full.shape[1] / 2
    actual_modes = min(num_modes-1, NUM_POD)    # Exclude the mean mode (mode 0)
    
    if actual_modes < 2:
        logger.error("Less than 2 POD modes found. Cannot perform MTE analysis.")
        return None, None, actual_modes

    logger.info(f"Found {num_modes} POD modes, using {actual_modes} modes for analysis.")

    tmcoeff_pairs = tmcoeff_full[:, 1:((actual_modes+1)*2)] # We exclude the mean mode (mode 0) here
    tmcoeff_slice = tmcoeff_pairs[:, 0::2]                  # Exclude pairs (e.g., only even indices)
    
    # Normalize the data
    mean_vals = np.mean(tmcoeff_slice, axis=0)
    std_devs = np.std(tmcoeff_slice, axis=0)
    # Avoid division by zero if a mode is constant
    std_devs[std_devs == 0] = 1.0 

    mp_tmcoeff = (tmcoeff_slice - mean_vals) / std_devs
    logger.info("Data normalised (Z-score per mode).")
    
    mp_multivar_func = multivar_te
    global_settings = SETTINGS.copy()  # Copy settings for workers
    
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
    
    # MTE calculation setup
    # Create a list of tasks (targets) to process
    # Each task corresponds to a target mode for which we will compute MTE against all other modes
    tasks = list(range(mp_tmcoeff.shape[1]))
            
    # I don't think this branch is ever reached
    if not tasks:
        logger.warning("No tasks to process (e.g., only 1 mode selected).")
        # Still initialize matrices in case
        mte_matrix = np.full((mp_tmcoeff.shape[1], mp_tmcoeff.shape[1]), np.nan)
        return mte_matrix, corr_matrix, mp_tmcoeff.shape[1]
    
    logger.info(f"Starting Multivariate Transfer Entropy calculations for {len(tasks)} targets using {num_proc} processes...")
    start_time = time.time()

    chunksize = 1 # Maybe increase later if cpu utilization is low
    
    results_list = []
    with mp.Pool(processes=num_proc) as pool:
        async_result = pool.map_async(mte_worker, tasks, chunksize=chunksize)
        
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
    # mte_matrix[target, source] = conditional TE(source -> target | other_selected_sources)
    num_cols = mp_tmcoeff.shape[1]
    mte_matrix = np.full((num_cols, num_cols), 0.0) # Default to 0 if not selected

    for target_idx, source_te_map in results_list:
        if source_te_map is None or not isinstance(source_te_map, dict):
            logger.warning(f"Received invalid/None source-TE map for target {target_idx}. Skipping.")
            continue
        for source_idx, conditional_te_val in source_te_map.items():
            if 0 <= target_idx < num_cols and \
               0 <= source_idx < num_cols:
                mte_matrix[target_idx, source_idx] = conditional_te_val
            else:
                logger.error(f"Invalid indices from MTE worker: target={target_idx}, source={source_idx} for matrix {num_cols}x{num_cols}")
                
    # Get pearson correlation matrix
    logger.info("Calculating full Pearson correlation matrix on normalized data...")
    corr_matrix = np.full((num_cols, num_cols), np.nan)
    for i in range(num_cols):
        for j in range(num_cols):
            if i == j:
                corr_matrix[i, j] = 1.0
            else:
                corr_matrix[i, j] = np.corrcoef(mp_tmcoeff[:, i], mp_tmcoeff[:, j])[0, 1]
    
    corr_matrix_filename = "mte_pearson_corr_" + str(NUM_POD) + ".csv"
    np.savetxt(corr_matrix_filename, corr_matrix, delimiter=",")
    logger.info(f"Full Pearson correlation matrix saved to {corr_matrix_filename}")
    logger.info(f"Correlation Matrix sample (first 3x3 or less):\n{corr_matrix[:3,:3]}")
        
    # Save and log the MTE results
    mte_matrix_filename = "mte_" + str(NUM_POD) + ".csv"
    np.savetxt(mte_matrix_filename, mte_matrix, delimiter=",")
    logger.info(f"Multivariate TE matrix saved to {mte_matrix_filename}")
    logger.info(f"MTE Matrix (Target x Source) sample (first 3x3 or less):\n{mte_matrix[:3,:3]}")
        
    return mte_matrix, corr_matrix, num_cols

if __name__ == "__main__":
    # Run the TE study in parallel
    mte_matrix, corr_matrix, modes_count = mte_study_parallel()
    
    if mte_matrix is not None and corr_matrix is not None:
        # Using print here for final feedback to SLURM stdout if desired, or remove
        print(f"MTE study complete. Processed {modes_count} dynamic modes.")
        print(f"MTE Matrix shape: {mte_matrix.shape}")
        print(f"Correlation Matrix shape: {corr_matrix.shape}")
        print(f"Results saved. Check '{LOG_FILENAME}', 'multivariate_te_matrix.csv', and 'pearson_corr_matrix_mte_study.csv'.")
    else:
        print("MTE study did not complete successfully. Check log file.")