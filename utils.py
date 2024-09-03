import numpy as np
from scipy.stats import f, chi2
from scipy.io import mmread
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool

gray = '#8b96ad'
red = '#c74546'

def estimate_s(N1, N2=None, plot=True, ax=None, min_mean=0.1, max_mean=np.inf, 
               bins=np.arange(-0.5, 1.5, 0.01) - 0.005, color='lightgray', modcolor='#0070c0', 
               meancolor='#3d405b'):
    """
    Estimates the extrinsic noise `s`.

    This function computes normalized covariance to estimate `s`.
    Optionally, it can plot a histogram of the normalized covariance values and highlight the mean and
    mode.

    Parameters
    ----------
    N1 : ndarray
        A 2D numpy array representing the first gene count matrix with
        cells as rows and genes as columns.

    N2 : ndarray, optional
        A 2D numpy array representing the second gene count matrix with
        cells as rows and genes as columns. If `None`, the calculation is performed
        only on `N1`. Default is `None`.

    plot : bool, optional
        If `True`, a histogram of the covariance values is plotted. Default is `True`.

    ax : matplotlib.axes.Axes, optional
        A matplotlib axis object where the histogram will be plotted. If `None`,
        a new figure and axis are created. Default is `None`.

    min_mean : float, optional
        The minimum mean expression threshold for genes to be included in the
        calculation. Default is 0.1.

    max_mean : float, optional
        The maximum mean expression threshold for genes to be included in the
        calculation. Default is `np.inf`.

    bins : ndarray, optional
        The bins for the histogram. Default is `np.arange(0, 1, 0.01) - 0.005`.

    color : str, optional
        The color of the histogram bars. Default is `'lightgray'`.

    modcolor : str, optional
        The color of the vertical line indicating the mode of the histogram. Default is `'#0070c0'`.

    meancolor : str, optional
        The color of the vertical line indicating the mean of the histogram. Default is `'#3d405b'`.

    Returns
    -------
    s_mod : float
        The mode of the normalized covariance values calculated as the midpoint of the most
        frequent bin in the histogram.

    """
    ### calculate normalized covariance
    if N2 is None:
        idx = (N1.mean(0) > min_mean) & (N1.mean(0) < max_mean)
        X = N1[:, idx]
        X_mean = X.mean(0)
        p = len(X_mean)
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        np.fill_diagonal(eta, np.nan)
        eta = eta[~np.isnan(eta)]

    else:
        idx1 = (N1.mean(0) > min_mean) & (N1.mean(0) < max_mean)
        idx2 = (N2.mean(0) > min_mean) & (N2.mean(0) < max_mean)
        X = np.concatenate((N1[:, idx1], N2[:, idx2]), axis=1)
        X_mean = X.mean(0)
        p1 = idx1.sum()
        p2 = idx2.sum()
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        eta = eta[p1:, :p1]
        
    ### calculate s as the mean
    s = np.mean(eta)

    ### calculate s_mod as the midpoint of the most frequent bin in the histogram.
    if plot is False:
        hist, bins = np.histogram(eta.flatten(), bins=bins)
        s_mod = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if N2 is None:
            hist, bins, patches = ax.hist(eta.flatten(), bins=bins, label=str(p) + ' genes', color=color)
        else:
            hist, bins, patches = ax.hist(eta.flatten(), bins=bins, label=str(p1) + r'$\times$' + str(p2) + ' genes', color=color)
        s_mod = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
        ax.axvline(x=s, c=meancolor, zorder=0, linewidth=6, label='mean=' + str(np.around(s, 3)))
        ax.axvline(x=s_mod, c=modcolor, zorder=0, linewidth=6, label='mode=' + str(np.around(s_mod, 3)))
        ax.legend(loc='upper right')
    return {'mod':s_mod,'mean':s}

def bootstrapping_s_single(b, bootstrap_indices, data1, data2, min_mean):
    """
    Computes the extrinsic noise `s` for a single bootstrap sample.

    Parameters
    ----------
    b : int
        The bootstrap sample index.
    bootstrap_indices : ndarray
        Array of indices for resampling the data.
    data1 : ndarray
        The first dataset to compute extrinsic noise `s`.
    data2 : ndarray or None
        The second dataset to compute extrinsic noise `s`, or None if not used.
    min_mean : float
        Minimum mean threshold to filter features.

    Returns
    -------
    s1 : float
        The estimated extrinsic noise `s` for `data1`.
    s2 : float or None
        The estimated extrinsic noise `s` for `data2`, or None if `data2` is not provided.
    """
    b_idx = bootstrap_indices[b]

    # Process data1
    N1 = data1[b_idx]
    idx = (N1.mean(0) > min_mean)
    X = N1[:, idx]
    X_mean = X.mean(0)
    p = len(X_mean)
    eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
    np.fill_diagonal(eta, np.nan)
    eta = eta[~np.isnan(eta)]
    s1_mean = np.mean(eta)
    hist, bins = np.histogram(eta.flatten(), bins=np.arange(0, 1, 0.01) - 0.005)
    s1 = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2

    # Process data2 if provided
    if data2 is not None:
        N2 = data2[b_idx]
        idx = (N2.mean(0) > min_mean)
        X = N2[:, idx]
        X_mean = X.mean(0)
        p = len(X_mean)
        eta = np.cov(X, rowvar=False) / X_mean[:, None] / X_mean[None, :]
        np.fill_diagonal(eta, np.nan)
        eta = eta[~np.isnan(eta)]
        s2_mean = np.mean(eta)
        hist, bins = np.histogram(eta.flatten(), bins=np.arange(0, 1, 0.01) - 0.005)
        s2 = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    else:
        s2 = None

    return s1, s2


def bootstrapping_s(data1, data2=None, B=1000, seed=0, min_mean=0.1, n_cores=1):
    """
    Performs bootstrapping to estimate the extrinsic noise `s` for given datasets.

    Parameters
    ----------
    data1 : ndarray
        The first dataset to compute extrinsic noise `s`.
    data2 : ndarray or None, optional
        The second dataset to compute extrinsic noise `s`, or None if not used.
    B : int, optional
        Number of bootstrap samples. Default is 1000.
    seed : int, optional
        Seed for random number generation. Default is 0.
    min_mean : float, optional
        Minimum mean threshold to filter features. Default is 0.1.
    n_cores : int, optional
        Number of cores to use for parallel processing. Default is 1.

    Returns
    -------
    s_bootstrap : ndarray
        Array of extrinsic noise estimates `s` for each bootstrap sample.
    """
    np.random.seed(seed)
    n, p = data1.shape
    bootstrap_indices = np.random.choice(n, size=(B, n), replace=True)

    # Create a pool of workers
    with Pool(processes=n_cores) as pool:
        # Map the worker function to each bootstrap sample
        s_bootstrap = list(pool.starmap(bootstrapping_s_single, 
                                             [(b, bootstrap_indices, data1, data2, min_mean) for b in range(B)]), 
                                total=B)

    return np.array(s_bootstrap)
    
def bootstrapping_var_single(args):
    """
    Computes the bootstrapped variance and residue for a single sample, including extrinsic noise estimation.

    Parameters
    ----------
    args : tuple
        A tuple containing:
        - data : ndarray
            Original dataset with observations as rows and features as columns.
        - indices : ndarray
            Indices for resampling the data.
        - bootstrap_s : bool
            If True, calculates the extrinsic noise `s` for covariance adjustment.

    Returns
    -------
    residue : ndarray
        The adjusted residue of the variance for each feature.
    s : float
        The estimated extrinsic noise `s` if `bootstrap_s` is True, otherwise 0.
    """
    data, indices, bootstrap_s = args
    X = data[indices]  # Resample data using given indices
    bootstrap_var = X.var(axis=0)
    bootstrap_mean = X.mean(0)

    if bootstrap_s:
        # Compute mode of normalized covariance s as the extrinsic noise
        eta = np.cov(X, rowvar=False) / bootstrap_mean[:, None] / bootstrap_mean[None, :]
        np.fill_diagonal(eta, np.nan)
        eta = eta[~np.isnan(eta)]
        hist, bins = np.histogram(eta.flatten(), bins=np.arange(0, 1.0, 0.01) - 0.005)
        s = (bins[np.argmax(hist)] + bins[np.argmax(hist) + 1]) / 2
    else:
        s = 0

    # Calculate residue
    residue = (bootstrap_var - bootstrap_mean) / bootstrap_mean**2 - s
    return residue, s


def bootstrapping_var(data, bootstrap_s=False, alpha=0.05, B=1000, seed=0, num_cores=1):
    """
    Performs bootstrapping to estimate the confidence intervals for (variance - mean)/mean^2, including extrinsic noise estimation.

    Parameters
    ----------
    data : ndarray
        The original dataset with observations as rows and features as columns.
    bootstrap_s : bool, optional
        If True, computes the extrinsic noise `s` for each bootstrap sample. Default is False.
    alpha : float, optional
        Significance level for confidence intervals. Default is 0.05.
    B : int, optional
        Number of bootstrap samples. Default is 1000.
    seed : int, optional
        Seed for random number generation. Default is 0.
    num_cores : int, optional
        Number of cores to use for parallel processing. Default is 1.

    Returns
    -------
    lower_bound : ndarray
        Lower bound of the confidence interval for the variance residues.
    upper_bound : ndarray
        Upper bound of the confidence interval for the variance residues.
    bootstrap_s : ndarray
        The extrinsic noise `s` for each bootstrap sample if `bootstrap_s` is True.
    """
    np.random.seed(seed)
    n, p = data.shape
    bootstrap_indices = np.random.choice(n, size=(B, n), replace=True)
    
    # Prepare arguments for multiprocessing
    args = [(data, bootstrap_indices[b], bootstrap_s) for b in range(B)]
    
    bootstrap_residue = np.zeros((B, p))
    bootstrap_s = np.zeros(B)

    # Use multiprocessing Pool to perform bootstrapping in parallel
    with Pool(processes=num_cores) as pool:
        results = list(tqdm(pool.imap(bootstrapping_var_single, args), total=B))

    # Collect results from each bootstrap sample
    for i, (residue, s) in enumerate(results):
        bootstrap_residue[i] = residue
        bootstrap_s[i] = s

    # Calculate the confidence interval for the residues
    lower_bound = np.nanpercentile(bootstrap_residue, alpha/2 * 100, axis=0)
    upper_bound = np.nanpercentile(bootstrap_residue, (1 - alpha/2) * 100, axis=0)

    return lower_bound, upper_bound, bootstrap_s
    
def CCC(y1, y2):
    """
    Calculates the Concordance Correlation Coefficient (CCC) between two sets of ratings.

    The CCC evaluates the agreement between two variables by measuring both precision 
    (the Pearson correlation) and accuracy (the deviation from the 45-degree line through the origin).

    Parameters
    ----------
    y1 : array-like
        First set of ratings or measurements.
    y2 : array-like
        Second set of ratings or measurements.

    Returns
    -------
    CCC : float
        The Concordance Correlation Coefficient between `y1` and `y2`, ranging from -1 to 1.
        A value of 1 indicates perfect concordance, 0 indicates no concordance, and -1 indicates perfect discordance.
    """
    # Convert ratings to numpy arrays
    y1_array = np.array(y1)
    y2_array = np.array(y2)
    
    # Calculate means
    mean_y1 = np.mean(y1_array)
    mean_y2 = np.mean(y2_array)
    
    # Calculate variances
    var_y1 = np.var(y1_array, ddof=1)
    var_y2 = np.var(y2_array, ddof=1)
    
    # Calculate covariance
    cov_y1y2 = np.cov(y1_array, y2_array)[0, 1]
    
    # Calculate bias correction factor
    CCC = 2 * cov_y1y2 / (var_y1 + var_y2 + (mean_y1 - mean_y2) ** 2)
    
    return CCC

def load_10x(datadir):
    """
    Loads 10x Genomics single-cell RNA-seq data from a specified directory.

    This function reads the barcodes, features, and matrix files typically found
    in a 10x Genomics output directory and constructs an `AnnData` object.

    Parameters
    ----------
    datadir : str
        Path to the directory containing the 10x Genomics data files
        (`barcodes.tsv`, `features.tsv`, and `matrix.mtx`).

    Returns
    -------
    tenx : AnnData
        An `AnnData` object containing the expression matrix with cells as observations
        and genes/features as variables.
    """
    barcode = pd.read_csv(datadir + '/barcodes.tsv', sep='\t', header=None)
    feature = pd.read_csv(datadir + '/features.tsv', sep='\t', header=None)
    matrix = mmread(datadir + '/matrix.mtx')
    tenx = ad.AnnData(matrix.T, obs=barcode, var=feature)
    return tenx

def intersect_idx(kb_bcd, tenx_bcd):
    """
    Finds the indices of common barcodes between two lists.

    Parameters
    ----------
    kb_bcd : array-like
        List or array of barcodes from the first dataset.
    tenx_bcd : array-like
        List or array of barcodes from the second dataset.

    Returns
    -------
    kb_common_bc_idx : ndarray
        Indices of the common barcodes in `kb_bcd`.
    tenx_common_bc_idx : ndarray
        Indices of the common barcodes in `tenx_bcd`.
    """
    common_bc = np.intersect1d(np.array(kb_bcd), np.array(tenx_bcd))

    # Find indices of common barcodes in both lists
    kb_common_bc_idx = np.array([np.where(np.array(kb_bcd) == bc)[0][0] for bc in common_bc])
    tenx_common_bc_idx = np.array([np.where(np.array(tenx_bcd) == bc)[0][0] for bc in common_bc])

    return kb_common_bc_idx, tenx_common_bc_idx

def get_ensg_id(gene_name):
    base_url = "https://rest.ensembl.org"
    endpoint = f"/xrefs/symbol/human/{gene_name}"

    url = f"{base_url}{endpoint}"

    response = requests.get(url, headers={"Content-Type": "application/json"})

    if response.status_code == 200:
        data = response.json()
        if data:
            ensg_id = data[0]['id']
            return ensg_id
        else:
            return None
    else:
        print(f"Failed to retrieve data. Status code: {response.status_code}")
        return None

def calculate_gene_length(gtf_file):
    gene_lengths = {}

    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            if fields[2] == 'gene':
                gene_id = fields[8].split(';')[0].split('"')[1]
                start = int(fields[3])
                end = int(fields[4])
                length = end - start + 1  # Add 1 to include both start and end positions
                if gene_id not in gene_lengths:
                    gene_lengths[gene_id] = length
                else:
                    gene_lengths[gene_id] += length

    return gene_lengths

def calculate_exon_number(gtf_file):
    exon_counts = {}

    with open(gtf_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue  # Skip header lines
            fields = line.strip().split('\t')
            if fields[2] == 'exon':
                gene_id = fields[8].split(';')[0].split('"')[1]
                if gene_id not in exon_counts:
                    exon_counts[gene_id] = 1
                else:
                    exon_counts[gene_id] += 1

    return exon_counts