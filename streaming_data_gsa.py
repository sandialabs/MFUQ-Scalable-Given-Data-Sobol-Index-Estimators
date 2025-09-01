import numpy as np
from scipy.stats import binned_statistic
from scipy.stats import gaussian_kde
try:
    from scipy.integrate import cumtrapz
except:
    from scipy.integrate import cumulative_trapezoid as cumtrapz
import scipy.stats as ss
import h5py
from scipy.interpolate import interp1d

def edf_scale(X):
    """
    Map data into U[0,1] via the empirical density

    This is pulled from Justin Winokur's Empirical Sobol implementation,
    but I am assuming samples are passed in as the transpose of his. 
    """
    # Modified Bert D. 10/19/23 to use ordinal method instead of max
    # for handling duplicate entries in data set. Using ordinal
    # keeps the scipy.stats.binned_statistic function from crashing
    # when there are many duplicate values. However, it may create a result
    # that depends on the ordering of the data set. So only use this 
    # when the inputs with many duplicate values are not important.
    # U = ss.rankdata(X, method="max", axis=0)
    U = ss.rankdata(X, method="ordinal", axis=0) 
    Umn = U.min(axis=0)
    Umx = U.max(axis=0)
    U = (U - Umn) / (Umx - Umn)
    return U

#%%
class StreamingDataGSABatched:
    """Class object to manage data to compute Sobol' indices 
    from streaming data. This class is a modification of the 
    previous, which processed samples one at a time. This class
    instead will enable batches of samples to be passed in at once.

    Attributes:

    """

    def __init__(self, n_x, n_y, n_bins, n_samp_start=500, bin_edges=None, binning_method='kde', bin_to_inf=True):
        """Initialize class attributes

        Args:
            n_x: The number of input dimensions, integer scalar
            n_y: The number of function outputs, integer scalar
            n_bins: The number of bins to use in the binned statistics for each input, integer scalar
            n_samp_start (integer scalar, optional): If bin_edges is set to None, then the first n_samp_start samples are used
                to estimate the bin edges for the inputs
            bin_edges (np.ndarray, optional) If not None, provides the bin_edges for all inputs. Dimension [n_x, n_bins+1]. If not
                provided, then the bin_edges are estimated from the first n_samp_start samples
            binning_method: The method used to compute the bin edges. See options below.
            bin_to_inf: whether to set bin edges to -inf, inf.
        """

        # TODO: add validity check for these options 
        self.n_x = n_x # number of input dimensions
        self.n_y = n_y # number of output values for each function evaluation
        self.n_bins = n_bins # number of bins to gather statistics in

        if not isinstance(bin_edges, np.ndarray):
            self.n_samp_start  = n_samp_start # number of samples to estimate input distribution bin edges
            self.bin_edges     = None
            self.bin_breakpts  = None
            self.sample_x_buffer = np.zeros((self.n_samp_start,self.n_x)) # Buffer to hold x samples until we have enough to estimate input histogram bin edges
            self.sample_y_buffer = np.zeros((self.n_samp_start,self.n_y)) # Buffer to hold y samples until we have enough to estimate input histogram bin edges
        else:
            # check dimension of passed-in edges array
            if bin_edges.shape == (self.n_x,self.n_bins+1):
                self.bin_edges    = bin_edges
                self.bin_breakpts = self.bin_edges[:,1:-1]
            else:
                raise ValueError(f"The passed-in bin_edges array has the wrong size: {bin_edges.shape} instead of {(self.n_x,self.n_bins+1)}")

        binning_methods = ['kde', 'histogram', 'ecdf', 'quantile', 'dx']    
        if not binning_method in binning_methods:
            raise ValueError(f"{binning_method} is not a valid option for binning_method. You may select "\
                             ", ".join(binning_methods[:-1])+"or "+binning_methods[-1] )
        self.binning_method = binning_method
        self.bin_to_inf = bin_to_inf

        self.sample_count  = 0 # number of samples processed so far
        
        self.bin_counts    = np.zeros((self.n_x,self.n_bins),dtype=int) # sample counts in each bin for each input
        self.bin_y_stats   = np.zeros((self.n_x,self.n_bins,self.n_y)) # stats on y in each bin
        self.bin_M2_stats  = np.zeros((self.n_x, self.n_bins, self.n_y)) # unnormalized variance of y
        self.glob_y_stats  = np.zeros((self.n_y)) # stats on y globally
        self.glob_M2_stats = np.zeros((self.n_y)) # stats on unnormalized y variance globally
        return
    
    def estimate_bin_edges(self):
        """
        Based on initial samples, estimate the bin edges for each input so that 
        each bin has equal probability

        This function is meant to only be called internally.

        Inputs:
            method ['kde', 'histogram']: determines the method used to approximate the bins.
        """

        # Initialize
        self.bin_edges = np.zeros((self.n_x,self.n_bins+1)) # bin edges in histograms for each input

        for x_idx, x_vals in enumerate(self.sample_x_buffer.T):
            if self.binning_method=='kde':
                self.estimate_bin_edges_from_kde(x_idx, x_vals)
            if self.binning_method=='histogram':
                self.estimate_bin_edges_from_histogram(x_idx, x_vals)
            if self.binning_method=='ecdf':
                self.estimate_bin_edges_from_ecdf(x_idx, x_vals)
            if self.binning_method=='quantile':
                self.estimate_bin_edges_from_quantile(x_idx, x_vals)
            if self.binning_method=='dx':
                self.estimate_equidistant_bin_edges(x_idx, x_vals)
            
            if self.bin_to_inf:
                self.bin_edges[x_idx,0] = -np.inf
                self.bin_edges[x_idx,-1] = np.inf

        # Set bin breakpoints
        self.bin_breakpts = self.bin_edges[:,1:-1]

        return
    
    def estimate_bin_edges_from_kde(self, x_idx, x_vals):
        # Courtesy of ChatGPT 40-mini (12/9/24)
            # Step 1: Estimate the PDF using Kernel Density Estimation
            kde = gaussian_kde(x_vals)

            # Create a range of values for which to evaluate the PDF
            x = np.linspace(min(x_vals), max(x_vals), 1000)
            pdf = kde(x)

            # Step 2: Calculate the CDF by integrating the PDF
            cdf = cumtrapz(pdf, x, initial=0)

            # Normalize the CDF to ensure it goes from 0 to 1
            cdf /= cdf[-1]

            # Step 3: Determine the bin edges by inverting the CDF for internal bins
            for bin_idx in range(1,self.n_bins):
                # Calculate the cumulative probability for the bin edge
                cumulative_probability = float(bin_idx) / float(self.n_bins)

                # Find the corresponding x value for the cumulative probability
                # Use interpolation to find the x value that corresponds to the cumulative probability
                bin_edge = np.interp(cumulative_probability, cdf, x)

                # Store the bin edge
                self.bin_edges[x_idx,bin_idx] = bin_edge
            
            # Step 4: avoid any values below or above the first and last bin
            # so that binned_statistics never returns a bin number of 0 or n_bins+1
            self.bin_edges[x_idx,0] = x_vals.min()
            self.bin_edges[x_idx,self.n_bins] = x_vals.max() 

    def estimate_bin_edges_from_histogram(self, x_idx, x_vals):      
        U = edf_scale(x_vals)
        mins = binned_statistic(U,x_vals,statistic='min', bins=self.n_bins)[0]
        maxs = binned_statistic(U,x_vals, statistic='max', bins=self.n_bins)[0]
        self.bin_edges[x_idx,1:-1] = 0.5 * (mins[1:] + maxs[:-1])
        self.bin_edges[x_idx,0] = mins[0]
        self.bin_edges[x_idx,-1] = maxs[-1]

    def estimate_bin_edges_from_ecdf(self, x_idx, x_vals):
        U = edf_scale(x_vals)
        f = interp1d(U,x_vals)
        prob_levels = np.linspace(0,1,self.n_bins+1)
        self.bin_edges[x_idx] = f(prob_levels)

    def estimate_bin_edges_from_quantile(self, x_idx, x_vals):
        prob_levels = np.linspace(0,1,self.n_bins+1)
        self.bin_edges[x_idx] = np.quantile(x_vals, prob_levels, method='interpolated_inverted_cdf')

    def estimate_equidistant_bin_edges(self, x_idx, x_vals):
        # TODO: for unbounded, may need update this to truncate the domain.
        self.bin_edges[x_idx] = np.linspace(x_vals.min(), x_vals.max(), self.n_bins+1)

    def process_samples(self, x_samples, y_samples, scipy_flag=True):
        """Update statistics using new samples

        Args:
            x_samples (vector or array) : N samples of all n_x inputs
            y_samples_in (vector or array: N function values corresponding to x_sample (length n_y)
            scipy_flag (bool) : If True, uses scipy binned stats to update binned statistics; 
                                if False, loops over bins.
        """
        x_samples, y_samples = self.prep_samples(x_samples, y_samples)

        # Check to see if we have estimates for the bins on the inputs yet; if not, 
        # input distributions have not been binned.
        if not isinstance(self.bin_edges, np.ndarray):
            
            x_overflow, y_overflow = self.augment_sample_buffer(x_samples, y_samples)

            # Check if we have enough samples to estimate input bin edges
            if self.sample_count >= self.n_samp_start:

                # Estimate bin edges
                self.estimate_bin_edges()

                self.instantiate_binned_stats()

                self.instantiate_global_stats()

                # If there were overflow samples after getting to n_sample_start,
                # call the method again to use them in a streaming update.
                if not x_overflow is None:
                    self.process_samples(x_overflow, y_overflow, scipy_flag)
        else:

            # Updating bin statistics for each input
            for x_idx, x_vals in enumerate(x_samples.T):
                if scipy_flag:
                    self.update_binned_stats_w_scipy(x_idx, x_vals, y_samples)
                else:
                    bin_idxs = self.get_bin_indices(x_idx, x_vals)
                    self.update_binned_stats(x_idx, y_samples, bin_idxs)

            self.update_global_stats(y_samples)

    def prep_samples(self, x_sample, y_sample):
        # Ensures the samples passed in are returned as 2D arrays 
        # with shape [Nsamples, Nx] or [Nsamples, Ny]

        y_sample = np.array(y_sample)

        # Since we're doing this reshaping I think we will get an error if
        # someone doesn't pass in samples of the correct shape.    
        if len(x_sample.shape)==1: # Make sure x_sample is an array Nsamples x Nx
            x_sample = np.reshape(x_sample, (1,self.n_x))
        # Make sure y_sample is an array Nsamples x Ny
        y_sample = np.reshape(y_sample, (x_sample.shape[0], self.n_y) )
    
        return x_sample, y_sample

    def get_bin_indices(self, x_idx, x_vals):
        # Find the bin that x_val falls in. Switched to digitize from bisect
        # because there didn't seem to be a major timing difference.
        return np.digitize(x_vals, bins=self.bin_breakpts[x_idx,:], right=False)
    
    def update_binned_stats_w_scipy(self, x_idx, x_samples, y_samples):
        """ 
        Assumes output samples are shaped like N_samples x N_outputs.

        Using a batch streaming update formula, where nA was the sample size before
        the update and nB is the sample size of the batch.
        The formula doesn't work well if nA ~ nB and both are large. 
        Since we're streaming in data in small chunks relative to total sample size, 
        I don't anticipate this being an issue in general. 
        However, if we're concerned we could add some sort of check.

        nAB = nA + nB
        delta = <x>B - <x>A
        <x>AB = <x>A + delta*nB/nAB

        M2AB = M2A + M2B + delta^2 * nA * nB / nAB

        To compute variance, M2 must be divided by nAB-1. 
        """
        # ==========================================
        # Updating the binned mean
        # ==========================================
        ym_work,  _, bin_nos = binned_statistic(x_samples, y_samples.T, statistic="mean", bins=self.bin_edges[x_idx])
        bin_idxs, bin_counts = np.unique(bin_nos-1,return_counts=True)

        # Reshaping to enable bin-wise computation after this.
        ym_work = ym_work[:,bin_idxs].T.reshape((bin_idxs.size, self.n_y))

        # Get the delta only for the bins with new samples
        delta = ym_work - self.bin_y_stats[x_idx,bin_idxs]
        
        # Get vectorized nB/nAB, or ratio of batch sample size to new total
        ratio = bin_counts / (self.bin_counts[x_idx,bin_idxs]+bin_counts)

        # <x>AB = <x>A + delta*ratio 
        self.bin_y_stats[x_idx, bin_idxs] += delta * ratio[:,np.newaxis]

        # ==========================================
        # Updating the binned unnormalized variance
        # ==========================================
        s = binned_statistic(x_samples, y_samples.T, statistic='std', bins=self.bin_edges[x_idx])[0]

        # Get unnormalized variance from binned std dev statistic
        M2_work = ( s[:,bin_idxs]**2 * bin_counts ).T.reshape((bin_idxs.size, self.n_y))
         
        # ratio = nA * nB / nAB 
        # (i.e., old sample size x batch sample size) / new total sample size
        ratio *= self.bin_counts[x_idx,bin_idxs]

        # M2AB = M2A + M2B + delta^2 * ratio
        self.bin_M2_stats[x_idx, bin_idxs] += M2_work + delta**2 * ratio[:,np.newaxis]

        # Increment bin counts after the fact to make previous math easier.
        self.bin_counts[x_idx, bin_idxs] += bin_counts

    def update_binned_stats(self, x_idx, y_sample, bin_idxs):
        """ 
        Assumes output samples are shaped like N_samples x N_outputs.

        Using a batch streaming update formula, where nA was the sample size before
        the update and nB is the sample size of the batch.
        The formula doesn't work well if nA ~ nB and both are large. 
        Since we're streaming in data in small chunks relative to total sample size, 
        I don't anticipate this being an issue in general. 
        However, if we're concerned we could add some sort of check.

        nAB = nA + nB
        delta = <x>B - <x>A
        <x>AB = <x>A + delta*nB/nAB

        M2AB = M2A + M2B + delta^2 * nA * nB / nAB

        To compute variance, M2 must be divided by nAB-1. 
        """

        bins_with_new_samples, new_counts = np.unique(bin_idxs, return_counts=True)
        old_counts = self.bin_counts[x_idx, bins_with_new_samples]
        self.bin_counts[x_idx,bins_with_new_samples] += new_counts
        new_total_counts = self.bin_counts[x_idx, bins_with_new_samples]

        # Only loop over the bins that have new samples in them.
        for update_bin_idx, nA, nB, nAB in zip(bins_with_new_samples, 
                                        old_counts, new_counts, new_total_counts):
            
            y_bin = y_sample[update_bin_idx==bin_idxs]

            new_y_mean = np.mean(y_bin, axis=0)
            delta = new_y_mean - self.bin_y_stats[x_idx,update_bin_idx,:] 
            self.bin_y_stats[x_idx,update_bin_idx,:] += delta * nB / nAB

            new_M2 = np.var(y_bin, axis=0) * nB
            self.bin_M2_stats[x_idx, update_bin_idx] += new_M2 + delta**2 * nA * nB / nAB

    def update_global_stats(self, y_samples):
        """
        Updating global y statistics with the following batch update formulas, where
        nA is the previous total sample count and nB is the sample size of the batch.

        nAB = nA + nB
        delta = <x>B - <x>A
        <x>AB = <x>A + delta*nB/nAB

        M2AB = M2A + M2B + delta^2 * nA * nB / nAB

        To compute variance, M2 must be divided by nAB-1. 
        """
        # Statistics over all y values
        nA = self.sample_count
        nB = y_samples.shape[0]

        # Incrementing the sample count
        self.sample_count += nB

        nAB = self.sample_count 

        new_y_mean = np.mean(y_samples, axis=0)
        delta = new_y_mean - self.glob_y_stats
        self.glob_y_stats += delta * nB / nAB

        #Update unnormalized variance
        new_M2 = np.var(y_samples, axis=0) * nB
        self.glob_M2_stats += new_M2 + delta**2 * nA * nB / nAB


    def augment_sample_buffer(self, x_samples, y_samples):
        # Add samples to our buffer

        # We increment sample_count within this method while we are 
        # filling the sample buffer to initialize bin edges and binned
        # statistics.

        curr_sample_size = y_samples.shape[0]
        # If we are still below the target sample count for computing
        # bin edges and instantiating the statistics, proceed as usual.
        if self.sample_count + curr_sample_size <= self.n_samp_start:
            start_ind = self.sample_count
            end_ind = self.sample_count + curr_sample_size
            start_ind = end_ind - curr_sample_size
            self.sample_x_buffer[start_ind:end_ind,:] = x_samples
            self.sample_y_buffer[start_ind:end_ind,:] = y_samples

            self.sample_count += y_samples.shape[0]
            return None, None
        # Otherwise, need to only include a portion of these samples
        # in the buffers. 
        else:
            n_overflow = self.sample_count + curr_sample_size - self.n_samp_start
            self.sample_x_buffer[self.sample_count:,:] = x_samples[:-n_overflow]
            self.sample_y_buffer[self.sample_count:,:] = y_samples[:-n_overflow]

            self.sample_count += curr_sample_size-n_overflow

            # Return the overflow samples for the streaming update.
            return x_samples[-n_overflow:], y_samples[-n_overflow:]

    def instantiate_binned_stats(self):
        
        # Loop over input variables to instantiate bins.
        for x_idx, x_vals in enumerate(self.sample_x_buffer.T):
            self.update_binned_stats_w_scipy(x_idx, x_vals, self.sample_y_buffer)

    def instantiate_global_stats(self):
        # Set current average of y-values
        self.glob_y_stats  = np.mean(self.sample_y_buffer,    axis=0)

        # Set unnormalized variance.
        self.glob_M2_stats = np.var(self.sample_y_buffer, axis=0) * self.sample_y_buffer.shape[0] 

    def write_checkpoint(self,file_name="CheckPointFile.h5"):
        """Write checkpoint data to hdf5 file format"""

        if self.bin_edges.all() == None:
            print("Checkpointing for now can only handle case where bin edges are already available")
            print("No checkpoint file has been written.")
            return
    
        # Write arrays to an HDF5 file
        with h5py.File(file_name, 'w') as f:
            f.create_dataset('n_x'          , data=self.n_x)
            f.create_dataset('n_y'          , data=self.n_y)
            f.create_dataset('n_bins'       , data=self.n_bins)
            f.create_dataset('sample_count' , data=self.sample_count)
            f.create_dataset('bin_edges'    , data=self.bin_edges)
            f.create_dataset('bin_counts'   , data=self.bin_counts)
            f.create_dataset('bin_y_stats'  , data=self.bin_y_stats)
            f.create_dataset('bin_M2_stats', data=self.bin_M2_stats)
            f.create_dataset('glob_y_stats' , data=self.glob_y_stats)
            f.create_dataset('glob_M2_stats', data=self.glob_M2_stats)
        return
    
    def load_checkpoint(self,file_name="CheckPointFile.h5"):
        """Read checkpoint data from hdf5 file"""
    
        # Read arrays from the HDF5 file
        with h5py.File(file_name, 'r') as f:
            self.n_x           = f['n_x'][()]
            self.n_y           = f['n_y'][()]
            self.n_bins        = f['n_bins'][()]
            self.sample_count  = f['sample_count'][()]
            self.bin_edges     = f['bin_edges'][:]
            self.bin_counts    = f['bin_counts'][:]
            self.bin_y_stats   = f['bin_y_stats'][:]
            self.bin_M2_stats  = f['bin_M2_stats'][:]
            self.glob_y_stats  = f['glob_y_stats'][:]
            self.glob_M2_stats = f['glob_M2_stats'][:]

        # Derived quantities
        self.bin_breakpts = self.bin_edges[:,1:-1]

        return
    
    def finalize_statistics(self):
        """Normalizes the variance
        """
        if not hasattr(self, 'glob_var'):
            self.glob_var = self.glob_M2_stats / (self.sample_count-1)

            self.bin_var = np.zeros_like(self.bin_M2_stats)
            # Filter out any bins with no more than 1 sample; these have zero variance.
            self.bin_var[self.bin_counts>1] = self.bin_M2_stats[self.bin_counts>1] / (self.bin_counts[self.bin_counts>1] - 1)[:,np.newaxis]

    def get_sobol_indices(self,fix_neg=False, normalize=True):
        """Get Sobol' indices based on current statistics
        
        Based Justin Winokur's original "empirical_sobol" code provided in this
        repository.

        Inputs:
            fix_neg: boolean, optional: if False, small Sobol' indices may be negative due to sampling noise.
                If this flag is set to True, it will set those indices to 0.
            normalize: boolean, optional: Normalize the contribution by the total variance. This is the typical
            definition

        Returns:
            S: (n_x,) or (n_x,n_y) array of Sobol' indices
        """
        
        self.finalize_statistics()
        # Sobol index computation using data from running averages
        S = np.zeros((self.n_x, self.n_y)) # To hold Sobol' indices
        VV = self.glob_var if normalize else np.ones(self.n_y)  # Normalize
        weights = self.bin_counts / self.sample_count

        # Loop over all input dimensions
        for x_idx in range(self.n_x):      
            S[x_idx] = (self.glob_var - np.average(self.bin_var[x_idx], weights=weights[x_idx], axis=0)) / VV

        if fix_neg:
            S[S < 0] = 0
        
        return np.squeeze(S)
