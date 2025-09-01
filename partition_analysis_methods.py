import scipy.integrate as si
import scipy.stats as ss
import numpy as np
import matplotlib.pyplot as plt
import streaming_data_gsa as sdg
import empirical_sobol as es

def plot_evf(ax, model, x_idx, bin_edges, plot_mode):
    if plot_mode=='cumulative':
        ax.vlines(model.EVs[x_idx], 0, ax.get_ylim()[1], color='C0') 
    else:
        bin_centers = 0.5 * (bin_edges[1:]+bin_edges[:-1])
        nbins = bin_edges.size-1
        EVA = np.zeros(nbins)
        PA = np.zeros(nbins)
        for k, (xlo, xhi) in enumerate(zip(bin_edges[:-1], bin_edges[1:])):
            EVA[k] = model.rv.expect(model.partial_variances[x_idx], lb=xlo, ub=xhi, conditional=True)
            PA[k] = model.PA(xlo,xhi)
    
        if plot_mode=='times_probability':
            ax.plot(bin_centers, EVA*PA, '.-')
        elif plot_mode=='inner_statistic_only':
            ax.plot(bin_centers, EVA, '.-')
        else: # bin_probabilities
            ax.plot(bin_centers, PA, '.-')

        ax.set_xlim([bin_edges[0],bin_edges[-1]])
    
def plot_replicate_binned_stat_specified_bins(ax, model, x_idx, nsamples, nreps, bin_edges, seed, plot_mode):

    np.random.seed(seed)
    nbins=bin_edges.size-1

    bin_v = np.zeros((nreps,nbins))
    bin_PA = np.zeros((nreps,nbins))
    bin_cVP = np.zeros((nreps, nbins))
    for k in range(nreps):
        x = model.generate_samples(nsamples)
        y = model.evaluate(x)
        xi = x[:,x_idx]
        bin_v[k], bin_PA[k], bin_cVP[k] = compute_binned_stat_specified_bins(xi, y, bin_edges)

    ax.set_axisbelow(True)
    ax.set_xlabel(rf'$X_{{{x_idx+1}}}$')
    ax.spines[['top','right']].set_visible(False)

    if plot_mode=='cumulative':
        ax.hist(bin_cVP[:,-1], histtype='step', color='C1');
    else:    
        bin_centers = 0.5 * (bin_edges[1:]+bin_edges[:-1])

        if plot_mode=='times_probability':
            temp = bin_v * bin_PA
        elif plot_mode=='inner_statistic_only':
            temp = bin_v
        else: # bin_probabilities
            temp = bin_PA
        q5, q95 = np.quantile(temp, (0.05,0.95), axis=0)
        mean = np.mean(temp, axis=0)
        ax.fill_between(bin_centers, q5, q95, color='C1', alpha=0.3)
        ax.plot(bin_centers, mean, 'C1.-')

        ax.set_xticks(bin_edges)
        ax.set_xticklabels([])
        ax.grid(axis='x')
        ax.set_axisbelow(True)

def compute_binned_stat_specified_bins(xi, y, bin_edges):
    nbins=bin_edges.size-1    
    bin_s = ss.binned_statistic(xi, y, 'std', bin_edges)[0]
    counts = np.histogram(xi, bins=bin_edges)[0]
    bin_v = np.zeros(nbins)
    bin_v[counts>1] = bin_s[counts>1]**2 * counts[counts>1] / (counts[counts>1] - 1) 
    bin_PA = counts / y.size

    return bin_v, bin_PA, np.cumsum(bin_v*bin_PA)
