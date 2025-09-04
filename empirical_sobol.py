import numpy as np
import scipy.stats

#=========================================
# Continuous empirical Sobol' index 
# implementation, author: Justin Winokur
#=========================================
def empirical_main_sobol(X, F, mode="B", nbins=None, fix_neg=False, normalize=True):
    """
    Sobol' empirical main effects only but also optimized for multiple responses

    Compute the estimated empirical Main Sobol' Sensitivity Index
    based on a random sample `X` (N,ndim) and function values `F` (N,). Also accepts
    multiple function values in the same (N,Nr).

    Based on the method in [1] with minor modification.
    Inputs:
    -------
    X
        (N,ndim) input sample

    F
        (N,) or (N,nr) function values at X

    mode ['B']
        Mode A: Directly compute V(E(f|x_d))
        Mode B: Use the Law of Total Variance to compute
                V(E(f|d_x)) = V(f) - E(V(f|x_d))
        According to [1] mode B is more accurate but less robust to
        insufficient data since the variance computation will be zero
        without enough data. Mode A is more robust to that but less accurate.

        Mode B is the default

    nbins [ floor(sqrt(N)) ]
        Specify the number of bins

    fix_neg [True]
        Small sensitivities with mode B may return negative. If
        True, will bound all results to be >= 0. Mode A will never be negative

    normalize [True]
        Normalize the contribution by the total variance. This is the typical
        definition.

        Note: If set to False, the units are those of [F]^2. For example if W is a
        measure of length, the units will be length^2. Do not just take the sqrt as the
        sum of squares is not the square of the sum!

    Returns:
    --------
        S: (ndim,) or (ndim,Nr) array of sensitivity values

    A note on precision:
    ---------------------
    While the methods used here are correct and convergent, when used with small sample
    sizes, numerical noise may be present and break some properties. Notably that

            0 <= S_i <= \sum_i S_i <= 1

    Mode "B" can yield negative values (which should just be interpreted as low) and
    the sum of the main effects can be greater than unity due to precision.

    In practical use, this is not a problem because the goals of sensitivity analysis
    do not require a high degree of precision (e.g. An Sx=0.5 vs Sx=0.3 still indicate
    that x is an important paramater) but the user should be aware of this.


    Algorithm:
    ----------
    Compute the Main Sobol' Index:

        S_i = V(E(f|x_i)) / V(f)

    First, compute V(f) as the discrete variance of all samples.

    Second, use the empirical distribution function (EDF) to convert from an arbitrary,
    unknown univariate distribution to one that is identically U[0,1]. (Note: this is
    like an empirical version of Sklar's theorem and copula transformations. It
    affects only the _marginal_ distributions, not the coupling)

    Third, compute E(f|x_i) as a binned statistic of the average (Mode A) or V(f|x_i)
    as a binned statistic variance (Mode B).

    Finally, compute V(E(f|x_i)) as the discrete variance of the binned average (mode A)
    or E(V(f|x_i)) as the discrete average of the binned variance.

        Mode A: S_i = V(E(f|x_i)) / V(f)
        Mode B: S_i = 1 - E(V(f|x_i)) / V(f)

    References:
    -----------

    [1] C. Li and S. Mahadevan. An efficient modularized sample-based method to
        estimate the first-order Sobol’ index. Reliability Engineering & System
        Safety, 153:110–121, 2016.
    """

    X = np.atleast_2d(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, ndim = X.shape

    U = edf_scale(X)

    F = np.c_[F]  # make it (N,Nr) even if Nr == 1
    _N, Nr = F.shape
    if not _N == N:
        raise ValueError("F must be the same length as X")

    nb = nbins if nbins else int(np.sqrt(N))
    V = np.var(F, ddof=1, axis=0)  # (Nr,)
    VV = V if normalize else np.ones(Nr)  # Normalize
    S = np.zeros((ndim, Nr))  # (ndim,Nr)

    for d, u in enumerate(U.T):
        if mode == "A":
            m, _, _ = scipy.stats.binned_statistic(u, F.T, statistic="mean", bins=nb)
            S[d] = np.var(m, axis=1, ddof=1) / VV
        else:
            # Compute counts each time in case of non-square N's or repeated x. Use std
            # since var is not always an option and when it is, it's the wrong ddof
            counts = np.histogram(u, bins=nb)[0]
            s, _, _ = scipy.stats.binned_statistic(u, F.T, statistic="std", bins=nb)
            v = s**2 * counts / (counts - 1)  # Make var and ddof=1
            S[d] = (V - np.mean(v, axis=1)) / VV

    if fix_neg:
        S[S < 0] = 0

    return np.squeeze(S)  # (N,) if Nr == 1 else (N,Nr)


def edf_scale(X):
    """
    Map data into U[0,1] via the empirical density
    """
    U = scipy.stats.rankdata(X, method="max", axis=0)
    Umn = U.min(axis=0)
    Umx = U.max(axis=0)
    U = (U - Umn) / (Umx - Umn)
    return U

# Just an alias for the name
empirical_sobol = empirical_main_sobol

#=========================================
# Discrete empirical Sobol' index 
# implementation, author: Teresa Portone
# Note: this is a shameless adaptation
# of Justin's implementation above.
#=========================================

def empirical_main_sobol_discrete(X, F, mode="B", fix_neg=False, normalize=True):
    """
    Sobol' empirical main effects only but also optimized for multiple responses
    
    Compute the estimated empirical Main Sobol' Sensitivity Index 
    based on a random sample `X` (N,ndim) and function values `F` (N,). Also accepts
    multiple function values in the same (N,Nr).
    
    This method adapts [1] for discrete random variables.

    Inputs:
    -------
    X
        (N,ndim) input sample
    
    F
        (N,) or (N,nr) function values at X
    
    mode ['B']
        Mode A: Directly compute V(E(f|x_d))
        Mode B: Use the law of total variance to compute 
                V(E(f|d_x)) = V(f) - E(V(f|x_d))
        According to [1] mode B is more accurate but less robust to 
        insufficient data since the variance computation will be zero 
        without enough data. Mode A is more robust to that but less accurate. 
        
        Mode B is the default
    
    fix_neg [True]
        Small sensitivities with mode B may return negative. If 
        True, will bound all results to be >= 0. Mode A will never be negative
    
    normalize [True]
        Normalize the contribution by the total variance. This is the typical
        definition. 
        
        Note: If set to False, the units are those of [F]^2. For example if W is a 
        measure of length, the units will be length^2. Do not just take the sqrt as the
        sum of squares is not the square of the sum!
    
    Returns:
    --------
        S: (ndim,) or (ndim,Nr) array of sensitivity values
    
    Algorithm:
    ----------
    Compute the Main Sobol' Index:
    
        S_i = V(E(f|x_i)) / V(f)
        
    First, compute V(f) as the discrete variance of all samples. 

    Second, samples are binned according to the discrete random variable values.

    Third, compute E(f|x_i) as a binned statistic of the average (Mode A) or V(f|x_i) 
    as a binned statistic variance (Mode B).
    
    Finally, compute V(E(f|x_i)) as the discrete variance of the binned average (mode A)
    or E(V(f|x_i)) as the discrete average of the binned variance.
    
        Mode A: S_i = V(E(f|x_i)) / V(f)
        Mode B: S_i = 1 - E(V(f|x_i)) / V(f)

    In this case, what is meant by "discrete variance" and "discrete average" is meant in the 
    discrete random variable sense. That is, for the mean,

    E(Q(x_i)) = sum over x_i discrete values Q(x_i) p(x_i)
    V(Q(x_i)) = sum over x_i discrete values (Q(x_i) - E(Q(x_i)))^2 p(x_i)
    
    References:
    -----------
    
    [1] C. Li and S. Mahadevan. An efficient modularized sample-based method to 
        estimate the first-order Sobol’ index. Reliability Engineering & System 
        Safety, 153:110–121, 2016.
    """

    X = np.atleast_2d(X)
    if X.ndim != 2:
        raise ValueError("X must be 2D")
    N, ndim = X.shape

    F = np.c_[F]  # make it (N,Nr) even if Nr == 1
    _N, Nr = F.shape
    if not _N == N:
        raise ValueError("F must be the same length as X")

    V = np.var(F, ddof=1, axis=0)  # (Nr,)
    VV = V if normalize else np.ones(Nr)  # Normalize
    S = np.zeros((ndim, Nr))  # (ndim,Nr)

    for d, x in enumerate(X.T):
        x_vals, counts = np.unique(x, return_counts=True)
        pmf = counts / np.sum(counts)
        nb = x_vals.size

        if mode == "A":
            print("Warning: method A is not implemented for discrete RVs")
        else:
            v = np.zeros((x_vals.size, F.shape[1]))
            for i, (xv,c) in enumerate(zip(x_vals,counts)):
                if c > 1:
                    # Compute the output variance for all x samples matching the 
                    # current discrete value.
                    v[i] = np.var(F[ x==xv ], axis=0, ddof=1)
            S[d] = (V - np.average(v, weights=pmf, axis=0)) / VV

    if fix_neg:
        S[S < 0] = 0

    return np.squeeze(S)  # (N,) if Nr == 1 else (N,Nr)
