# %%
import scipy.stats as ss
import numpy as np
import sympy as sp
from sympy.stats import Uniform, Normal, Exponential, E, variance, given, P, Bernoulli, random_symbols
import empirical_sobol as es
import matplotlib.pyplot as plt

# %%
class Ishigami:
    def __init__(self, a=7, b=0.1):
        self.name='Ishigami'
        self.a = a
        self.b = b

        self.X1 = Uniform('X1', -sp.pi, sp.pi)
        self.X2 = Uniform('X2', -sp.pi, sp.pi)
        self.X3 = Uniform('X3', -sp.pi, sp.pi)
        self.f = sp.sin(self.X1) + self.a * sp.sin(self.X2)**2 + self.b*self.X3**4 * sp.sin(self.X1)

        self.define_exact_statistics()

        self.rv = ss.uniform(loc=-np.pi, scale=2*np.pi)

        self.PA = lambda xlo, xhi: self.rv.cdf(xhi) - self.rv.cdf(xlo)

    def define_exact_statistics(self):

        self.V1 = variance(given(self.f, self.X1))
        self.V2 = variance(given(self.f, self.X2))
        self.V3 = variance(given(self.f, self.X3))

        self.V1fun = sp.lambdify(self.X1, self.V1, 'numpy')
        # Need to modify this one because it returns a single constant value no matter the size of input vector
        self.V2fun = lambda x: np.full(x.shape,sp.lambdify(self.X2, self.V2, 'numpy')(x))
        self.V3fun = sp.lambdify(self.X3, self.V3, 'numpy')
        self.partial_variances = [self.V1fun, self.V2fun, self.V3fun]

        self.EV1 = float(E(self.V1.replace(self.X1.symbol, self.X1)))
        self.EV2 = float(E(self.V2.replace(self.X2.symbol, self.X2)))
        self.EV3 = float(E(self.V3.replace(self.X3.symbol, self.X3)))
        self.EVs = [self.EV1, self.EV2, self.EV3]

        self.mean = self.a/2
        self.var = self.a**2 / 8 + self.b*(np.pi**4)/5+self.b**2*np.pi**8/18 + 0.5

        self.S1 = 0.5 * (1+self.b*np.pi**4/5)**2 / self.var
        self.S2 = self.a**2 / 8 / self.var
        self.S3 = 0

        self.main_effects = self.S = np.array([self.S1, self.S2, self.S3])

    def generate_samples(self, N):
        return self.rv.rvs((N,3))
    
    def evaluate(self,x):
        # assumed x shape = (N, 3)
        x = np.atleast_2d(x)

        return np.sin(x[:,0]) + self.a*np.power(np.sin(x[:,1]),2) + self.b*np.power(x[:,2],4.0) * np.sin(x[:,0])
    
    def eval_scipy(self,x):
        # returns in the right shape for scipy sobol function
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.sobol_indices.html
        # with x of shape (d, n) and output of shape (s, n) where: 
        # d is the input dimensionality of func (number of input variables),
        # s is the output dimensionality of func (number of output variables), and
        # n is the number of samples
        N = x.shape[1]
        tmp = self.evaluate(x.T)
        return np.atleast_2d(tmp)

# %%

class polynomialSobols:
    def __init__(self, a=1, b=1, c=1, dist_type='uniform'):
        """
        Implements Sobol' index calculation, both empirically and analytically, 
        for the function
        f(X1, X2) = aX1 + bX2^2 + cX1X2.

        For uniform and exponential, a good combination of parameters
        is (1,1,10). For normal it's (1,1,1). This leads to nonnegligible 
        Sobol' indices while still having interactions.

        When computing things numerically or sampling inputs, the class 
        also samples a variable X3 that is not included in the
        function. This serves as a measure of how well an unimportant
        variable is identified.
        """

        self.a = a
        self.b = b
        self.c = c

        self.asym, self.bsym, self.csym = sp.symbols('a b c')

        dist_options=['uniform', 'normal', 'exponential']
        if not dist_type in dist_options:
            ValueError(f"dist_type must be one of the following: {dist_options}")
        
        self.dist_type = dist_type
        self.name = f'{dist_type}Polynomial'
        if dist_type=='uniform':
            self.X1 = Uniform('X1', 0, 1)
            self.X2 = Uniform('X2', 0, 1)
            self.X3 = Uniform('X3', 0, 1)
            self.rv = ss.uniform()
        if dist_type=='normal':
            self.X1 = Normal('X1', 0, 1)
            self.X2 = Normal('X2', 0, 1)
            self.X3 = Normal('X3', 0, 1)
            self.rv = ss.norm()
        if dist_type=='exponential':
            self.X1 = Exponential('X1', 1)
            self.X2 = Exponential('X2', 1)
            self.X3 = Exponential('X3', 1)
            self.rv = ss.expon()

        self.PA = lambda xlo, xhi: self.rv.cdf(xhi) - self.rv.cdf(xlo)

        # Define the function f = a X1 + b X2^2 + c X1 X2
        self.f = self.asym*self.X1 + self.bsym*self.X2**2 + self.csym*self.X1*self.X2
        self.V = variance(self.f)

        self.V1 = variance(given(self.f, self.X1))
        self.V2 = variance(given(self.f, self.X2))
        self.V3 = variance(given(self.f, self.X3))

        self.E1 = E(given(self.f, self.X1))
        self.E2 = E(given(self.f, self.X2))
        self.E3 = E(given(self.f, self.X3))
        self.E2fun = sp.lambdify(self.X2, self.subvals(self.E2), 'numpy')

        self.V1fun = sp.lambdify(self.X1, self.subvals(self.V1), 'numpy')
        self.V2fun = sp.lambdify(self.X2, self.subvals(self.V2), 'numpy')
        # Need to modify this one because it returns a single constant 
        # value no matter the size of input vector
        self.V3fun = lambda x: np.full(x.shape,sp.lambdify(self.X3, self.subvals(self.V3), 'numpy')(x))
        self.partial_variances = [self.V1fun, self.V2fun, self.V3fun]

        self.EV1 = float(E(self.subvals(self.V1).replace(self.X1.symbol, self.X1)))
        self.EV2 = float(E(self.subvals(self.V2).replace(self.X2.symbol, self.X2)))
        self.EV3 = float(E(self.subvals(self.V3).replace(self.X3.symbol, self.X3)))
        self.EVs = [self.EV1, self.EV2, self.EV3]

        self.E = E(self.f)
        self.compute_sobol_indices()

    def subvals(self, expr):
        return expr.subs(self.asym,self.a).subs(self.bsym,self.b).subs(self.csym,self.c)
       
    def compute_sobol_indices(self):
        self.compute_main_effects()
        self.compute_total_effects()

    def compute_main_effects(self):
        self.S1 = variance(E(given(self.f, self.X1)).replace(self.X1.symbol, self.X1)) / self.V
        self.S2 = variance(E(given(self.f, self.X2)).replace(self.X2.symbol, self.X2)) / self.V
        self.main_effects_analytical = (self.S1, self.S2, 0) 
        self.main_effects = np.array([float(self.subvals(self.S1)), float(self.subvals(self.S2)), 0])
    
    def compute_total_effects(self):
        self.T1 = E(variance(given(self.f,self.X2)).replace(self.X2.symbol, self.X2)) / self.V
        self.T2 = E(variance(given(self.f,self.X1)).replace(self.X1.symbol, self.X1)) / self.V
        self.total_effects_analytical = (self.T1, self.T2, 0)
        self.total_effects = np.array([float(self.subvals(self.T1)), float(self.subvals(self.T2)), 0])

    def generate_samples(self,N=1):
        return self.rv.rvs((N,3))
    
    def evaluate(self,x):
        # Note we sample X3 but don't use it in the function.
        X1, X2, X3 = x.T
        return self.a*X1 + self.b*X2**2 + self.c*X1*X2
    
    def approximate_main_effects(self, N=100000):
        X = self.generate_samples(N)
        Y = self.evaluate(X)
        return es.empirical_sobol(X,Y)

# %%

class spikeSlabPolynomial:
    """
    This class implements a polynomial of X1 and X2, where X1 and X2 are 
    "spike and slab" random variables, which are defined as
    Z = Bern(p), p = probability Z=1
    X = Z * N(mu, sigma^2)

    The polynomial is the same as before:

    """
    def __init__(self, p=0.5, a=1, b=1, c=1):
        self.name = 'spikeSlabPolynomial'
        self.dist_type='Spike-slab'

        self.p = p
        self.brv = ss.bernoulli(p)
        self.nrv = ss.norm(loc=1)

        self.a=a
        self.b=b
        self.c=c

        self.asym, self.bsym, self.csym, self.X1sym, self.X2sym = sp.symbols('a b c X1 X2')

        self.N1 = Normal('N1', 1, 1)
        self.B1 = Bernoulli('B1', 0.5)
        self.X1 = self.N1 * self.B1

        self.N2 = Normal('N2', 1, 1)
        self.B2 = Bernoulli('B2', 0.5)
        self.X2 = self.N2 * self.B2

        # Define the function f = a X1 + b X2^2 + c X1 X2
        self.f = self.asym*self.X1 + self.bsym*self.X2**2 + self.csym*self.X1*self.X2
        self.V = variance(self.f)

        # # SymPy doesn't like discrete RVs so needed to compute
        # # these conditional variances analytically
        self.V1 = 4 * self.bsym**2 + 3 * self.bsym * self.csym * self.X1sym + 0.75*self.csym**2 * self.X1sym
        self.V2 = 0.75 * (self.asym + self.csym * self.X2sym)**2
        self.V3 = 0

        self.EV1 = E(self.subvals(self.V1).replace(self.X1sym, self.X1))
        self.EV2 = E(self.subvals(self.V2).replace(self.X2sym, self.X2))
        self.EV3 = 0

        self.E1 = (self.asym + 0.5 * self.csym)*self.X1sym + self.bsym
        self.E2 = 0.5 * (self.asym + self.csym * self.X2sym) + self.bsym * self.X2sym**2
        self.VE1 = variance(self.E1.replace(self.X1sym, self.X1))
        self.VE2 = variance(self.E2.replace(self.X2sym, self.X2))

        self.pick_freeze_sobols()

    def plot_pdf(self):
        support = np.linspace(self.nrv.ppf(1e-4), self.nrv.ppf(1-1e-4),1000)
        fig = plt.figure(figsize=(2.5, 1.5))
        ax = fig.add_subplot(111)
        ax.plot(support, self.p*self.nrv.pdf(support), 'k')
        slab_at_0 = self.p*self.nrv.pdf(0)*0.98
        ax.vlines(0, slab_at_0, ax.get_ylim()[1]*2, 'k' )
        ax.spines[['top','left','right']].set_visible(False)
        ax.set_yticks([])
        ax.set_ylabel('Probability\ndensity', rotation=0, ha='right', va='center')
        ax.set_xlabel("Spike-slab random variable")
        ax.set_ylim([0,ax.get_ylim()[1]])
        ax.set_xticks([0,1])

        fig.tight_layout()
        return fig

    def subvals(self, expr):
        return expr.subs(self.asym,self.a).subs(self.bsym,self.b).subs(self.csym,self.c)
       
    def compute_sobol_indices(self):
        # This method is currently not functional because I couldn't get
        # SymPy to play nice with the product variable.
        self.compute_main_effects()
        self.compute_total_effects()

    def compute_main_effects(self):
        self.S1 = self.VE1 / self.V
        self.S2 = self.VE2 / self.V
        self.main_effects_analytical = (self.S1, self.S2, 0) 
        self.main_effects = np.array([float(self.subvals(self.S1)), float(self.subvals(self.S2)), 0])
    
    def compute_total_effects(self):
        self.T1 = self.EV2 / self.V
        self.T2 = self.EV1 / self.V
        self.total_effects_analytical = (self.T1, self.T2, 0)
        self.total_effects = np.array([float(self.subvals(self.T1)), float(self.subvals(self.T2)), 0])

    def generate_samples(self, N=1):
        X = np.zeros((N,3))
        X[:,0] = self.brv.rvs(N) * self.nrv.rvs(N)
        X[:,1] = self.brv.rvs(N) * self.nrv.rvs(N)
        X[:,2] = self.brv.rvs(N) * self.nrv.rvs(N)
        return X
    
    def evaluate(self,x):
        # Note we sample X3 but don't use it in the function.
        X1, X2, X3 = x.T
        return self.a*X1 + self.b*X2**2 + self.c*X1*X2 
    
    def pick_freeze_sobols(self,N=10000000):
        XA = self.generate_samples(N)
        XB = self.generate_samples(N)
        YA = self.evaluate(XA)
        YB = self.evaluate(XB)
        V = 0.5 * (np.var(YA) + np.var(YB))
        Y = np.concatenate((YA,YB))
        mu = np.mean(Y)

        XAb = np.zeros_like(XA)
        YAb = np.zeros(N)
        self.main_effects = np.zeros(3)
        self.total_effects = np.zeros(3)
        for i in range(3):
            XAb = XA.copy()
            XAb[:,i] = XB[:,i]
            YAb = self.evaluate(XAb)
            Vi = np.mean((YB-mu)*(YAb - YA))
            self.main_effects[i] = Vi/V
            self.total_effects[i] = 1/(2*N) * np.sum( (YA-YAb)**2. ) / V

# %%
class SobolG:
    """
    Implementing the Sobol' G function discussed in [1,2]

    G = prod_i=1^p g_i, 

    g_i = ( |4 x_i -2| + a_i ) / (1 + a_i )
    
    Let mu'_2,i is the 2nd raw moment of g_i: 
        mu'_2,i = 1 + 1/3*(a_i+1)^(-2) 

    Then 
        Vu = ( prod_(i in u) mu'_2,i ) - 1
        V = ( prod_(i=1)^p mu'_2,i ) - 1
    and
    Su = Vu / V
    Tu = 1 - Vuc / V, where uc = i not in u

    a >= 9 => not important
    a ~ 0 => very important
    multiple a ~ 0 => interactions important

    References:
    [1] Sobol′, I.M. 2001. “Global Sensitivity Indices for Nonlinear Mathematical Models 
        and Their Monte Carlo Estimates.” The Second IMACS Seminar on Monte Carlo Methods 
        55 (1): 271–80. https://doi.org/10.1016/S0378-4754(00)00270-6.
    [2] Saltelli, Andrea, et al. “Variance Based Sensitivity Analysis of Model Output. 
        Design and Estimator for the Total Sensitivity Index.” Computer Physics Communications, 
        vol. 181, no. 2, Feb. 2010, pp. 259–70, https://doi.org/10.1016/j.cpc.2009.09.018.
        """
    def __init__(self, n_params, a=None ):
        self.n_params = n_params
        self.a = np.linspace(0, 10, n_params) if a is None else a

        self.rv = ss.uniform()
        
        # The 2nd raw moment of g (lowercase intentional) function computed element-wise
        self.muprime2 = 1. + (self.a + 1.)**(-2.)/3.
        self.V = np.prod( self.muprime2 )-1

        self.get_analytical_indices()

    def get_analytical_indices(self):
        self.main_effects = (self.muprime2-1)/ self.V
        self.total_effects = np.zeros_like(self.main_effects)
        for i in range(self.n_params):
            self.total_effects[i] = 1 - (np.prod(np.delete(self.muprime2,i))-1) / self.V

    def generate_samples(self, N=1):
        return self.rv.rvs((N,self.n_params))
    
    def evaluate(self,x):
        # Applies the g function 
        #   prod_i g_i = ( |4 x_i -2| + a_i ) / (1 + a_i )
        # elementwise to the vector x
        return np.prod( ( np.abs( 4.* x - 2. ) + self.a ) / (1. + self.a ), axis=1 )

# %%
class exponentialFunction:
    """
    This class implements an exponential of X1 and X2, where X1 and X2 are 
    uniform random variables. 
    
    It is basically the polynomial except that is now the argument of an exponential:
        f(X1,X2) = np.exp(a X1 + b X2^2 + c X1 X2)

    """
    def __init__(self, p=0.5, a=1, b=1, c=1):
        self.name = 'exponentialFunction'
        self.rv = ss.uniform()

        self.a=a
        self.b=b
        self.c=c

        self.pick_freeze_sobols()

    def generate_samples(self, N=1):
        return self.rv.rvs(size=(N,3))
    
    def evaluate(self,x):
        # Note we sample X3 but don't use it in the function.
        X1, X2, X3 = x.T
        return np.exp(self.a*X1 + self.b*X2**2 + self.c*X1*X2)
    
    def pick_freeze_sobols(self,N=10000000):
        XA = self.generate_samples(N)
        XB = self.generate_samples(N)
        YA = self.evaluate(XA)
        YB = self.evaluate(XB)
        V = 0.5 * (np.var(YA) + np.var(YB))
        Y = np.concatenate((YA,YB))
        mu = np.mean(Y)

        XAb = np.zeros_like(XA)
        YAb = np.zeros(N)
        self.main_effects = np.zeros(3)
        self.total_effects = np.zeros(3)
        for i in range(3):
            XAb = XA.copy()
            XAb[:,i] = XB[:,i]
            YAb = self.evaluate(XAb)
            Vi = np.mean((YB-mu)*(YAb - YA))
            self.main_effects[i] = Vi/V
            self.total_effects[i] = 1/(2*N) * np.sum( (YA-YAb)**2. ) / V
