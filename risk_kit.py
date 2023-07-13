import numpy as np
import pandas as pd 
import scipy
from scipy.optimize import minimize
def drawdown(return_series: pd.Series):
    """
    Takes a times series of asset returns
    Computes and returns a DataFrame that contains:
    the wealth index
    the previous peaks
    percent drawdowns
    """

    wealth_index = 1000*(1+return_series).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks)/previous_peaks

    return pd.DataFrame({
        "Wealth": wealth_index,
        "Peaks": previous_peaks,
        "Drawdown": drawdowns
    })

def get_ffme_returns():
    """
    Load the Fama-French Dataset for the returns of the Top and Bottom Deciles by MarketCap
    """
    me_m = pd.read_csv("Data/Portfolios_Formed_on_ME_monthly_EW.csv",
                       header=0, index_col=0, na_values=99.99)
    rets = me_m[['Lo 10', 'Hi 10']]
    rets.columns=["Small Cap", "LargeCap"]
    rets = rets/100
    rets.index = pd.to_datetime(rets.index, format="%Y%m").to_period("M")
    return rets

def get_hfi_returns():
    """
    Load and format the EDHEC Hedge Fund Index Returns
    """

    hfi = pd.read_csv("data/edhec-hedgefundindices.csv",
                      header = 0, index_col=0, parse_dates=True)
    hfi = hfi/100
    hfi.index = hfi.index.to_period("M")
    return hfi

def semideviation(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame
    """
    is_negative = r < 0
    return r[is_negative].std(ddof=0)

def skewness(r):
    """
    Alternative to scipy.stats.skew()
    Computes the skewness of the supplied Series or DataFrame
    Returns a float or a Series
    """

    demeaned_r = r - r.mean()
    #use the population standard deviation , so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**3).mean()
    return exp/sigma_r**3

def kurtosis(r):
    """
    Alternative to scipy.stats.kurtosis()
    Computes the kurtosis of the supplied Series or DataFrame
    Returns a float or a Series
    """

    demeaned_r = r - r.mean()
    #use the population standard deviation , so set dof=0
    sigma_r = r.std(ddof=0)
    exp = (demeaned_r**4).mean()
    return exp/sigma_r**4

def is_normal(r, level=0.01):
    """
    Applies the Jarque-Bera test to determine if a Series is normal or not
    Test is applied at the 1% level by default
    Returns True if the hypothesis of nromality is accepted, False otherwise
    """
    statistic, p_val = scipy.stats.jarque_bera(r)
    return p_val > level

from scipy.stats import norm
def var_gaussian(r, level = 5):
    """
    Retuns the Parametric VaR of a Series or DataFrame
    """

    # Compute the Z score assuming it was Gaussian

    z = norm.ppf(level/100)
    return -(r.mean() + z*r.std(ddof=0))

def var_gaussian(r, level = 5, modified = False):
    """
    Retuns the Parametric  Gaussian VaR of a Series or DataFrame
    If "modified" is True, then the modified VaR is returned,
    using the Cornish-Fisher modificiation
    """

    # Compute the Z score assuming it was Gaussian

    z = norm.ppf(level/100)

    if modified:
        # modify the Z score based on observed skewness and kurtosis
        s = skewness(r)
        k = kurtosis(r)
        z = (z + 
                (z**2 - 1)*s/6 +
                (z**3 -3*z) *(k - 3)/24 -
                (2*z**3 - 5*z)*(s**2)/36
            )
    return -(r.mean() + z*r.std(ddof=0))

def var_historic(r, level = 5):
    """
    VaR Historic
    """
    if isinstance(r, pd.DataFrame):
        return r.aggregate(var_historic, level = level)
    
    elif isinstance(r, pd.Series):
        return -np.percentile(r, level)
    else:
        raise TypeError("Expected r to be Series or DataFrame")
    
def cvar_historic(r, level = 5):
    """
    Computes the Conditional VaR of Series or DataFrame
    """

    if isinstance(r, pd.Series):
        is_beyond = r <= -var_historic(r, level = level)
        return -r[is_beyond].mean()
    elif isinstance(r, pd.DataFrame):
        return r.aggregate(cvar_historic, level = level)
    else: 
        raise TypeError("Expected r to be a Sereis or DataFrame")
    
def semideviation3(r):
    """
    Returns the semideviation aka negative semideviation of r
    r must be a Series or a DataFrame, else raises a TypeError
    """
    excess= r-r.mean()                                        # We demean the returns
    excess_negative = excess[excess<0]                        # We take only the returns below the mean
    excess_negative_square = excess_negative**2               # We square the demeaned returns below the mean
    n_negative = (excess<0).sum()                             # number of returns under the mean
    return (excess_negative_square.sum()/n_negative)**0.5     # semideviation

def get_ind_returns():
    """
    Load and fromat the Ken French 30 Industry Portfolios Value Weighted Monthly Returns
    """

    ind = pd.read_csv("data/ind30_m_vw_rets.csv", header=0, index_col=0)/100
    ind.index = pd.to_datetime(ind.index, format="%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_size():
    """
    Load the size file for the Ken French 30 Industry Portfolios
    """
    ind = pd.read_csv("data/ind30_m_size.csv", header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind

def get_ind_nfirms():
    """
    Load the size file for the Ken French 30 Industry Portfolios
    """
    ind = pd.read_csv("data/ind30_m_nfirms.csv", header = 0, index_col = 0)
    ind.index = pd.to_datetime(ind.index, format = "%Y%m").to_period("M")
    ind.columns = ind.columns.str.strip()
    return ind
    
def annualize_rets(r, periods_per_year):
    """
    Annualizes a set of returns
    """

    compounded_growth = (1+r).prod()
    n_periods = r.shape[0]
    return compounded_growth**(periods_per_year/n_periods)-1

def annualize_vol(r, periods_per_year):
    """
    Annualizes the vol of a set of returns
    """

    return r.std()*(periods_per_year**0.05)

def sharpe_ratio(r, riskfree_rate, periods_per_year):
    """
    Computes the annualized sharpe ratio of a set of returns
    """

    # convert the annual riskfree rate to per period

    rf_per_period = (1 + riskfree_rate)**(1/periods_per_year)-1
    excess_ret = r-rf_per_period
    ann_ex_ret = annualize_rets(excess_ret, periods_per_year)
    ann_vol = annualize_vol(r, periods_per_year)
    return ann_ex_ret/ann_vol

def portfolio_return(weights, returns):
    """
    Weights -> Returns
    """
    return weights.T @ returns

def portfolio_vol(weights, covmat):
    """
    Weights -> Volatility
    """
    return (weights.T @ covmat @ weights)**0.5

def plot_2aef(n_points, er, cov, style=".-"):
    """
    Plots the 2-asset efficient frontier
    """

    if er.shape[0] != 2:
        raise ValueError("plot_2aef can only plot 2-asset frontiers")
    weights = [np.array([w, 1-w]) for w in np.linspace(0, 1, n_points)]
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    return ef.plot.line(x = "Volatility", y="Returns", style=style)

def minimize_vol(target_return, er, cov):
    # Function utilizes optimization techniques to find the weights that minimize portfolio volatility while achieving 
    # the desired target return. It uses the covariance matrix to consider the 
    # relationships between different assets when determining the optimal weights.
    """
    Returns the optimal weights that achieve the target return
    given a set of expected returns and a covariance matrix
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    return_is_target = {'type': 'eq',
                        'args': (er,),
                        'fun': lambda weights, er: target_return - portfolio_return(weights,er)
    }
    weights = minimize(portfolio_vol, init_guess,
                       args=(cov,), method='SLSQP', #Sequential Least Squares Programming optimization algorithm
                       options={'disp': False},
                       constraints=(weights_sum_to_1,return_is_target),
                       bounds=bounds)
    return weights.x

def optimal_weights(n_points, er, cov):
    """
    List of weights to run the optimizer on to minimize the vol
    """
    target_rs = np.linspace(er.min(), er.max(), n_points)
    weights = [minimize_vol(target_return, er, cov) for target_return in target_rs]
    return weights

def plot_ef(n_points, er, cov, style=".-", show_cml = False, riskfree_rate = 0, show_ew=False, show_gmv=False):
    """
    Plots the N-asset efficient frontier
    """
    weights = optimal_weights(n_points, er, cov)
    rets = [portfolio_return(w, er) for w in weights]
    vols = [portfolio_vol(w, cov) for w in weights]
    ef = pd.DataFrame({
        "Returns": rets,
        "Volatility": vols
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style = style)
    if show_gmv:
        w_gmv = gmv(cov)
        r_gmv = portfolio_return(w_gmv, er)
        vol_gmv = portfolio_vol(w_gmv, cov)
        # display GMV
        ax.plot([vol_gmv], [r_gmv], color="midnightblue", marker="o", markersize = 10)
    if show_ew:
        n = er.shape[0]
        w_ew = np.repeat(1/n, n)
        r_ew = portfolio_return(w_ew, er)
        vol_ew = portfolio_vol(w_ew, cov)
        # display EW
        ax.plot([vol_ew], [r_ew], color="goldenrod", marker="o", markersize = 10)
    if show_cml:
        ax.set_xlim(left = 0)
        w_msr = msr(riskfree_rate, er, cov)
        r_msr = portfolio_return(w_msr, er)
        vol_msr = portfolio_vol(w_msr, cov)
        # Add CML -> Capital Market Line
        cml_x = [0, vol_msr]
        cml_y = [riskfree_rate, r_msr]
        ax.plot(cml_x, cml_y, color = "green", marker="o", linestyle = "dashed")
    return ax

def gmv(cov):
    """
    Returns the weight of the Global Minimum Vol portfolio
    given the covaraince matrix
    """
    n = cov.shape[0]
    return msr(0, np.repeat(1, n), cov)


def msr(riskfree_rate, er, cov):
    """
    RiskFree Rate + Expected Return + Covariance -> W
    """
    n = er.shape[0]
    init_guess = np.repeat(1/n, n)
    bounds = ((0.0, 1.0),) * n # an N-tuple of 2-tuples!
    # construct the constraints
    weights_sum_to_1 = {'type': 'eq',
                        'fun': lambda weights: np.sum(weights) - 1
    }
    def neg_sharpe_ratio(weights, riskfree_rate, er, cov):
        """
        Returns the negative of the sharpe ratio, given weights
        """
        r = portfolio_return(weights, er)
        vol = portfolio_vol(weights, cov)
        return -(r - riskfree_rate)/vol

    results = minimize(neg_sharpe_ratio, init_guess,
                       args=(riskfree_rate, er, cov,), method='SLSQP', #Sequential Least Squares Programming optimization algorithm
                       options={'disp': False},
                       constraints=(weights_sum_to_1),
                       bounds=bounds)
    return results.x

def get_total_market_index_returns():
    """
    Load the 30 industry portfolio data and derive the returns of a capweighted total market index
    """
    ind_nfirms = get_ind_nfirms()
    ind_size = get_ind_size()
    ind_return = get_ind_returns()
    ind_mktcap = ind_nfirms * ind_size
    total_mktcap = ind_mktcap.sum(axis=1)
    ind_capweight = ind_mktcap.divide(total_mktcap, axis="rows")
    total_market_return = (ind_capweight * ind_return).sum(axis="columns")
    return total_market_return

def run_ccpi(risky_r, safe_r=None, m=3, start=1000, floor=0.8, riskfree_rate=0.03):
    """
    Run a backtest of the CPPI strategy, given a set of returns for the risky asset
    Returns a dictionary containing: Asset Value History, Risk Budget History, Risky Weight History
    """
    # set up the CPPI parameters
    dates = risky_r.index
    n_steps = len(dates)
    account_value = start
    floor_value = start*floor

    if isinstance(risky_r, pd.Series):
        risky_r = pd.DataFrame(risky_r, columns=["R"])

    if safe_r is None:
        safe_r = pd.DataFrame().reindex_like(risky_r)
        safe_r.values[:] = riskfree_rate/12 #fast way to set all values to a number
    # set up some DataFrames for saving intermediate values
    account_history = pd.DataFrame().reindex_like(risky_r)
    cushion_history = pd.DataFrame().reindex_like(risky_r)
    risky_w_history = pd.DataFrame().reindex_like(risky_r)

    for step in range(n_steps):
        cushion = (account_value - floor_value)/account_value
        risky_w = m*cushion
        risky_w = np.minimum(risky_w, 1)
        risky_w = np.maximum(risky_w, 0)
        safe_w = 1-risky_w
        risky_alloc = account_value*risky_w
        safe_alloc = account_value*safe_w
        ## update the account value for this time step
        account_value = risky_alloc*(1+risky_r.iloc[step]) + safe_alloc*(1+safe_r.iloc[step])
        # save the values so I can look at the history and plot it etc
        cushion_history.iloc[step] = cushion
        risky_w_history.iloc[step] = risky_w
        account_history.iloc[step] = account_value

    risky_wealth = start*(1+risky_r).cumprod()

    backtest_result = {
        "Wealth": account_history,
        "Risky Wealth": risky_wealth, 
        "Risk Budget": cushion_history,
        "Risky Allocation": risky_w_history,
        "m": m,
        "start": start,
        "floor": floor,
        "risky_r":risky_r,
        "safe_r": safe_r,
        "drawdown": drawdown,
        "peak": peak_history,
        "floor": floorval_history
    }

    return backtest_result

    