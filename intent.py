''' 
Simulation illustrating control function approach to price endogeneity for probit choice model, from

''The Effectiveness of Field Price Discretion: Empirical Evidence from Auto Lending''
Robert Phillips, A. Serdar Şimşek, Garrett van Ryzin

Each groupwise demand shock enters price equation. Adding residuals from groupwise means as predictor effectively 
''' 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from numba import jit, njit, vectorize


'''Parameters'''
rand = np.random.default_rng(12)
N_CUSTOMER = 2000
N_SELL = 400
draws = N_SELL*N_CUSTOMER


util_labels = ['ancil', 'over', 'price']
util = np.array([2, 1])
price_coef = -0.1
util_const = -0.2

cost_labels = ['ancil', 'over']
cost = np.array([1, 0.5])
cost_const = 0.2
SHOCK_SCALE = 2
sigma_err = np.eye(2)
sigma_err[sigma_err == 0] = 0.2

'''Make DF template'''

seller_id = rand.integers(0, N_SELL, draws)
index_names = ['seller_id', 'transaction']
index = pd.MultiIndex.from_tuples(
        zip(*[seller_id, range(draws)]),
        names=index_names)

''' Price and cost structure'''

errors = rand.multivariate_normal(np.repeat(0, 2), sigma_err, size=draws)
err_labels = ['err_util', 'err_price']
df = pd.DataFrame(errors,
                  columns=err_labels, index=index)

# add constant
df = df.assign(const=1)

# price determinants.
full_traits = pd.DataFrame(
        rand.integers(2, size=(draws, len(cost_labels))),
        columns=cost_labels,
        index=index)

# price endogenous, since err_price correlated to err_util
price = cost_const + full_traits @ cost + df.err_price

# seller specific shock (exogenous)
shock_temp = SHOCK_SCALE*rand.random(N_SELL) - SHOCK_SCALE/2
for sell in range(N_SELL):
    price[seller_id == sell] += shock_temp[sell]

'''Merge Data, add constant'''

df = (
        df
        .assign(
            price=price,
            choice=lambda dat: (
                util_const + full_traits @ util +
                dat.price*price_coef + dat.err_util > 0)
            )
        .merge(full_traits, left_index=True, right_index=True)
        .drop(err_labels, axis=1)
        )



def taker(ind):
    group = df.index[ind][0]
    regdata = df.iloc[~ind].loc[group]
    predvals





@vectorize
# convenience util for groupwise residuals
def numbaratiovec(x,y,z):
    return (x- y)/(z-1)

# @profile
def first_stage(df):
    '''assumes copy, modifies passed df'''

    clusters = ['seller_id', 'ancil', 'over']
    groups = (df
            .reset_index()
            .groupby(by = clusters)['price']
    )
    
    
    # Groupwise price residual, removing endogenous component
    # ''regression'' on seller_id, ancil, and over variables (all discrete) is just conditional averaging.
    # Residual is just difference between group mean and own variable.
    df = df.assign(
            price_sum = groups.transform(np.sum).values,
            group_count = groups.transform('count').values,
            predicted_price = lambda d: numbaratiovec(
                d.price_sum.to_numpy(),
                d.price.to_numpy(),
                d.group_count.to_numpy()
                    ),
            resid_price = lambda d: d.predicted_price - d.price
            )


    df =  df.drop(['price_sum', 'group_count', 'predicted_price'], axis = 1)

    return df


def second_stage(data):
    # incliudes price residual as regressor. For probit, particular error structure, this implements control function approach.
    two_step = sm.Probit(data.choice, data.drop('choice', axis=1)).fit(disp=False)
    return two_step.params


def one_pass(data, rng):
    '''
    With none, pandas just uses system entropy.
    This guarantees identical results in parallel for each run.

    If instead rand is passed
    '''
    data = data.sample(frac=1, replace=True, random_state=rng)
    data = first_stage(data)
    return second_stage(data)


def bootstrap_pool(data, iters, seed = 200432726853384018591358734668569137389):
    from multiprocessing import Pool
    from itertools import product
    seed_seq = np.random.SeedSequence(seed).spawn(iters)
    # Note printouts are in random order since messages report FIFO 
    rngs = (np.random.default_rng(sd) for sd in seed_seq)
    with Pool() as p:
        coefs = p.starmap(one_pass, product([data], rngs))
    return coefs


if __name__ == "__main__":
    estimates = one_pass(df,rand)
    res = pd.DataFrame(bootstrap_pool(df, 20))
    fig, ax = plt.subplots()
    res.loc[:,"price"].hist(ax = ax, alpha = 0.5)
    ax.axvline(estimates.price, label = 'point estimate', linewidth = 2, color = 'red')
    ax.axvline(res.price.quantile(0.025), label = 'conf interval', linewidth = 2, color = 'green')
    ax.axvline(res.price.quantile(0.975), label = 'conf interval', linewidth = 2, color = 'green')
    ax.legend()
    plt.show()
    fig.show()
    # print(pd.DataFrame(res).head())
    # parallelize bootstrap in two ways, one with pool, one with openmp + c
    # plt.scatter(df.price, df.choice)
    # plt.show()
