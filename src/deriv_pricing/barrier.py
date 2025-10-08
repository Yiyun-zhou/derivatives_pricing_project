from __future__ import annotations
import numpy as np
from scipy.stats import norm
from typing import Callable
from .black_scholes import bs_european_option
from .monte_carlo import mc_european_call
from .greeks import finite_diff


def _phi_call_put(option):  # 'call'/'put' -> +1/-1
    o = option.lower().strip()
    return +1 if o == 'call' else -1

def _eta_up_down(barrier_type):  # 'up'/'down' -> +1/-1
    b = barrier_type.lower().strip()
    return +1 if b == 'up' else -1

def price_barrier_rr(S0,K,B,r,q,sigma,T, option='call', barrier_type='down', knock='out'):
    """
    Reiner-Rubinstein single barrier, continuous monitoring, no rebate.
    option: 'call'/'put'; barrier_type: 'down'/'up'; knock: 'in'/'out'
    """
    if (barrier_type == 'down' and B >= S0) or (barrier_type == 'up' and B <= S0):
        # already knocked (at inception) -> simple corner cases
        return 0.0 if knock=='out' else bs_european_option(S0,K,r,q,sigma,T,option) 

    phi = _phi_call_put(option)
    eta = _eta_up_down(barrier_type)

    mu = (r - q - 0.5*sigma**2) / (sigma**2)
    lam = (r - q + 0.5*sigma**2) / (sigma**2)
    sigT = sigma*np.sqrt(T)

    x1 = (np.log(S0/K)/sigT) + lam*sigT
    x2 = x1 - sigT
    y1 = (np.log(B**2/(S0*K))/sigT) + lam*sigT
    y2 = y1 - sigT
    z  = (np.log(B/S0)/sigT) + lam*sigT

    # helper powers
    S_fac = (B/S0)**(2*lam)
    K_fac = (B/S0)**(2*lam -2)

    if knock == 'out':
        # Four RR cases collated (see RR 1991). Here we implement via call/put symmetry.
        if option=='call' and barrier_type=='down':
            # Down-and-out call
            term1 = S0*np.exp(-q*T)*( norm.cdf(x1) - S_fac*norm.cdf(y1) )
            term2 = -K*np.exp(-r*T)*( norm.cdf(x2) - K_fac*norm.cdf(y2) )
            return float(term1 + term2)
        if option=='put' and barrier_type=='up':
            # Up-and-out put
            term1 = -S0*np.exp(-q*T)*( norm.cdf(-x1) - S_fac*norm.cdf(-y1) )
            term2 =  K*np.exp(-r*T)*( norm.cdf(-x2) - K_fac*norm.cdf(-y2) )
            return float(term1 + term2)
        # The other two via put-call/barrier symmetry:
        if option=='call' and barrier_type=='up':
            # Parity: up&out call = vanilla call - down&in call (mirror)
            vanilla = bs_european_option(S0,K,r,q,sigma,T,'call')
            din = price_barrier_rr(S0,K,B,r,q,sigma,T,'call','down','in')
            return float(vanilla - din)
        if option=='put' and barrier_type=='down':
            vanilla = bs_european_option(S0,K,r,q,sigma,T,'put')
            uin = price_barrier_rr(S0,K,B,r,q,sigma,T,'put','up','in')
            return float(vanilla - uin)

    else:  # knock-in
        # knock-in = vanilla - knock-out (same option)
        vanilla = bs_european_option(S0,K,r,q,sigma,T,option)
        kout = price_barrier_rr(S0,K,B,r,q,sigma,T,option,barrier_type,'out')
        return float(vanilla - kout)
    
def mc_barrier_knockout_call(S0, K, B, r, q, sigma, T, n_paths, n_steps, antithetic, seed, brownian_bridge=False, barrier_type='down'):
    '''
    Use Monte-Carlo to price a barrier option
    Step 1: For each time step, decide whether barrier is violated or not, if yes, value = 0.
    Step 2: Find mean of all sample paths
    '''

    dt = T / n_steps
    disc = np.exp(-r*T)

    rng = np.random.default_rng(seed)
    if antithetic:
        z = rng.standard_normal((n_paths // 2, n_steps))
        z = np.concatenate([z, -z])
    else:
        z = rng.standard_normal((n_paths, n_steps))
    
    payoff = np.zeros(n_paths)
    
    S = np.full(n_paths, S0, dtype=float) # 一维数组存每条路径当前价
    knocked_out = np.zeros(n_paths, dtype=bool) # 用 bool 数组记录是否已触碰障碍, so originally all false, ie. not violating barrier

    # |= 是**按位“或并赋值”**运算符，原来true就不变，原来false如果满足条件，就变成true

    for t in range(0, n_steps):
        alive = ~knocked_out # options that are still alive, index

        S[alive] *= np.exp((r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z[alive, t])

        if barrier_type == 'down': # down-out
            knocked_out |= S <= B  # turn True if S <= B, ie violated barrier
        elif barrier_type == 'up': # up-out
            knocked_out |= S >= B 
    
    payoff[alive] = np.maximum(S[alive] - K, 0.0)
    price = disc * payoff.mean()
    stdder = disc * payoff.std(ddof=1) / np.sqrt(len(payoff))

    return price, stdder




# Greeks for barrier (use finite difference)
# Barrier Greeks（数值差分）
# 复用：
# 你已经写好的数值差分：finite_diff / vega_fd
# 薄封装：
# barrier_delta = finite_diff(lambda s: price_barrier_rr(s, ...), S0)（或对 MC 版本差分）
# barrier_vega = finite_diff(lambda v: price_barrier_rr(S0, ..., v, ...), sigma)
# 用相对步长 h（你之前用过），稳定性更好。


