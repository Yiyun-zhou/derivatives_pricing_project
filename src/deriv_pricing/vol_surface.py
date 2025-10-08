from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# === 你已有的 Hagan SABR IV 近似 ===
def hagan_sabr_iv(F, K, T, alpha, beta, rho, nu, eps=1e-12):
    if F == K:
        num = alpha
        den = F**(1-beta)
        A = 1 + ((1-beta)**2/24)*(np.log(F/(K+eps))**2) + ((1-beta)**4/1920)*(np.log(F/(K+eps))**4)
        return (num/den) * A
    FK = (F*K)**((1-beta)/2.0)
    logFK = np.log(F/(K+eps))
    z = (nu/alpha) * FK * logFK
    xz = np.log((np.sqrt(1 - 2*rho*z + z*z) + z - rho) / (1 - rho + eps))
    vol0 = (alpha / FK) * (z / (xz + eps))
    c1 = ((1 - beta)**2/24.0)*(alpha**2)/(FK**2) + 0.25*(rho*beta*alpha*nu)/FK + (2 - 3*rho**2)*(nu**2)/24.0
    return vol0 * (1 + c1*T)

# === term-structure 合成器：逐期限生成 SABR 参数，再调用你已有的 hagan_sabr_iv ===
def make_synthetic_surface_ts(S0, r, q, strikes, tenors,
                              beta=1.0,
                              sigma_atm_fn=None, rho_fn=None, nu_fn=None):
    """
    返回 dict: {(T, K): iv}
    - beta 固定（equity/FX 常取 0.9~1.0）
    - 其余参数（alpha、rho、nu）随 T 变化：
        alpha(T) = sigma_atm(T) * F(T)^(1 - beta)
        rho(T)   = rho_fn(T)
        nu(T)    = nu_fn(T)
    """
    if sigma_atm_fn is None:
        # Equity常见：短端20%，往长端缓升 ~5 vol pts
        sigma_atm_fn = lambda T: 0.20 + 0.05 * (1.0 - np.exp(-T/1.0))
    if rho_fn is None:
        # 负skew，随期限回归 0（绝对值变小）
        rho_fn = lambda T: -0.6 * np.exp(-T/1.5)
    if nu_fn is None:
        # 短端更弯，长端更平
        nu_fn = lambda T: 0.35 / np.sqrt(1.0 + T)

    F = lambda T: S0 * np.exp((r - q) * T)

    iv = {}
    for T in tenors:
        F_T = F(T)
        sigma_atm_T = sigma_atm_fn(T)
        alpha_T = sigma_atm_T * (F_T**(1.0 - beta))  # beta=1 时 alpha=ATM vol
        rho_T   = rho_fn(T)
        nu_T    = nu_fn(T)

        for K in strikes:
            iv[(T, K)] = hagan_sabr_iv(F_T, K, T, alpha_T, beta, rho_T, nu_T)

    return iv

# === 一个“像真的”默认配置（直接跑） ===
if __name__ == "__main__":
    S0, r, q = 100.0, 0.02, 0.00
    tenors   = np.array([1/12, 0.25, 0.5, 1.0, 2.0])  # 1M, 3M, 6M, 1Y, 2Y
    strikes  = np.linspace(70, 130, 13)               # 全期限共用一组K（简单起见）

    # Equity风格：beta=1，负skew、短端更弯
    beta = 1.0
    sigma_atm = lambda T: 0.20 + 0.05 * (1.0 - np.exp(-T/1.0))
    rho_T     = lambda T: -0.6 * np.exp(-T/1.5)
    nu_T      = lambda T: 0.35 / np.sqrt(1.0 + T)

    iv_surface = make_synthetic_surface_ts(
        S0=S0, r=r, q=q,
        strikes=strikes,
        tenors=tenors,
        beta=beta,
        sigma_atm_fn=sigma_atm,
        rho_fn=rho_T,
        nu_fn=nu_T
    )

    # iv_surface 现在是 {(T,K): implied_vol} 的字典
    # 你可以直接把它喂给 SVI 拟合器；或先转换为总方差 w = iv^2 * T 更稳。
    # 示例：转为 (T, K, k=ln(K/F), w) 表
    F = lambda T: S0 * np.exp((r - q) * T)
    table = []
    for (T, K), iv in iv_surface.items():
        k = np.log(K / F(T))
        w = (iv ** 2) * T
        table.append((T, K, k, iv, w))

    # table 可丢进 pandas.DataFrame 便于画图/校准
    # import pandas as pd
    # df = pd.DataFrame(table, columns=["T","K","k","iv","w"])
    # df.head()


# Smile for a fixed T
# iv dict from make_synthetic_surface
def plot_smile(iv, T, strikes, S0, r=0.0, q=0.0):
    F_T = S0 * np.exp((r - q) * T)          # 远期
    xs = [K / F_T for K in strikes]         # 用 K/F(T)
    ys = [iv[(T, K)] for K in strikes]
    plt.figure(figsize=(6,4))
    plt.plot(xs, ys, marker='o')
    plt.xlabel('K / F(T)  (forward moneyness)')
    plt.ylabel('Implied Vol')
    plt.title(f'IV Smile at T={T:.2f}y')
    plt.grid(alpha=0.4)

def plot_surface(iv, strikes, tenors, S0, r=0.0, q=0.0):
    import numpy as np
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa

    # 构造每个期限自己的 forward moneyness 轴
    X = np.zeros((len(tenors), len(strikes)))  # K/F(T)
    Z = np.zeros_like(X)                       # IV

    for i, T in enumerate(tenors):
        F_T = S0 * np.exp((r - q) * T)
        for j, K in enumerate(strikes):
            X[i, j] = K / F_T                  # forward moneyness
            Z[i, j] = iv[(T, K)]

    Y = np.array(tenors)[:, None] * np.ones_like(X)  # 每行都是该 T

    fig = plt.figure(figsize=(8,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, alpha=0.9)
    ax.set_xlabel('K / F(T)')
    ax.set_ylabel('Tenor T (years)')
    ax.set_zlabel('Implied Vol')
    ax.set_title('Implied Vol Surface (forward moneyness)')


# CALIBRATE VOL SURFACE
def prepare_kw_table(iv_dict, S0, r, q):
    # iv_dict: {(T, K): iv}
    rows = []
    for (T, K), iv in iv_dict.items():
        F = S0 * np.exp((r - q) * T)
        k = np.log(K / F)
        w = (iv ** 2) * T
        rows.append((T, k, w))
    # 分到期聚合
    by_T = {}
    for T, k, w in rows:
        by_T.setdefault(T, {"k": [], "w": []})
        by_T[T]["k"].append(k)
        by_T[T]["w"].append(w)
    return by_T

def svi_raw(k, a, b, rho, m, sigma):
    # 返回 w(k)
    return a + b * (rho * (k - m) + np.sqrt((k - m)**2 + sigma**2))

def objective_theta(theta, k, w, weights=None):
    a, b, rho, m, sig = theta
    w_fit = svi_raw(np.array(k), a, b, rho, m, sig)
    res = w_fit - np.array(w)
    if weights is not None:
        res = res * np.array(weights)
    return np.sum(res**2)

def init_theta_heuristic(k, w):
    k = np.array(k); w = np.array(w)
    i0 = np.argmin(w)
    m0 = float(k[i0])
    rho0 = -0.3
    sig0 = 0.2
    b0 = 0.5
    a0 = float(np.min(w) - b0 * sig0 * np.sqrt(1 - rho0**2) * 0.1)
    return np.array([a0, b0, rho0, m0, sig0])

bounds = [(-5, 5), (1e-6, 10.0), (-0.999, 0.999), (-2.0, 2.0), (1e-6, 5.0)]

from scipy.optimize import minimize

def calibrate_svi_single_T(k, w, weights=None):
    theta0 = init_theta_heuristic(k, w)
    res = minimize(objective_theta, theta0, args=(k, w, weights),
                   bounds=bounds, method='L-BFGS-B',
                   options={'maxiter': 500, 'ftol': 1e-10})
    theta = res.x
    return theta, res

def calibrate_svi_by_tenor(by_T):
    fitted = {}   # T -> theta
    prev = None
    for T in sorted(by_T.keys()):
        k = by_T[T]['k']; w = by_T[T]['w']
        # 可选：对 ATM 附近加权
        weights = 1.0 / (0.5 + np.abs(np.array(k)))
        theta, res = calibrate_svi_single_T(k, w, weights)
        # 简单平滑：和上一期限的参数做 convex-combo（防跳）
        if prev is not None:
            theta = 0.7*theta + 0.3*prev
        fitted[T] = theta
        prev = theta
    return fitted  # dict: T -> (a,b,rho,m,sigma)

def interp_theta(T, thetas_by_T):
    Ts = np.array(sorted(thetas_by_T.keys()))
    thetas = np.array([thetas_by_T[t] for t in Ts])  # shape (nT,5)
    if T <= Ts[0]: return thetas[0]
    if T >= Ts[-1]: return thetas[-1]
    i = np.searchsorted(Ts, T) - 1
    t0, t1 = Ts[i], Ts[i+1]
    w = (T - t0) / (t1 - t0)
    th0, th1 = thetas[i], thetas[i+1]
    # 对 rho 做安全插值
    def arctanh(x): return 0.5*np.log((1+x)/(1-x))
    def tanh(x): return (np.exp(2*x)-1)/(np.exp(2*x)+1)
    a0,b0,r0,m0,s0 = th0; a1,b1,r1,m1,s1 = th1
    r0z, r1z = arctanh(r0), arctanh(r1)
    rz = (1-w)*r0z + w*r1z
    rho = tanh(rz)
    a = (1-w)*a0 + w*a1
    b = (1-w)*b0 + w*b1
    m = (1-w)*m0 + w*m1
    sig = (1-w)*s0 + w*s1
    return np.array([a,b,rho,m,sig])

def sigma_from_surface(S0, r, q, thetas_by_T, K, T):
    F = S0 * np.exp((r - q) * T)
    k = np.log(K / F)
    a,b,rho,m,sig = interp_theta(T, thetas_by_T)
    w = svi_raw(k, a,b,rho,m,sig)
    return np.sqrt(max(w, 1e-12) / max(T, 1e-12))








