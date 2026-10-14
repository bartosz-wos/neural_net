"""randomized_smoothing_oracle.py -- the oracle behind every numeric
expectation in tests/test_randomized_smoothing.cpp.

Run with a venv that has numpy/scipy/statsmodels:
    uv pip install --python <venv>/bin/python numpy scipy statsmodels
    <venv>/bin/python tests/randomized_smoothing_oracle.py

Validated against scipy 1.18.1, statsmodels 0.15.0, numpy 2.5.4.

CITATION: Cohen, Rosenfeld, Kolter, "Certified Adversarial Robustness via
Randomized Smoothing", ICML 2019, arXiv:1902.02918. (The expansion queue
proposed 1612.00483, which the arXiv API returns as an astrophysics paper on
methanol masers.)

Every number the test suite asserts was produced by this script. The reasons
for that discipline are recorded in EXPANSION_QUEUE.md: an oracle
transcribed by the same author as the implementation is not independent, and
the KD / metrics / adversarial oracles in this repo each had bugs of their own
before a line of C++ existed.

NOTE: statsmodels 0.15.0 REMOVED method="exact" from proportion_confint. The
paper's Appendix C.2 cites `proportion_confint(k, n, alpha=2*alpha,
method="beta")`, which still works, and the script verifies directly that the
underlying bound satisfies its defining property P(X >= k) == alpha.
"""
import numpy as np, math
from scipy.stats import norm, binom, binomtest, beta
np.set_printoptions(precision=17)

def sec(t): print("\n===== "+t+" =====")

sec("Phi(x) = 0.5*erfc(-x/sqrt2)   [scipy norm.cdf]")
for x in [-8.0,-3.5,-2.0,-1.0,-0.5,0.0,0.5,1.0,2.0,3.5,8.0]:
    print(f"  x={x:6.1f}  {float(norm.cdf(x)):.17g}")

sec("Phi^-1(p)  [scipy norm.ppf]")
for p in [1e-12,1e-6,1e-4,0.001,0.005,0.01,0.05,0.1,0.25,0.5,0.75,0.9,0.95,0.99,0.995,0.999,1-1e-6,1-1e-12]:
    print(f"  p={p:<12.10g}  {float(norm.ppf(p)):.17g}")

sec("Clopper-Pearson one-sided (1-alpha) LOWER: Beta^-1(alpha; k, n-k+1), k=0 -> 0")
for (k,n,a) in [(0,10,0.05),(1,10,0.05),(0,100,0.001),(1,100,0.001),(3,100,0.05),(50,100,0.05),
                (70,100,0.05),(95,100,0.05),(99,100,0.01),(100,100,0.001),(100,1000,0.001),
                (999,1000,0.001),(927,1000,0.001),(500,1000,0.001)]:
    lo = 0.0 if k==0 else float(beta.ppf(a,k,n-k+1))
    print(f"  k={k:5d} n={n:5d} a={a:<7} -> {lo:.17g}")

sec("two-sided binom p-value at p=0.5  [scipy binomtest]")
for (nA,nB) in [(1,0),(2,1),(3,0),(10,9),(10,5),(6,4),(5,5),(20,11),(100,1),(3,10),(9,6)]:
    n=nA+nB
    print(f"  nA={nA:4d} nB={nB:4d} -> {binomtest(nA,n,0.5).pvalue:.17g}")

sec("Theorem 1:  R = sigma/2 * (Phi^-1(pA) - Phi^-1(pB))")
for (pA,pB,s) in [(0.9,0.1,1.0),(0.7,0.3,2.0),(0.99,0.01,0.5),(0.6,0.4,1.0),(0.92785496303410619,0.07214503696589381,0.5)]:
    print(f"  pA={pA:.17g} pB={pB:.17g} sigma={s} -> {s/2*(float(norm.ppf(pA))-float(norm.ppf(pB))):.17g}")

sec("Theorem 1 symmetric form R = sigma*Phi^-1(pA) with pB = 1-pA")
for (pA,s) in [(0.9,1.0),(0.7,2.0),(0.99,0.5),(0.6,1.0),(0.92785496303410619,0.5)]:
    print(f"  pA={pA:.17g} sigma={s} -> {s*float(norm.ppf(pA)):.17g}")

sec("Theorem 6:  (1/(1-alpha)) * (Y - alpha*m - sqrt(max(0, 2m*alpha*(1-alpha)*ln(1/rho) - ln(1/rho)/3)))/m")
def thm6(Y,m,alpha,rho):
    if m<=0: return 0.0
    t = 2*m*alpha*(1-alpha)*math.log(1.0/rho) - math.log(1.0/rho)/3.0
    s = math.sqrt(t) if t>0 else 0.0
    return (1.0/(1.0-alpha))*(Y-alpha*m-s)/m
for (Y,m,a,rho) in [(0,100,0.05,0.05),(100,100,0.001,0.001),(900,1000,0.001,0.001),(500,1000,0.001,0.001),(950,1000,0.001,0.001),(0,1000,0.001,0.001)]:
    print(f"  Y={Y:5d} m={m:5d} a={a:<7} rho={rho:<7} -> {thm6(Y,m,a,rho):.17g}")

sec("Prop 3/4 closed form: true radius |w.x+b|/||w||, exact p_A = Phi(r/sigma)")
w=np.array([0.6,-0.8,0.0]); b=0.35; x=np.array([0.9,0.2,-0.5])
nw=float(np.linalg.norm(w)); true_r=abs(float(w@x)+b)/nw
sig=0.5
print(f"  ||w||={nw:.17g}  true_radius={true_r:.17g}  p_A_exact={float(norm.cdf(true_r/sig)):.17g}")
print(f"  Phi^-1(p_A)={float(norm.ppf(float(norm.cdf(true_r/sig)))):.17g}  sigma*that={sig*float(norm.ppf(float(norm.cdf(true_r/sig)))):.17g}")

sec("Gaussian sign-flip probability: P(x0+sigma*Z > 0) = Phi(x0/sigma)")
for (x0,s) in [(0.9,0.5),(-0.4,0.5),(1.5,1.0),(0.0,1.0)]:
    print(f"  x0={x0} sigma={s} -> {float(norm.cdf(x0/s)):.17g}")
