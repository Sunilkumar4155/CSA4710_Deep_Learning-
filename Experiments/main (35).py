import numpy as np
from scipy.stats import entropy
import matplotlib.pyplot as plt
p=np.array([.5,.3,.2]); H_scipy=entropy(p,base=2); H_manual=-np.sum(p*np.log2(p))
actual=np.array([0,0,1]); q1=np.array([.2,.3,.5]);q2=np.array([.1,.1,.8])
ce1=-np.sum(actual*np.log(q1));ce2=-np.sum(actual*np.log(q2))
P=np.array([.6,.3,.1]);A=np.array([.5,.4,.1]);B=np.array([.3,.3,.4])
klA=entropy(P,A);klB=entropy(P,B)
print(f"Weather entropy SciPy={H_scipy:.4f} bits; manual={H_manual:.4f}")
print(f"Cross entropy original={ce1:.4f}; improved={ce2:.4f}")
print(f"KL(P||A)={klA:.4f}; KL(P||B)={klB:.4f}; closer model={'A' if klA<klB else 'B'}")
plt.bar(["Model A","Model B"],[klA,klB]);plt.ylabel("KL divergence");plt.title("Model distribution comparison");plt.tight_layout();plt.savefig("kl_comparison.png")