import numpy as np
import matplotlib.pyplot as plt
rng=np.random.default_rng(1);X=rng.normal(size=(180,3));true=np.array([3.,-2.,1.5]);y=X@true+0.5
def train(kind,epochs=80,lr=.04):
 w=np.zeros(3);v=np.zeros(3);m=np.zeros(3);s=np.zeros(3);losses=[]
 for t in range(1,epochs+1):
  idx=np.arange(len(y)) if kind=="GD" else rng.integers(0,len(y),32)
  g=X[idx].T@(X[idx]@w-y[idx])/len(idx)
  if kind=="Momentum": v=.9*v+g; w-=lr*v
  elif kind=="Adam": m=.9*m+.1*g;s=.999*s+.001*g*g;w-=lr*(m/(1-.9**t))/(np.sqrt(s/(1-.999**t))+1e-8)
  else: w-=lr*g
  losses.append(np.mean((X@w-y)**2))
 return w,losses
for name in ["GD","SGD","Momentum","Adam"]:
 w,l=train(name);print(name,"weights",np.round(w,3),"final MSE",round(l[-1],5))
 plt.plot(l,label=name)
plt.legend();plt.xlabel("Epoch");plt.ylabel("MSE");plt.title("Optimizer convergence");plt.tight_layout();plt.savefig("optimizer_loss.png")