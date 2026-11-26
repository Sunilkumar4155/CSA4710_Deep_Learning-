import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
rng=np.random.default_rng(2);X=np.linspace(0,1,120).reshape(-1,1);y=np.sin(2*np.pi*X[:,0])+rng.normal(0,.15,120)
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.3,random_state=1)
for d in [1,3,15]:
 m=make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(Xtr,ytr)
 print(f"Degree {d}: train MSE={mean_squared_error(ytr,m.predict(Xtr)):.4f}, test MSE={mean_squared_error(yte,m.predict(Xte)):.4f}")
plt.scatter(Xte,yte,s=15,label="Test data")
xx=np.linspace(0,1,300).reshape(-1,1)
for d in [1,3,15]:
 m=make_pipeline(PolynomialFeatures(d),LinearRegression()).fit(Xtr,ytr);plt.plot(xx,m.predict(xx),label=f"Degree {d}")
plt.legend();plt.title("Underfit / good fit / overfit");plt.tight_layout();plt.savefig("fit_comparison.png")