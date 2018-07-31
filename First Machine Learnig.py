import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression

sns.set(color_codes =True)
df=pd.read_csv("iris.data",header = -1)
print df.head()

x=10*np.random.rand(100)
y=10*x+np.random.rand(100)

model=LinearRegression(fit_intercept=True)
X=x.reshape(-1,1)
model.fit(X,y)

x1Fit=np.linspace(-1,11)
xFit=x1Fit.reshape(-1,1)
yFit=model.predict(xFit)

plt.scatter(x,y)
plt.plot(xFit,yFit)
plt.show()


