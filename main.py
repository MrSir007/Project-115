import csv
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import numpy as np

getData = pd.read_csv("escape_velocity.csv")
velocityPop = getData["Velocity"].tolist()
escapePop = getData["Escaped"].tolist()

X = np.reshape(velocityPop,(len(velocityPop ),1))
Y = np.reshape(escapePop,(len(escapePop),1))

lr = LogisticRegression()
lr.fit(X,Y)

plt.figure()
plt.scatter(X.ravel(),Y,color="black",zorder=20)

def model (x) :
  return 1 / (1 + np.exp(-x))

xSpaced = np.linspace(0,100,200)
chances = model(xSpaced * lr.coef_ + lr.intercept_).ravel()

plt.plot(xSpaced,chances,color="red",linewidth=3)
plt.axhline(y=0,color="black",linestyle="-")
plt.axhline(y=0.5,color="blue",linestyle="--")
plt.axhline(y=1,color="black",linestyle="-")
plt.axvline(x=xSpaced[23],color="blue",linestyle="--")
plt.xlabel("Y")
plt.ylabel("X")
plt.xlim(0,30)

plt.show()