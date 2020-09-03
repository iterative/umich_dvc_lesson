import pandas as pd 
from sklearn.linear_model import LinearRegression
import pickle

X = pd.read_csv("data/features.csv")
y = pd.read_csv("data/labels.csv")

# Train a model
reg = LinearRegression().fit(X, y)
print(reg.score(X,y))

# Write it to file
filename = 'model.pkl'
pickle.dump(reg, open(filename, 'wb'))

