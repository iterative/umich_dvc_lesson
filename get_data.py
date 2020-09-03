from sklearn.datasets import make_regression
import pandas as pd
import os
import numpy as np

# If there's no dataset in the project directory, create a reasonably large one. 
# If it exists, append some new observations. 
if os.path.isfile("data/train_features.csv"):
    n = 1
else:
    os.mkdir("data")
    n = 10

for i in range(0,n):    
    X, y = make_regression(10000,n_features = 10)
    
    # Get test and train features
    df = pd.DataFrame(X)
    train_features = df[:8000]
    test_features = df[8000:]
    train_features.to_csv("data/train_features.csv",mode="a", index=False)
    test_features.to_csv("data/test_features.csv", mode="a", index=False)
    
    # Get test and train outputs
    labels = pd.DataFrame(y)
    train_y = labels[:8000]
    test_y = labels[8000:]
    train_y.to_csv("data/train_y.csv",mode="a",index=False)
    test_y.to_csv("data/test_y.csv",mode="a",index=False)


