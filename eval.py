import pickle
from sklearn.datasets import make_regression
from  sklearn.metrics import r2_score, mean_squared_error
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

model = pickle.load(open("models/model.pkl", "rb"))

# Load the test data
X_test = pd.read_csv("data/test_features.csv")
y_test = pd.read_csv("data/test_y.csv").iloc[:, 0].values

# Test on the model
y_hat = model.predict(X_test)

# Get r2
test_r2 = r2_score(y_test,y_hat)
# Print out MSE
test_mse = mean_squared_error(y_test,y_hat)

with open('test_metrics.json', 'w') as f:
    json.dump({'r2': test_r2,"MSE" : test_mse}, f)

##### Make a pretty picture! ######
res_df = pd.DataFrame(list(zip(y_test,y_hat)), columns = ["true","pred"])

axis_fs=14
title_fs=16

ax = sns.scatterplot(x="true", y="pred",data=res_df)
ax.set_aspect('equal')
ax.set_xlabel('True y',fontsize = axis_fs)
ax.set_ylabel('Predicted y', fontsize = axis_fs)
ax.set_title('Residuals', fontsize = title_fs)

# Make it pretty- square aspect ratio
ax.plot([-1000, 1000], [-1000, 1000], 'black', linewidth=1)
plt.ylim((-1000,1000))
plt.xlim((-1000,1000))

plt.tight_layout()
plt.savefig("residuals.png",dpi=120)
