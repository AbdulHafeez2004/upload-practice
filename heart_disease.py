#!C:\Users\Lenovo\AppData\Local\Programs\Python\Python37-32\python.exe

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import warnings
import pickle
warnings.filterwarnings("ignore")

data = pd.read_csv("heart_disease.csv")
# data = np.array(data)

X = data.drop('target', axis=1)
y = data['target']

# print(X,y)
X_train, X_test, y_train, y_test = train_test_split(
    X.values, y.values, test_size=0.2, random_state=0)
log_reg = LogisticRegression()


log_reg.fit(X_train, y_train)

y_preds = log_reg.predict(X_test)
y_preds_proba = log_reg.predict_proba(X_test)

print(y_preds)
# inputt = [int(x) for x in "45 32 60".split(' ')]


# final = [np.array(inputt)]

# b = log_reg.predict_proba(final)

# print(b)

pickle.dump(log_reg, open('model-heart-disease.pkl', 'wb'))

model = pickle.load(open('model-heart-disease.pkl', 'rb'))

# pickle_y_pred = model.predict_proba(X_test)
# print(pickle_y_pred)
