import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import pickle

X=pd.read_csv(".\\data\\x_train.csv")
Y=pd.read_csv(".\\data\\y_train.csv")
Y=Y["Transported"]
X_train, X_dev, y_train, y_dev = \
    train_test_split(X, Y, test_size = 0.05,shuffle=True,random_state=10)
X_cv, X_test, y_cv, y_test= \
    train_test_split(X_dev, y_dev, test_size = 0.30,shuffle=True,random_state=10)
    
xg_cls = xgb.XGBClassifier(objective ='binary:logistic', colsample_bytree = 1, learning_rate = 0.1,
                max_depth = 4, alpha = 8, n_estimators = 64)


xg_cls.fit(X_train,y_train)

preds = xg_cls.predict(X_train)
print(accuracy_score(preds,y_train),"___",f1_score(preds,y_train))
preds = xg_cls.predict(X_cv)
print(accuracy_score(preds,y_cv),"___",f1_score(preds,y_cv))
preds = xg_cls.predict(X_test)
print(accuracy_score(preds,y_test),"___",f1_score(preds,y_test))
pickle.dump(xg_cls,open("xgboostModel.pkl","wb"))