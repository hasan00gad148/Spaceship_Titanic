import numpy as np
import pandas as pd
import xgboost as xgb
import pickle

xgboostModel = pickle.load(open("xgboostModel.pkl","rb"))
XTest=pd.read_csv(".\\data\\x_test.csv")
cols = list(XTest.columns)
cols.remove("PassengerId")

X = XTest[cols]

tmp = xgboostModel.predict(X)
ss=np.reshape(tmp,(-1,1))
ss = pd.DataFrame(ss,columns=["Transported"]).reset_index(drop=True)
ss.loc[:,"Transported"] = ss.loc[:,"Transported"].apply(lambda x:True if x ==1 else False)
ss.loc[:,"Transported"] = ss.loc[:,"Transported"].astype("bool")
# print(ss.shape)
ss=pd.concat([XTest[["PassengerId"]].reset_index(drop=True),ss],axis=1)
ss.to_csv('.\\data\\submission.csv',index = False)
