
from cgi import print_form
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split

X=pd.read_csv(".\\data\\x_train.csv")
Y=pd.read_csv(".\\data\\y_train.csv")
Y=Y["Transported"]
X_train, X_dev, y_train, y_dev = \
    train_test_split(X, Y, test_size = 0.2,shuffle=True,random_state=10)
    
clf = GradientBoostingClassifier(n_estimators=64, learning_rate=0.25,
    max_depth=2,min_samples_split=32,min_samples_leaf=32,).fit(X_train, y_train)


XTest=pd.read_csv(".\\data\\x_test.csv")
cols = list(XTest.columns)
cols.remove("PassengerId")

X = XTest[cols]

tmp = clf.predict(X)
ss=np.reshape(tmp,(-1,1))
ss = pd.DataFrame(ss,columns=["Transported"]).reset_index(drop=True)
ss.loc[:,"Transported"] = ss.loc[:,"Transported"].apply(lambda x:True if x ==1 else False)
ss.loc[:,"Transported"] = ss.loc[:,"Transported"].astype("bool")
# print(ss.shape)
ss=pd.concat([XTest[["PassengerId"]].reset_index(drop=True),ss],axis=1)
ss.to_csv('.\\data\\submission.csv',index = False)
print(clf.score(X_dev, y_dev))