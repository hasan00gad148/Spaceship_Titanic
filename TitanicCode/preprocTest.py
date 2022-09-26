import numpy as np
import pandas as pd
import pickle
 
df = pd.read_csv(".\\data\\test.csv")

values = {
        'HomePlanet': "no_info",
        'CryoSleep':  0,
        'Cabin': "no_info/no_info/no_info",
        'Destination': "no_info",
        'Age': df.Age.mean(), 
        'VIP': 0, 
        'RoomService': df.RoomService.mean(), 
        'FoodCourt': df.FoodCourt.mean(), 
        'ShoppingMall': df.ShoppingMall.mean(), 
        'Spa': df.Spa.mean(), 
        'VRDeck': df.VRDeck.mean(), 
        'Name': "no_info"
        }
for c in df.columns:
    if c in ['Transported','PassengerId']:
        continue
    if c in['CryoSleep','VIP']:
       df.loc[:,c] = df.loc[:,c].astype("bool")
    df.loc[:,c].fillna(value=values[c],inplace=True,)


df["Cabin"] = df["Cabin"].apply(str)
df["Cabin"] = df["Cabin"].str.replace("/", "-")
df[["Cabin1__", "Cabin2__", "Cabin3__"]] = df['Cabin'].str.split("-", expand=True)
df.drop(['Cabin2__',], axis=1, inplace=True)
df.drop(['Cabin',], axis=1, inplace=True)

df["PassengerId"] = df["PassengerId"].apply(str)
df[["gggg", "pp",]] = df['PassengerId'].str.split("_", expand=True)
df.drop(['gggg',], axis=1, inplace=True)

# df.drop(['PassengerId',], axis=1, inplace=True)

df.drop(['Name',], axis=1, inplace=True)

X=df
encoders = pickle.load(open("encoders.pkl","rb"))
cols = list(encoders.keys())
for c in cols:

        print(np.sum(X.loc[:,c].isna()))
        tmp = X[[c]]
        tmp=np.reshape(tmp,(-1,1))
        
        print(tmp.shape)
        
        lb = encoders[c]

        onehe=lb.transform(tmp)
      
        col=[c+"__"+str(i) for i in range(onehe.shape[1])]
        tmp=pd.DataFrame(onehe,columns=col)
        print(tmp.shape)
        X=pd.concat([X.reset_index(drop=True),tmp.reset_index(drop=True)],axis=1)

        print(X.shape)
        X.drop([c], axis=1, inplace=True)
print(X.describe())
print(X.shape)        


X.to_csv('.\\data\\X_test.csv') 








