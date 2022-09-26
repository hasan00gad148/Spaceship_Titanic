
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score,f1_score
from sklearn.model_selection import train_test_split
import pickle
import keras.api._v2.keras as keras
from keras import Sequential
from keras.layers import Dense,BatchNormalization
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.activations import leaky_relu
import tensorflow as tf



BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,)


X=np.array(pd.read_csv(".\\data\\x_train.csv"))
Y=np.array(pd.read_csv(".\\data\\y_train.csv"))


    

model = Sequential([
# keras.Input(n=36),
Dense(units=25, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),

Dense(units=50, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
Dense(units=100, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
# Dense(units=200, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
# BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
Dense(units=250, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
# Dense(units=200, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
# BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
Dense(units=100, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
Dense(units=50, activation='relu',kernel_regularizer=tf.keras.regularizers.L2(0.01)),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),


Dense(units=25, activation='relu',),
BatchNormalization(axis=-1,momentum=0.99,epsilon=0.001,center=True,scale=True,),
Dense(units=1, activation='linear'),])

model.compile(Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    amsgrad=False,
    name='Adam',
),loss=BinaryCrossentropy(from_logits=True) )
model.fit(X,Y, batch_size=128,epochs=150)
logits = model(X)
f_x = tf.nn.sigmoid(logits)
s= np.array(f_x)
s= np.rint(s)




XTest=pd.read_csv(".\\data\\x_test.csv")
cols = list(XTest.columns)
cols.remove("PassengerId")
X = XTest[cols]
X = np.array(X)
logits = model(X)
f_x = tf.nn.sigmoid(logits)
ss= np.array(f_x)
ss = pd.DataFrame(ss,columns=["Transported"]).reset_index(drop=True)
ss.loc[:,"Transported"] = ss.loc[:,"Transported"].apply(lambda x:True if x >= 0.5 else False)
ss.loc[:,"Transported"] = ss.loc[:,"Transported"].astype("bool")
# print(ss.shape)
ss=pd.concat([XTest[["PassengerId"]].reset_index(drop=True),ss],axis=1)
ss.to_csv('.\\data\\submission11.csv',index = False)