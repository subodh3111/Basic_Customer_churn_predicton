#!/usr/bin/env python
# coding: utf-8

# In[3]:


#subodh Kumar
import pandas as pd 


# In[4]:


# below file contains customer churn prediction (whether customer exited bank or not)
df= pd.read_csv("C:/Users/hp/Downloads/Churn_Modelling (1).csv")


# In[5]:


# below fig dipicts overview of dataset
df.head()


# In[6]:


df.tail()


# In[7]:


#drop columns which is not relevent for our interest(not important for our data predictions)
df.drop(columns = ['RowNumber','CustomerId','Surname'],inplace=True)


# In[8]:


#filtered data overview
df.head()


# In[9]:


#below is dimension of the data which is 10k rows and 
#11 columns which consists of various attributes of customer 
df.shape


# In[10]:


#basic info about the attributes and their datatyes
df.info()


# In[11]:


#cheching duplicates in a row, here 0 indicates tht there is no duplicates present 
#in above dataset
df.duplicated().sum()


# In[12]:


df['Exited'].value_counts()


# In[13]:


df['Geography'].value_counts()


# In[14]:


#one - hot encoding of geography and gender
df=pd.get_dummies(df,columns=['Geography','Gender'],drop_first=True)


# In[15]:


df


# In[16]:


#split the dataset into two parts 
X= df.drop(columns=['Exited'])
y=df['Exited']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=1)


# In[17]:


X


# In[18]:


y


# In[19]:


#X_train.shape
#X_test.shape
y_test.shape


# In[21]:


from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
X_train_trf = scaler.fit_transform(X_train)
X_test_trf = scaler.transform(X_test)


# In[26]:


X_train_trf.shape


# In[25]:


#import essentials libraries
import tensorflow
from tensorflow import keras
from tensorflow.keras import Sequential 
from tensorflow.keras.layers import Dense


# In[42]:


#building the model which consists of input layer(input layer contains 11 parameters)
#, hidden layer(hidden layers consists of two layers each consists of 11 perceptron 
#and use activation function to classify the output) and 
#output layer(output layer is only one perceptron/neurons 
#which basically classify the incoming data from hidden layers)
model = Sequential()
model.add(Dense(11,activation='relu',input_dim=11))
model.add(Dense(11,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# In[43]:


#here is the summary of entire model
model.summary()


# In[44]:


#Binary cross-entropy is ideal for binary classification tasks like predicting customer churn (Exited column in your case).
#Adam optimizer helps in achieving faster convergence with adaptive learning rates.
model.compile(loss='binary_crossentropy', optimizer='Adam')


# In[46]:


#calculating wights and biases
model.fit(X_train_trf,y_train,epochs=100)


# In[47]:


#wights and biases values of 0th hidden layers
model.layers[0].get_weights()


# In[48]:


#wights and biases values of 0th hidden layers
model.layers[1].get_weights()


# In[49]:


#wights and biases values of 0th hidden layers
model.layers[2].get_weights()


# In[50]:


model.layers[3].get_weights()


# In[51]:


#since our prediction is coming in the range of 0 to 1 so we have to define 
#the threshold for classification right now we can assume the threshold value 
#is 0.5(but actual threhold omly be defined after plotting roc curve  )
y_log=model.predict(X_test_trf)


# In[52]:


import numpy as np
y_pred=np.where(y_log>0.5,1,0)


# In[53]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:


#our current accuracy is 86.1 but it can be increased further by following methods
# 1) activation function should be relu-->relu gives more accurate result
# 2) by increasing no. of perceotron/node
# 3) no. of epochs 
# 4)

