#!/usr/bin/env python
# coding: utf-8

# In[112]:


import pandas as pd
import numpy as np


# In[113]:


import keras


# In[114]:


from keras.models import Sequential


# In[115]:


from keras.layers import Dense


# In[116]:


df=pd.read_csv(r"D:\ai\g.h.I\Salary_Data.csv")


# In[117]:


df.head(2)


# In[118]:


X=df.iloc[:,0:1].values
y=df.iloc[:,0:2].values


# In[119]:


from sklearn.model_selection import train_test_split


# In[120]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=1)


# In[121]:


X_train.shape


# In[122]:


model = Sequential()
model.add(Dense(200, input_dim=1, activation='relu'))
model.add(Dense(200, input_dim=200, activation='relu'))
model.add(Dense(1, activation='linear'))


# In[123]:


keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, amsgrad=False)
model.compile(loss='mean_squared_error', optimizer='RMSprop', metrics=['mean_absolute_percentage_error'])


# In[124]:


#Neural Network Model
model.summary()


# In[125]:


history = model.fit(X_train, y_train, epochs=50, batch_size=32,validation_split=0.15,validation_data=None,verbose=1)


# In[126]:


keras.backend.clear_session()


# In[127]:


X_test


# In[128]:


model.predict(X_test)


# In[129]:


from matplotlib import pyplot as plt


# In[132]:


plt.plot(model.predict(X_test))


# In[ ]:





# In[ ]:




