
# coding: utf-8

# In[30]:


import os


# In[31]:


os.getcwd()


# In[32]:


os.chdir("C:/Users/Dheeraj B/Desktop/dataset")


# In[33]:


ls


# In[34]:


import pandas as pd
import numpy as np


# In[35]:


from sklearn.neighbors import KNeighborsClassifier


# In[36]:


df = pd.read_csv("votes.csv")


# In[37]:


df.info()


# In[38]:


knn = KNeighborsClassifier(n_neighbors=5)


# In[39]:


df.head()


# In[40]:


X = df.drop(["party"], axis=1)


# In[41]:


Y =  df["party"]


# In[42]:


X.shape


# In[43]:


Y.shape


# In[48]:


from sklearn.preprocessing import LabelEncoder


# In[49]:


df = df.apply(LabelEncoder().fit_transform)


# In[50]:


df.head()


# In[52]:


X = df.drop(["party"], axis=1)
Y =  df["party"]


# In[54]:


knn.fit(X,Y)


# In[55]:


train = df.iloc[:400]


# In[56]:


test= df.iloc[400:]


# In[66]:


# Create arrays for the features and the response variables
y = train["party"].values
x=  train.drop('party',axis=1).values


# In[67]:


model1 = knn.fit(X,Y)


# In[69]:


y_test = test["party"]
test_pred = test.drop(["party"], axis =1)


# In[70]:


y_pred = model1.predict(test_pred)


# In[71]:


y_pred


# In[72]:


y_test


# In[73]:


list(y_pred)


# In[74]:


mis_classified = []
for x ,y in zip(y_test,y_pred):
    if x!=y:
        mis_classified.append(x)
print (1-(len(mis_classified)/len(y_test)))        


# In[75]:


df.head()

