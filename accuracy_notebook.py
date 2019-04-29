
# coding: utf-8

# In[25]:


import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier


# In[26]:


K = 50
X_train = np.fromfile("train_embedding.txt", dtype=np.float32).reshape(-1, 4096)
X_test = np.fromfile("test_embedding.txt", dtype=np.float32).reshape(-1, 4096)

train_df = pd.read_csv("training_triplet_sample.csv")
test_df = pd.read_csv("test_triplet_sample.csv")


# In[27]:


# X_train = X_train[0:X_train.shape[0]:4]
# print(X_train.shape)


# In[28]:


y_test = test_df["query"].apply(lambda x:x.split("/")[2])[0:X_test.shape[0]]
y_train = train_df["query"].apply(lambda x:x.split("/")[2])[0:X_train.shape[0]]


# In[29]:


y_train.shape


# In[30]:


knn_model = KNeighborsClassifier(n_neighbors=K, weights='distance', algorithm='kd_tree', n_jobs=-1)
knn_model.fit(X=X_train, y=y_train)


# In[31]:


_,indx = knn_model.kneighbors(X_test, n_neighbors=K)


# In[32]:


n_indx = []
for i in xrange(0, len(indx)):
    n_indx.append([y_train[x] for x in indx[i]])


# In[33]:


print(sum([1 if (y_test[i] in n_indx[i]) else 0 for i in range(0, len(indx))])/float(len(indx)))

