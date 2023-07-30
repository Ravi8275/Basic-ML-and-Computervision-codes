#!/usr/bin/env python
# coding: utf-8

# <span style="font-size: larger;">**Importing all the basic libraries**.</span>

# In[1]:


import numpy as np


# In[2]:


import seaborn as sb


# In[3]:


import pandas as pd


# In[4]:


import matplotlib.pyplot as plt


# In[5]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


cdata=pd.read_csv('Customers.csv')


# In[7]:


cdata


# In[8]:


cdata.info()


# In[15]:


nullrows=cdata[cdata.isnull().any(axis=1)]
print(nullrows)


# <span style="font-size: larger;">**Data Preprocessing**.</span>

# In[16]:


nullrows.info()


# In[17]:


finald=cdata.dropna()


# In[23]:


checkfornulls=finald[finald.isnull().any(axis=1)]


# In[24]:


checkfornulls


# In[25]:


finald.head()


# <span style="font-size: larger;">**Data Visualization**.</span>

# In[278]:


plt.figure(figsize=(10,5))
sb.countplot(x='Gender',data=finald,hue='Profession')
hue_legend=plt.legend(title='Profession',loc='upper left')


# In[133]:


plt.figure(figsize=(10,5))
sb.boxplot(x='Profession',y='Spending Score (1-100)',data=finald)


# In[193]:


finald.head()


# <span style="font-size: larger;">**Reordering the columns**.</span>

# In[279]:


finald=finald[['CustomerID','Age','Spending Score (1-100)','Annual Income ($)','Profession','Work Experience','Gender','Family Size','clusterdata']]


# In[280]:


finald.head()


# <span style="font-size: larger;">**Feature selection(it's subjective and done accroding to our target criteria)**.</span>

# In[281]:


x=finald.iloc[:,1:3]


# In[282]:


x


# <span style="font-size: larger;">**Importing kMeans library and preprocessing libraries for standardization**.</span>

# In[359]:


from sklearn.cluster import KMeans


# In[360]:


from sklearn import preprocessing


# <span style="font-size: larger;">**The no:of clusters is selected by elbow method which is performed below**.</span>

# In[365]:


km=KMeans(10)


# In[366]:


x_scaled=preprocessing.scale(x)


# In[367]:


clusters=km.fit_predict(x_scaled)


# <span style="font-size: larger;">**Wcss value as per the above given no:of clusters**.</span>

# In[368]:


km.inertia_


# <span style="font-size: larger;">**Wcss value can be minimized a bit more in general**.</span>

# In[369]:


finald['clusterdata']=clusters


# In[370]:


finald


# In[373]:


plt.figure(figsize=(12,5))
plt.scatter(finald['Age'],finald['Spending Score (1-100)'],c=finald['clusterdata'],cmap='rainbow')
plt.xlabel('Age')
plt.ylabel('Spending score')


# <span style="font-size: larger;">**choosing k value using elbow method**.</span>

# In[374]:


wcss=[]
for i in range(1,11):
    km=KMeans(i)
    km.fit(x_scaled)
    wcss_new=km.inertia_
    wcss.append(wcss_new)


# In[375]:


wcss


# In[377]:


no=range(1,11)
plt.plot(no,wcss)
plt.title('Elbow Method Graph to determine K value')
plt.xlabel('No:of clusters')
plt.ylabel('wcss')


# In[ ]:




