#!/usr/bin/env python
# coding: utf-8

# ## Regression Prediction and Customer Segmentation
# 
# Latar belakang kasus:
# Sebagai seorang Data Scientist di Kalbe Nutritional, tim inventory dan tim marketing memberikan project baru yaitu sebagai berikut.
# - Tim Inventory : membuat prediksi jumlah penjualan (quantity) dari total produk yang dijual Kalbe keseluruhan dengan tujuan membantu tim inventory untuk mengetahui perkiraan quantity produk yg terjual untuk memenuhi stok persediaan harian cukup
# - Tim Marketing : membuat customer segmentation untuk membantu tim marketing dalam memberikan personalized promotion dan sales treatment

# ### DATA PREPARATION

# In[19]:


#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# In[2]:


#load dataset
cust = pd.read_csv('customer.csv',sep=";")
prod = pd.read_csv('product.csv',sep=";")
store = pd.read_csv('store.csv',sep=";")
trans = pd.read_csv('transaction.csv',sep=";")


# In[3]:


cust


# In[4]:


prod


# In[5]:


store


# In[6]:


trans


# ### DATA PREPROCESSING

# In[7]:


# check info dari masing2 dataframe
print("DataFrame Customer info: ", cust.info())
print("DataFrame Product info: ", prod.info())
print("DataFrame Store info: ", store.info())
print("DataFrame Transaction info: ", trans.info())


# Terdapat 3 nilai missing value pada dataframe cust yaitu pada kolom Marital Status. Maka 3 nilai missing value akan diisi dengan nilai "Unknown"

# In[8]:


# fill missing value
cust['Marital Status'].fillna("Unknown", inplace=True)


# In[9]:


#check the missing value
cust.info()


# In[10]:


# change the format before convert to datetime
# convert to datatype
# trans['Date'] : obj to datetime
trans['Date'] = pd.to_datetime(trans['Date'], format = '%d/%m/%Y')


# In[11]:


trans.info()


# In[12]:


#merge all dataframe into 1
df = trans.merge(cust, on='CustomerID')                 .merge(store, on='StoreID')                 .merge(prod, on='ProductID')


# In[13]:


df


# In[14]:


# drop duplicate column
df.drop('Price_y',axis=1,inplace=True)
#rename column
df.rename(columns = {"Price_x":"price",
                    "Marital Status":"marital_status",
                    "Product Name":"product_name"}, inplace=True)


# In[15]:


#check df
df


# ### MACHINE LEARNING

# #### Regression (Time Series)

# Ketentuan Regression Time Series:
# 
# - Goal: prediksi total quantity harian dari produk yang dijual
# - Data merge untuk menggabungkan semua data
# - Data baru untuk regression: groupby by kolom Date yg diaggregasi SUM dari kolom qty
# - Output ada 365 rows
# - Metode time series ARIMA

# In[21]:


#plot the dataframe
plt.figure(figsize=(12, 6))
plt.plot(df_reg)
plt.title('Time Series Data')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.show()


# In[23]:


# Differencing the time series data
df_ts_diff = df_ts.diff(periods=1).dropna()

# Plot the differenced time series data
plt.figure(figsize=(12, 6))
plt.plot(df_ts_diff)
plt.title('Differenced Time Series Data (d=1)')
plt.xlabel('Date')
plt.ylabel('Qty')
plt.show()


# In[25]:





# In[ ]:





# In[16]:


# grouping data by date dan agregat kolom sum(qty)
reg_df = df.groupby('Date')['Qty'].sum()
reg_df


# In[17]:


# plot to line chart
#import viz lib
import matplotlib.pyplot as plt

plt.figure(figsize=(20,10))
plt.plot(reg_df)
plt.xlabel("Transaction Date")
plt.ylabel("Number of Quantity")
plt.show()


# In[ ]:





# In[25]:


# import regression lib
from sklearn.linear_model import LinearRegression
# import lib for time-series method
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller


# In[22]:


# choosing p,q,d for ARIMA model
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(reg_df)


# #### Clustering

# Ketentuan Clustering: 
# 
# - Goal: membuat cluster customer yang mirip
# - Membuat data baru untuk clustering (groupby by kolom customerID dgn kolom yg diagregasi yaitu COUNT(TransactionID), SUM(Qty), SUM(Total Amount))
# - Menggunakan metode clustering KMeans

# In[27]:


# new dataframe
clust_df = df.groupby('CustomerID').agg({'TransactionID':'count',
                                        'Qty': 'sum',
                                        'TotalAmount':'sum'}).reset_index()


# In[28]:


clust_df


# In[30]:


# X as feature
X = clust_df[['TransactionID', 'Qty', 'TotalAmount']]


# In[31]:


# standardize the X
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[34]:


# find the wcss
wcss = []

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)


# In[35]:


# visualize the 3 variables in scatter plot
plt.figure(figsize=(8, 5))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS (Within-Cluster Sum of Squares)')
plt.show()


# In[36]:


optimal_num_clusters = 3  # Adjust this based on your analysis

# Create the K-Means model with the chosen number of clusters
kmeans = KMeans(n_clusters=optimal_num_clusters, init='k-means++', random_state=42)

# Fit the K-Means model to the standardized data
kmeans.fit(X_scaled)


# In[37]:


# Add the cluster labels to the customer data
clust_df['Cluster'] = kmeans.labels_

# Display the resulting customer segments
print(clust_df)


# In[38]:


# Assess the quality of the clustering using silhouette score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f'Silhouette Score: {silhouette_avg}')

