#!/usr/bin/env python
# coding: utf-8

# In[1]:

# proses import library yang dibutuhkan
import re
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
plt.style.use("fivethirtyeight")
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go

except:
    #get_ipython().system('pip install plotly')
    import plotly.express as px
    import plotly.graph_objects as go


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

import scipy.cluster.hierarchy as sch
from sklearn. preprocessing import StandardScaler
from sklearn.cluster import KMeans 
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

try:
    from kneed import KneeLocator
except:
    #get_ipython().system('pip install kneed')
    from kneed import KneeLocator
#------------------------------------------------------------------
try:
    from yellowbrick.cluster import KElbowVisualizer
except:
    #get_ipython().system('pip install -U yellowbrick')
    from yellowbrick.cluster import KElbowVisualizer


# In[3]:

df = pd.read_csv(r"D:\Skripsi\jupy\dataset.csv", delimiter='|', quotechar='"', doublequote=True, on_bad_lines='skip')
df = df.sample(n=400000,random_state = 42)
df.info()


# In[4]:

df.head(10)

# In[5]:

df.drop(columns=['BRWICO', 'PORECO', 'RENAME','REATCO','PODISP','PODESC'], inplace=True)
df.head()

# In[6]:

df['NOMINAL'] = df['NOMINAL'].abs() #mengubah nilai pada kolom nominal menjadi nilai absolut karena sebelumnya nilainya minus
df.head()

# In[7]:


df.isnull().sum()


# In[8]:


df.duplicated().sum()


# In[8]:


df = df.drop_duplicates()


# In[9]:


df['PODTPO'] = pd.to_datetime(df['PODTPO'], format='%Y%m%d', errors='coerce').dt.date
df['CUDTLH'] = pd.to_datetime(df['CUDTLH'], format='%Y%m%d', errors='coerce').dt.date
df.head()


# In[10]:


#menghitung jumlah nasabah laki-laki dan perempuan serta lokasi customer
cat_columns = ['CUJEKL', 'CUADR3']
for col in cat_columns:
    print(f"Column: {col}")
    print(df[col].value_counts())
    print("\n")


# In[11]:


# Periksa nilai unik
unique_values = df['CUJEKL'].unique()
print("Nilai unik pada kolom 'CUJEKL':", unique_values)


# In[12]:


# Buat kondisi boolean untuk memeriksa apakah nilai 'L' atau 'P'
mask = (df['CUJEKL'] != 'L') & (df['CUJEKL'] != 'P')

# Hapus baris yang tidak memenuhi kondisi
df.drop(index=df[mask].index, inplace=True)

# Tampilkan nilai unik setelah penghapusan
print("Nilai unik setelah penghapusan:")
print(df['CUJEKL'].unique())

# Tampilkan DataFrame setelah penghapusan
print("\nDataFrame setelah penghapusan:")
print(df)


# In[13]:


df['CUDTLH'].value_counts() #Menghitung jumlah tanggal lahir nasabah


# In[14]:


pd.options.display.float_format = '{:.2f}'.format
df.describe()


# In[15]:


df['PODTPO'].value_counts() #Menghitung tanggal transaksi


# In[16]:


# Konversi kolom 'CUSTLAHIR' ke tipe data Timestamp jika belum
df['CUDTLH'] = pd.to_datetime(df['CUDTLH'])

# Hitung perbedaan hari antara tanggal hari ini dan tanggal lahir pelanggan
df['UMUR'] = (pd.Timestamp('today') - df['CUDTLH']).dt.days

# Konversi hari ke tahun
df['UMUR'] = (df['UMUR'] / 365).round(0)

df.head()


# In[17]:


df['UMUR'].describe() #untuk mengetahui data dari umur nasabah


# In[18]:


df1 = df.copy() #mengkopi dataframe yang telah selesai proses cleaning dan menyimpannya dengan nama df1


# In[19]:


# 1. distribusi nasabah berdasarkan jenis kelamin
plt.figure(figsize=(6, 4))
sns.countplot(x='CUJEKL',data = df1, palette='pastel')
plt.title('Distribusi Nasabah Berdasarkan Jenis Kelamin')
plt.show()

# 2. Distribusi nasabah berdasarkan umur
plt.figure(figsize=(10, 6))
sns.histplot(df1['UMUR'], bins=30, kde=True, color='skyblue')
plt.title('Distribusi Nasabah Berdasarkan Umur')
plt.xlabel('Umur')
plt.ylabel('Jumlah Nasabah')
plt.show()

# 3. distribusi nasabah berdasarkan lokasi atau kota
location_counts = df1['CUADR3'].value_counts().nlargest(10)
plt.figure(figsize=(12, 6))
sns.barplot(x=location_counts.index, y=location_counts.values, palette='viridis')
plt.title('10 lokasi teratas berdasarkan customer')
plt.xlabel('Kota')
plt.ylabel('Jumlah Customer')
plt.xticks(rotation=45, ha='right')
plt.show()


# In[20]:


df2 = df1.copy() #mengkopi kembali dataframe df1 menjadi df2 untuk melakukan proses analisa terhadao data transaksi nasabah


# In[21]:


def handle_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    data[column] = data[column].apply(lambda x: upper_bound if x > upper_bound else lower_bound if x < lower_bound else x)
    return data

# mengatasi outliers for SALDO dan NOMINAL tranksaksi
df1 = handle_outliers_iqr(df1, 'SALDO')
df1 = handle_outliers_iqr(df1, 'NOMINAL')

# Distribusi atas Saldo nasabah (SALDO) pada nasabah
plt.figure(figsize=(12, 6))
sns.histplot(df1['SALDO'], kde=True, color='skyblue')
plt.title('Distribusi atas Saldo nasabah pada nasabah (After Outlier Handling)')
plt.xlabel('SALDO (IDR)')
plt.ylabel('Jumlah nasabah')
plt.show()

# Distribusi dari nominal transaksi (NOMINAl) pada nasabah
plt.figure(figsize=(12, 6))
sns.histplot(df1['NOMINAL'], kde=True, color='lightcoral')
plt.title('Distribusi dari nominal transaksi pada nasabah (After Outlier Handling)')
plt.xlabel('Nominal (IDR)')
plt.ylabel('Jumlah Nasabah')
plt.show()

# korelasi antara saldo nasabah dan nominal transaksi nasabah
plt.figure(figsize=(8, 6))
sns.scatterplot(x='SALDO', y='NOMINAL', data=df1, color='green', alpha=0.6)
plt.title('korelasi antara saldo nasabah dan nominal transaksi nasabah')
plt.xlabel('SALDO')
plt.ylabel('NOMINAL')
plt.show()


# In[22]:


df1[["NOMINAL", "SALDO"]].describe()


# In[23]:


top_10_locations = df1['CUADR3'].value_counts().nlargest(10)

#memvisualisasikan distribusi customer pada 10 lokasi teratas
plt.figure(figsize=(12, 6))
sns.barplot(x=top_10_locations.index, y=top_10_locations.values, palette='coolwarm')
plt.title('Distribusi Nasabah berdasarkan lokasi 10 tertinggi')
plt.xlabel('Lokasi / Kota')
plt.ylabel('Jumlah Nasabah')
plt.xticks(rotation=45)
plt.show()

location_transaction_volumes = df1.groupby('CUADR3')['NOMINAL'].sum().nlargest(10).sort_values(ascending = False)

# Visualize the transaction volumes for each location using a bar plot
#visualisasi dari volume transaksi setiap lokasi
plt.figure(figsize=(12, 6))
sns.barplot(x=location_transaction_volumes.index, y=location_transaction_volumes.values, palette='coolwarm')
plt.title('Volume Transaksi untuk lokasi yang berbeda')
plt.xlabel('Lokasi / Kota')
plt.ylabel('Total Jumlah Nominal Transaksi (IDR)')
plt.xticks(rotation=45)
plt.show()


# In[24]:


df1['PODTPO'] = pd.to_datetime(df1['PODTPO'])

current_date = df1['PODTPO'].max()
rfm_data = df1.groupby('CUCODE').agg({
    'PODTPO': lambda x: (current_date - x.max()).days,  # Recency calculation
    'POREFN': 'count',  # Frequency calculation
    'NOMINAL': 'sum'  # Monetary calculation
})
# Recency (R): jumlah Hari nasabah melakukan transaksi kembali
# Frequency (F): jumlah transaksi yang dilakukan oleh setiap nasabah
# Monetary (M): jumlah nominal yang ditransaksikan nasabah

rfm_data.rename(columns={
    'PODTPO': 'Recency',
    'POREFN': 'Frequency',
    'NOMINAL': 'Monetary'
}, inplace=True)


# In[25]:


# Visualize the distributions of RFM features
plt.figure(figsize=(12, 6))
sns.histplot(rfm_data['Recency'], bins=50, kde=True, color='purple')
plt.title('Distribusi dari Recency')
plt.xlabel('Recency (Hari)')
plt.ylabel('Jumlah Nasabah')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(rfm_data['Frequency'], bins=50, kde=True, color='orange')
plt.title('Distribusi dari Frequency')
plt.xlabel('Frequency (Number of Transactions)')
plt.ylabel('Jumlah Nasabah')
plt.show()

plt.figure(figsize=(12, 6))
sns.histplot(rfm_data['Monetary'], bins=50, kde=True, color='green')
plt.title('Distribusi dari Monetary')
plt.xlabel('Monetary (Total Jumlah Tranasksi in IDR)')
plt.ylabel('Jumlah Nasabah')
plt.show()


# In[26]:


rfm_data.describe()


# In[27]:

df2.head()


# In[28]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
  
df2['CUJEKL']= le.fit_transform(df2['CUJEKL'])
df2['CUADR3']= le.fit_transform(df2['CUADR3'])
df2.head()


# In[29]:


df2 = df2.drop(['CUCODE','CUDTLH','POREFN','PODTPO'],axis=1)
df2.head()


# In[30]:


df2[df2['NOMINAL'] == 0].count()


# In[31]:


from sklearn.preprocessing import StandardScaler

columns_names=['CUJEKL','CUADR3','NOMINAL','SALDO','UMUR']
s = StandardScaler()
df2 = s.fit_transform(df2)
df2 = pd.DataFrame(df2,columns=columns_names)
df2.head()


# In[32]:


import random
from yellowbrick.cluster import KElbowVisualizer
from sklearn.cluster import KMeans


# In[33]:


# mendefinisikan ilai X dan Y dari dataframe yang telah kita tentukan
X = df2[['UMUR', 'NOMINAL']].values
Y = df2[['CUADR3', 'NOMINAL']].values

# menjalankan K-Means Clustering dan menghitung jumlah K optimal menggunakan Elbow Method
def perform_elbow_method(X, k_range, title):
    model = KMeans(init='k-means++', random_state=42)
    visualizer = KElbowVisualizer(model, k=k_range, timings=False)
    visualizer.fit(X)
    plt.title(title)
    visualizer.show()
    
perform_elbow_method(X, k_range=(2, 20), title='Elbow Method untuk Clustering berdasarkan Umur Nasabah')
perform_elbow_method(Y, k_range=(2, 20), title='Elbow Method untuk Clustering berdasarkan Lokasi Nasabah')


# In[34]:


# Function to perform KMeans clustering and return the cluster labels and centroids
def perform_kmeans_clustering(X, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=42)
    cluster_labels = kmeans.fit_predict(X)
    cluster_centers = kmeans.cluster_centers_
    return cluster_labels, cluster_centers

x_cluster_labels, x_cluster_centers = perform_kmeans_clustering(X, num_clusters=7)
y_cluster_labels, y_cluster_centers = perform_kmeans_clustering(Y, num_clusters=6)


# In[35]:


print("x_cluster_labels",x_cluster_labels)
print("x_cluster_centers",x_cluster_centers)
print("y_cluster_labels",y_cluster_labels)
print("y_cluster_centers",y_cluster_centers)


# In[37]:


# Add cluster labels to the DataFrame
df2['cluster_umur'] = x_cluster_labels
df2['cluster_lokasi'] = y_cluster_labels

# Add centroids to the DataFrame
df2['cen_xx'] = df2.cluster_umur.map({i: x_cluster_centers[i][0] for i in range(7)})
df2['cen_xy'] = df2.cluster_umur.map({i: x_cluster_centers[i][1] for i in range(7)})
df2['cen_yx'] = df2.cluster_lokasi.map({i: y_cluster_centers[i][0] for i in range(6)})
df2['cen_yy'] = df2.cluster_lokasi.map({i: y_cluster_centers[i][1] for i in range(6)})

df2.head()


# In[38]:


# Define colors for each cluster
colors_X = ['red', 'yellow', 'grey', 'green','blue','orange','pink']
colors_Y = ['red', 'yellow', 'grey', 'green','blue','orange']

# Add cluster colors to the DataFrame
df2['color_umur_km'] = df2.cluster_umur.map({i: colors_X[i] for i in range(7)})
df2['color_lokasi_km'] = df2.cluster_lokasi.map({i: colors_Y[i] for i in range(6)})


# In[39]:


# Plot the scatter plot with cluster colors and centroids for 'Umur' dan 'Nominal Transaksi'
plt.figure(figsize=(20, 10))
plt.subplot(1, 2, 1)
plt.scatter(df2['UMUR'], df2['NOMINAL'], c=df2.color_umur_km)
plt.scatter(df2['cen_xx'], df2['cen_xy'], marker='*', s=500, c='black')
plt.ylim([-0.5, 20])
plt.xlabel('Umur Nasabah')
plt.ylabel('Nominal Transaksi')
plt.title('Clustering Umur Nasabah')

# Plot the scatter plot with cluster colors and centroids for 'Lokasi' dan 'Nominal Transaksi'
plt.subplot(1, 2, 2)
plt.scatter(df2['CUADR3'], df2['NOMINAL'], c=df2.color_lokasi_km)
plt.scatter(df2['cen_yx'], df2['cen_yy'], marker='*', s=500, c='black')
plt.ylim([-0.5, 20])
plt.xlabel('Lokasi Nasabah')
plt.ylabel('Nominal Transaksi')
plt.title('Clustering Lokasi Nasabah')

plt.tight_layout()  # Adjusts the spacing between the plots
plt.show()


# In[41]:


# Sampling data (misalnya 10% dari data asli)
#sample_fraction = 0.1
#df2_sample = df2.sample(frac=sample_fraction, random_state=42)

# Mengubah tipe data untuk menghemat memori
#df2_sample['NOMINAL'] = df2_sample['NOMINAL'].astype('float32')
#df2_sample['UMUR'] = df2_sample['UMUR'].astype('float32')

# Jika CUADR3 adalah kategori, ubah menjadi tipe 'category'
#df2_sample['CUADR3'] = df2_sample['CUADR3'].astype('category').cat.codes

# Silhouette Score for cluster_umur (sampled data)
#silhouette_score_umur = silhouette_score(df2_sample[['NOMINAL', 'UMUR']], df2_sample['cluster_umur'])
#print("Silhouette Score for cluster_umur (sampled data):", silhouette_score_umur)

# Silhouette Score for cluster_lokasi (sampled data)
#silhouette_score_lokasi = silhouette_score(df2_sample[['CUADR3','NOMINAL']], df2_sample['cluster_lokasi'])
#print("Silhouette Score for cluster_lokasi (sampled data):", silhouette_score_lokasi)


# In[42]:


# Sampling data (misalnya 30% dari data asli)
#sample_fraction = 0.3
#df2_sample = df2.sample(frac=sample_fraction, random_state=42)

# Mengubah tipe data untuk menghemat memori
#df2_sample['NOMINAL'] = df2_sample['NOMINAL'].astype('float32')
#df2_sample['UMUR'] = df2_sample['UMUR'].astype('float32')

# Jika CUADR3 adalah kategori, ubah menjadi tipe 'category'
#df2_sample['CUADR3'] = df2_sample['CUADR3'].astype('category').cat.codes

# Silhouette Score for cluster_umur (sampled data)
#silhouette_score_umur = silhouette_score(df2_sample[['NOMINAL', 'UMUR']], df2_sample['cluster_umur'])
#print("Silhouette Score for cluster_umur (sampled data):", silhouette_score_umur)

# Silhouette Score for cluster_lokasi (sampled data)
#silhouette_score_lokasi = silhouette_score(df2_sample[['CUADR3','NOMINAL']], df2_sample['cluster_lokasi'])
#print("Silhouette Score for cluster_lokasi (sampled data):", silhouette_score_lokasi)


# In[43]:

df2.head()


# In[44]:

print(df2.columns)

# In[45]:

print(df2.dtypes)

# In[46]:

# Seleksi kolom numerik
numeric_columns = df2.select_dtypes(include=[np.number]).columns

# Menghapus kolom non-numerik dari grup
columns_to_group = ['cluster_umur', 'cluster_lokasi']
columns_to_average = [col for col in numeric_columns if col not in columns_to_group]

# In[47]:

# Grup berdasarkan `cluster_umur` dan hitung rata-rata
umur_cluster_avg = df2.groupby('cluster_umur')[columns_to_average].mean().reset_index()

# Grup berdasarkan `cluster_lokasi` dan hitung rata-rata
lokasi_cluster_avg = df2.groupby('cluster_lokasi')[columns_to_average].mean().reset_index()


# In[48]:

print("Cluster_Umur")
print(umur_cluster_avg[['CUADR3','SALDO','NOMINAL','UMUR']])
print("\n")
print("Cluster_Location")
print(lokasi_cluster_avg[['CUADR3','SALDO','NOMINAL','UMUR']])

# In[49]:

# Count the number of customers in each 'cluster_umur_km' and 'cluster_lokasi_km'
umur_cluster_counts = df2['cluster_umur'].value_counts().reset_index()
umur_cluster_counts.columns = ['Cluster', 'Count']

lokasi_cluster_counts = df2['cluster_lokasi'].value_counts().reset_index()
lokasi_cluster_counts.columns = ['Cluster', 'Count']

# Function to plot cluster distribution
def plot_cluster_distribution(cluster_counts, title):
    plt.figure(figsize=(8, 5))
    sns.barplot(x='Cluster', y='Count', data=cluster_counts, palette='viridis')
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('jumlah nasabah')
    plt.show()


plot_cluster_distribution(umur_cluster_counts, title='Customer Distribution across umur Clusters')
plot_cluster_distribution(lokasi_cluster_counts, title='Customer Distribution across lokasi Clusters')
