#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
df = pd.read_csv('avg_crop_data.csv')


# In[8]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

df = pd.read_csv('avg_crop_data.csv')

numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),  
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', 'drop', categorical_cols)  
    ])

pca = PCA(n_components=2)  
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('pca', pca)])

df_pca = pipeline.fit_transform(df)
plt.figure(figsize=(10, 6))
plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.8)
plt.title('PCA of Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()


# In[11]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

df = pd.read_csv('avg_crop_data.csv')
numeric_cols = df.select_dtypes(include=['number']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

numeric_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=3, random_state=42))  
])

df['cluster'] = pipeline.fit_predict(df)

plt.figure(figsize=(10, 6))

for cluster in df['cluster'].unique():
    plt.scatter(df[df['cluster'] == cluster]['NuContAvailable'], 
                df[df['cluster'] == cluster]['AvMoisture%'], 
                label=f'Cluster {cluster}')

plt.title('KMeans Clustering of Data')
plt.xlabel('NuContAvailable')
plt.ylabel('AvMoisture%')
plt.legend()
plt.show()


# In[18]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('avg_crop_data.csv')

numeric_cols = df.select_dtypes(include=['number']).columns
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_cols])

#  Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

#  Visualize t-SNE
plt.figure(figsize=(10, 6))
plt.scatter(df_tsne[:, 0], df_tsne[:, 1], alpha=0.8)
plt.title('t-SNE Visualization of Data')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()


# In[19]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('avg_crop_data.csv')

plt.figure(figsize=(10, 6))
sns.scatterplot(x='YieldUnit', y='DryMatter%_NAS', hue='CropCategory', data=df, palette='viridis', alpha=0.8)
plt.title('Scatter Plot with Color')
plt.xlabel('YieldUnit')
plt.ylabel('DryMatter%_NAS')
plt.xticks(rotation = 45)
plt.legend(title='Category')
plt.show()


# In[23]:


import pandas as pd
import matplotlib.pyplot as plt

# Step 1: Load Data from CSV
df = pd.read_csv('avg_crop_data.csv')

# Step 2: Preprocess Data (if needed)
# Assuming 'NuContAvailable' and 'YieldUnitWeight(lb)_set' are numeric columns
# Convert them to numeric if necessary
df['NuContAvailable'] = pd.to_numeric(df['NuContAvailable'], errors='coerce')
df['YieldUnitWeight(lb)_set'] = pd.to_numeric(df['YieldUnitWeight(lb)_set'], errors='coerce')

# Step 3: Create Hexbin Plot
plt.figure(figsize=(10, 6))
plt.hexbin(x='NuContAvailable', y='YieldUnitWeight(lb)_set', data=df, gridsize=30, cmap='viridis')
plt.colorbar(label='Density')  # Add colorbar to show density scale
plt.title('Hexbin Plot of Data')
plt.xlabel('NuContAvailable')
plt.ylabel('YieldUnitWeight(lb)_set')
plt.show()


# In[ ]:




