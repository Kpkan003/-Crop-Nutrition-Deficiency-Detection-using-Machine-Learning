# -Crop-Nutrition-Deficiency-Detection-using-Machine-Learning
This project uses Machine Learning and data visualization to detect nutrient deficiencies in crops from avg_crop_data.csv. Using PCA, K-Means, and t-SNE, it identifies patterns and clusters in crop data. Built with Python, it provides visual insights through plots to help improve crop productivity and nutrient management.

Overview

This project applies Machine Learning and Data Visualization techniques to analyze agricultural crop data and identify potential nutrient deficiencies.
By using PCA, K-Means clustering, t-SNE, and visual analytics, the project provides insights into crop nutritional conditions and their relationships with environmental factors.

Objectives

Identify patterns and clusters related to crop nutrition.

Visualize relationships between soil nutrients, moisture, and yield.

Reduce data dimensionality for better visualization and interpretation.

Detect groups of crops that may share similar nutritional deficiencies.

Tools and Technologies
Category	Tools / Libraries
Programming Language	Python
Data Handling	Pandas
Visualization	Matplotlib, Seaborn
Machine Learning	Scikit-learn (PCA, KMeans, t-SNE)
Data Source	avg_crop_data.csv
🧩 Methodology
1. Data Preprocessing

Imported the dataset avg_crop_data.csv using pandas.

Separated numerical and categorical columns.

Applied SimpleImputer (median strategy) for missing numeric values.

Standardized features using StandardScaler.

Dropped categorical data where not needed for numerical analysis.

2. Principal Component Analysis (PCA)

Reduced dataset to 2 principal components.

Visualized the distribution of data in reduced space to identify major patterns.

Helped understand feature variance and overall data structure.

pca = PCA(n_components=2)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('pca', pca)])

3. K-Means Clustering

Applied K-Means with 3 clusters to group similar crop patterns.

Preprocessed both numerical and categorical data using ColumnTransformer.

Visualized clusters using features like NuContAvailable and AvMoisture%.

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('kmeans', KMeans(n_clusters=3, random_state=42))])

4. t-SNE Visualization

Used t-SNE (t-Distributed Stochastic Neighbor Embedding) for non-linear dimensionality reduction.

Projected high-dimensional crop features into 2D space for better cluster visualization.

tsne = TSNE(n_components=2, random_state=42)
df_tsne = tsne.fit_transform(df_scaled)

5. Visual Analysis

Created scatter plots and hexbin plots to explore feature relationships:

Scatter Plot: Yield vs. Dry Matter% (colored by Crop Category).

Hexbin Plot: Density between Nutrient Content Available and Yield Weight.

Helped detect nutritional imbalance and yield variation patterns visually.

 Visualizations Included

PCA Plot: Visualizing data structure and feature variance.

K-Means Clusters: Grouping of similar crops by nutrient properties.

t-SNE Plot: 2D mapping of high-dimensional nutrient data.

Scatter Plot (Seaborn): Relationship between yield and dry matter across categories.

Hexbin Plot: Density visualization for nutrient vs. yield relationship.

 Insights

Crops group naturally based on nutrient composition and moisture.

Certain clusters indicate potential nutrient deficiencies affecting yield.

PCA and t-SNE help in identifying feature relationships and data separability.

Visual trends assist in agricultural planning and fertilizer optimization.

 Key Python Concepts Used

Pipelines & ColumnTransformer – for streamlined preprocessing and model chaining.

Feature Scaling & Imputation – to handle missing data and normalize features.

Clustering & Dimensionality Reduction – for uncovering hidden structures in data.

Advanced Visualization – to interpret model results effectively.

 Conclusion

This project demonstrates how Machine Learning and Data Visualization can be combined to analyze agricultural datasets and detect crop nutrition deficiencies.
By applying clustering, dimensionality reduction, and visual analysis, it provides actionable insights for improving crop health and productivity.
