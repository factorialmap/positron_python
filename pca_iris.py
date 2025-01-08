from sklearn import datasets

#get data
data_iris = datasets.load_iris()

#check
data_iris.feature_names
data_iris.target_names
data_iris.data

#plots
import matplotlib.pyplot as plt
scatter_plot = plt.scatter(data_iris.data[:,0], data_iris.data[:,1], c = data_iris.target)
plt.xlabel(data_iris.feature_names[0])
plt.ylabel(data_iris.feature_names[1])
plt.legend(
    scatter_plot.legend_elements()[0],
    data_iris.target_names, 
    loc ="lower right",
    title = "Classes")

#plot pca
import mpl_toolkits.mplot3d

from sklearn.decomposition import PCA

plot = plt.figure(1, figsize=(10,10))
ax = plot.add_subplot(111, projection = "3d", elev = -150, azim = 110)

pca_iris = PCA(n_components=3).fit_transform(data_iris.data)
ax.scatter(
    pca_iris[:,0],
    pca_iris[:,1],
    pca_iris[:,2],
    c = data_iris.target,
    s = 40
)


#plot 3d pca iris
import plotly.express as px

df= px.data.iris()
x = df[["sepal_length", "sepal_width","petal_length","petal_width"]]

pca_iris = PCA(n_components = 3)
components_iris = pca_iris.fit_transform(x)

total_var = pca_iris.explained_variance_ratio_.sum()*100


fig = px.scatter_3d(
    components_iris, x =0, y =1, z = 2, color = df['species'],
    title=f'Total Explained Variance: {total_var:.2f}%',
    labels = {'0':'PC1', '1':'PC2','2':'PC3'}
)
fig
