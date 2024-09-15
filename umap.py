#unsupervised model and dim reduction umap

#package
import plotly.express as px

data_iris = px.data.iris()

#plot scatter matrix
features = ["sepal_length", "sepal_width", "petal_length","petal_width"]
fig = px.scatter_matrix(data_iris, dimensions = features, color = "species")
fig.show

#plot t-sne
from sklearn.manifold import TSNE
import plotly.express as px

data_iris = px.data.iris()

tsne_features = data_iris.loc[:,:"petal_width"]
tsne_features.head(3)

mdl_tsne = TSNE(n_components = 2, random_state=0)
projection = mdl_tsne.fit_transform(tsne_features)

#plot tsne
fig = px.scatter(
    projection,
    x=  0,
    y = 1,
    color = data_iris.species,
    labels = {'color': "species"}
)

fig


#package
import plotly.express as px

#get data
data_iris = px.data.iris()

#check some rows
data_iris.head(3)

#plot scatter matrix
features = ["sepal_length", "sepal_width", "petal_length","petal_width"]

#plot tsne 3D component = 3, axis x + y + z

#create a model
mdl_tsne_3d = TSNE(n_components = 3, random_state=0)
mdl_tsne_3d_projection = mdl_tsne_3d.fit_transform(tsne_features,)

#plot the model
fig_tsne_3d = px.scatter_3d(
    mdl_tsne_3d_projection, 
    x = 0,
    y = 1,
    z = 2,
    color= data_iris.species, 
    labels = {'color': 'species'}
)

fig_tsne_3d


#projections using UMAP ------------------------------------------------
from umap import UMAP
import plotly.express as px

data_iris = px.data.iris()
features_iris = data_iris.loc[:, :'petal_width']

#create umap 2d and 3d



-

#Resources -------------------------------------------------------------
#https://plotly.com/python/t-sne-and-umap-projections/
#https://umap-learn.readthedocs.io/en/latest/basic_usage.html
#https://umap-learn.readthedocs.io/en/latest/

