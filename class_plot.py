import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def Main():
    cube = np.load("/home/compute2/Compute/Mokslai/Full-dataset/cube_2/data_1.npy")
    # clss = np.load("/home/compute2/Compute/Mokslai/Full-dataset/cube_2/clss.npy")
    shp = cube.shape
    cube = np.reshape(cube, (shp[0] * shp[1], shp[2]))
    # clss = np.reshape(clss, (shp[0] * shp[1])) 
    print(cube.shape)
    # print(clss.shape)

    pca = PCA(n_components=2)
    res = pca.fit_transform(cube)
    print(res.shape)
    np.save("pca_res.npy", res)
    plt.scatter(res[:,0], res[:,1], s=0.02)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two principal components")
    plt.savefig('class_pca.png', bbox_inches='tight', dpi=600)
    plt.clf()
    """
    transformer = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=50, n_jobs=50)
    res = transformer.fit_transform(cube)
    print(res.shape)
    np.save("tsne_res.npy", res)
    plt.scatter(res[:,0], res[:,1], c=clss, s=0.02)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("First two principal components")
    plt.savefig('class_ica.png', bbox_inches='tight', dpi=600)
    """


Main()
