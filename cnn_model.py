import numpy as np
import pandas as pd
import sys, time, os, warnings

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from pickle import dump
from collections import OrderedDict
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import metrics

from keras import models
from keras.applications import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input


def load_vgg16():
    """
    Load the vgg16 model
    """
    modelvgg = VGG16(include_top=True,weights=None)
    modelvgg.load_weights("C:/Users/Xpro/autocaption/output/vgg16_weights_tf_dim_ordering_tf_kernels.h5")
    # Exclude the last classification layer
    modelvgg.layers.pop()
    modelvgg = models.Model(inputs=modelvgg.inputs, outputs=modelvgg.layers[-1].output)
    modelvgg.summary()
    return modelvgg

def run_vgg16(dir_Flickr_jpg):
    """
    Generate the image features (4096 elements) from the VGG16 model without the last classification layer
    """
    modelvgg = load_vgg16()
    jpgs = os.listdir(dir_Flickr_jpg)
    images = OrderedDict()
    npix = 224
    target_size = (npix,npix,3)
    data = np.zeros((len(jpgs),npix,npix,3))
    for i,name in enumerate(jpgs):
        # load an image from file
        filename = dir_Flickr_jpg + '/' + name
        image = load_img(filename, target_size=target_size)
        # convert the image pixels to a numpy array
        image = img_to_array(image)
        nimage = preprocess_input(image)

        y_pred = modelvgg.predict(nimage.reshape( (1,) + nimage.shape[:3]))
        #expects 4 dimensions, sample, rows,columns,channel
        images[name] = y_pred.flatten()
        dump(images, open('C:/Users/Xpro/autocaption/testimage/images.pkl', 'wb'))

def link_text_image(df_txt0, images):
    """
    Link the text and image data
    """
    dimages, keepindex = [],[]
    #print(df_txt0.head())
    for i, fnm in enumerate(df_txt0.filename):
        if fnm in images.keys():
            dimages.append(images[fnm])
            keepindex.append(i)

    fnames = df_txt0["filename"].iloc[keepindex].values
    dcaptions = df_txt0["caption"].iloc[keepindex].values
    dimages = np.array(dimages)
    return fnames, dcaptions, dimages

def plot_elbow(X):
    """Plots elbow plot for optimal k
    Inputs
    ----------
    X: image features
    Ouputs
    -------
    Elbow plot
    """
    distortions = []
    K = range(1,10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k).fit(X)
        kmeanModel.fit(X)
        distortions.append(sum(np.min(cdist(X, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
    # Plot the elbow
    plt.plot(K, distortions, 'bx-')
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.show()

def plot_pca_2d(X):
    """Visualize the image features (4096 elements) generated from the VGG16 model in 2 dimentional spaces
    Inputs
    ----------
    X: image features in 2 dimentional space using PCA
    Outputs
    -------
    2d embedding plot of the image features
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)
    plt.figure(figsize=(8, 8), dpi=250, tight_layout=True)
    plt.style.use('dark_background')
    ax = plt.subplot(111)
    ax.axis('off')
    ax.patch.set_visible(False)
    kmeans = KMeans(n_clusters = 4)
    X_clustered = kmeans.fit_predict(X)
    for irow in range(X.shape[0]):
        ax.annotate(irow,X[irow,:],color=plt.cm.Set1(X_clustered[irow]),alpha=0.9)

    ax.set_xlabel("1st principal component",fontsize=15)
    ax.set_ylabel("2nd principal component",fontsize=15)
    plt.savefig('C:/Users/Xpro/autocaption/image/pca_embedding_plot_new.png',bbox_inches='tight')

def plot_pca_3d(X):
    """Visualize the image features (4096 elements) generated from the VGG16 model in 3 dimentional spaces
    Inputs
    ----------
    X: image features in 3 dimentional space using PCA
    Outputs
    -------
    3d embedding plot of the image features
    """
    x_min, x_max = np.min(X, 0), np.max(X, 0)
    X = (X - x_min) / (x_max - x_min)

    plt.figure(figsize=(8, 8), dpi=250, tight_layout=True)
    plt.style.use('dark_background')

    fig = plt.figure(1, figsize=(10, 10))
    ax = Axes3D(fig)
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False

    ax.axis('off')
    ax.patch.set_visible(False)

    kmeans = KMeans(n_clusters = 4)

    X_clustered = kmeans.fit_predict(X)

    for irow in range(X.shape[0]):
        ax.text3D(X[irow,0],X[irow,1],X[irow,2],irow,color=plt.cm.Set1(X_clustered[irow]),alpha=0.8)

    ax.set_xlabel("1st PC",fontsize=10)
    ax.set_ylabel("2nd PC",fontsize=10)
    ax.set_zlabel('3rd PC',fontsize=10)
    plt.savefig('C:/Users/Xpro/autocaption/image/pca_embedding_plot_3d.png',bbox_inches='tight')

    #rotate the axes and update
    for angle in range(0, 360, 2):
        ax.view_init(30, angle)
        plt.draw()
        plt.savefig('C:/Users/Xpro/autocaption/image/3d/3d_{0:0=3d}.png'.format(angle))

def plot_pca_image(picked_pic, dir_Flickr_jpg):
    '''Plot sample images from each cluster
    Inputs
    ----------
    picked_pic: the dictionary of the sample images
    dir_Flickr_jpg: the location of the images
    Outputs
    -------
    Plot of sample images from each cluster
    '''
    fnames=[]
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(18,6))
    count = 1
    target_size = (224,224,3)
    for color, irows in picked_pic.items():
        for ivec in irows:
            filename = dir_Flickr_jpg + '/' + fnames[ivec]
            image = load_img(filename, target_size=target_size)

            ax = fig.add_subplot(len(picked_pic),6,count,
                             xticks=[],yticks=[])
            count += 1
            plt.imshow(image)
            plt.title("{} ({})".format(ivec,color))
    plt.savefig('C:/Users/Xpro/autocaption/image/pca_image_new.png',bbox_inches='tight')

def plot_image_features(i, dir_Flickr_jpg):
        fnames,dimages,dcaptions=[]
        print(dcaptions[i])
    # Plot the i'th image from the test set
        target_size = (224,224,3)
        filename = dir_Flickr_jpg + '/' + fnames[i]
        image = load_img(filename, target_size=target_size)
        plt.imshow(image)
        plt.show()
        plt.savefig('C:/Users/Xpro/autocaption/image/8k/'+str(i)+'.png',bbox_inches='tight')
    # Transform the image features into an image
        img = dimages[i]
        img = img.reshape((64, 64))
        plt.imshow(img, interpolation='nearest', cmap='Reds')
        plt.show()
        plt.savefig('C:/Users/Xpro/autocaption/image/8k/'+str(i)+'_vgg16.png', bbox_inches='tight')

def main():
    dir_Flickr_jpg = "C:/Users/Xpro/autocaption/Flickr8k_Dataset/Flicker8k_Dataset/"
    #run_vgg16(dir_Flickr_jpg)
    images = pd.read_pickle('C:/Users/Xpro/autocaption/data/images.pkl', compression='infer')
    df_txt0 = pd.read_csv('C:/Users/Xpro/autocaption/data/token0.txt', sep='\t')
    fnames, dcaptions, dimages = link_text_image(df_txt0, images)

    #pca = PCA(n_components=3)
    #X_pca_3d = pca.fit_transform(dimages)
    #plot_elbow(X_pca_3d)
    #plot_pca_3d(X_pca_3d[:1000])

    #picked_pic = OrderedDict()
    #picked_pic["purple"] = [517, 644, 867, 225, 11, 128]
    #picked_pic["blue"] = [401,718,591,348,686, 47]
    #plot_pca_image(picked_pic, dir_Flickr_jpg)

if __name__ == '__main__':
    main()