import numpy as np
import itertools
from sklearn.cluster import KMeans
from sklearn.neighbors import BallTree
import pickle
import os
import glob
import cv2
from model.Descriptors import describeORB,describeSIFT,describeSURF
from datasets.dataset import get_dataset



class VLADfun():
    def __init__(self, config):
        self.dataset = get_dataset(config)
        self.threshold = 0.1

    def getVLADDescriptors(self,frame_id,descriptor, visualDictionary):
        descriptors = list()
        idImage = list()

        for i in frame_id:

            des = descriptor[i.item()]

            if des is not None:
                v = self.VLAD(des, visualDictionary)
                descriptors.append(v)

        descriptors = np.asarray(descriptors)

        return descriptors, idImage

    def query(self,image,descriptor, visualDictionary,tree,step,k):
        np.set_printoptions(threshold=np.inf)

        if descriptor is not None:

            if np.isnan(descriptor).any():
                print("NaN values found in descriptor. Removing NaN values...")
                descriptor = np.nan_to_num(descriptor)

            v = self.VLAD(descriptor, visualDictionary)


            result_dict = {}
            dist, ind = tree.query(v.reshape(1, -1), tree.data.shape[0])

            dist = dist[0]
            ind = ind[0]

            for i in range(len(ind)):
                result_dict[ind[i]] = dist[i]
            return dist, ind
        else:
            return None,None



    def VLAD(self,X, visualDictionary):
        predictedLabels = visualDictionary.predict(X)
        centers = visualDictionary.cluster_centers_

        k = 64
        m, d = X.shape
        V = np.zeros([k, d])
        for i in range(k):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels == i) > 0:
                # add the diferences
                V[i] = np.sum(X[predictedLabels == i, :] - centers[i], axis=0)

        V = V.flatten()
        # power normalization, also called square-rooting normalization
        V = np.sign(V) * np.sqrt(np.abs(V))

        # L2 normalization

        V = V / np.sqrt(np.dot(V, V))
        return V


    def indexBallTree(self,X,leafSize):
        tree = BallTree(X, leaf_size=leafSize)
        return tree
