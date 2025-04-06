import numpy as np
from sklearn import preprocessing
from scipy.cluster.vq import kmeans, vq
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from model.Descriptor import VLADfun
from model.Descriptors import *
from sklearn.cluster import MiniBatchKMeans

class VLAD(object):
    def __init__(self, config,num_clusters=64) -> None:
        print('Initializing VLAD')

        self.visualDictionary = None
        self.tree = None
        self.k = num_clusters
        self.vlad = VLADfun(config)
        self.descriptor = "ORB"
        self.v = []
        self.idImages = []
    def fit(self,X, frm2features):
        if X is None:
            print("Error: Input X is None.")
            return
            # Perform k-means clustering to find the visual words
        print("Start k-means: %d words, %d key points" % (self.k, X.shape[0]))
        self.visualDictionary = KMeans(self.k, init='k-means++', tol=0.0001, max_iter=2, verbose=0).fit(X)



    def predict(self, frame_ids,frame_id,frm2features,local_feature,k = 10,step=15):
        ind_array = []
        V, idImages = self.vlad.getVLADDescriptors(frame_ids, frm2features, self.visualDictionary)
        self.tree = self.vlad.indexBallTree(V, 40)
        dist_temp, ind_temp = self.vlad.query(frame_id,local_feature, self.visualDictionary, self.tree,step,k)
        if ind_temp is not None:
            for i in ind_temp:
                ind_array.append(frame_ids[i].item())
            return ind_array, dist_temp
        else:
            return None, None



class BoW(object):
    def __init__(self, num_clusters=64) -> None:
        print('Initializing BoW')
        self.num_clusters = num_clusters
        self.visual_words = None  #有哪些聚类中心
        self.vw_variances = None  #存储视觉方差信息
        self.db_features = None   #每个地点帧每个词汇的数量
        self.idf = None

        self.frm_idx2id = {}#地点帧的索引


    def fit(self, X, frm2features): #得聚类中心仅由地点帧生成
        '''
        X: [N_KFs*N_samples, 32]
        frm2features: dict, {id: features}
        '''
        if X is None:
            print("Error: Input X is None.")
            return
        # Perform k-means clustering to find the visual words
        print("Start k-means: %d words, %d key points" %(self.num_clusters, X.shape[0]))
        self.visual_words, self.vw_variances = kmeans(X, self.num_clusters, 1)

        # Calculate the histogram of features
        print("Get histogram features")
        im_features = np.zeros((len(frm2features), self.num_clusters), "float32")
        for frm_idx, (frm_id, features) in enumerate(frm2features.items()):
            # store idx-id correspondence
            self.frm_idx2id[frm_idx] = frm_id
            # for each image feature, find its closest visual word (index) and increment count
            words, distance = vq(features, self.visual_words)
            for w in words:
                im_features[frm_idx][w] += 1

        # Perform L2 normalization
        print('l2 norm')
        self.db_features = preprocessing.normalize(im_features, norm='l2')


    def predict(self, X,frama_id,local_feature):
        '''
        X: [N_samples, 32]
        '''

        test_features = np.zeros((1, self.num_clusters), "float32")
        if np.isnan(local_feature).any() or np.isinf(local_feature).any():
            print("警告：检测到 NaN 或 inf 值，正在清理数据...")
            local_feature[np.isnan(local_feature)] = 0
            local_feature[np.isinf(local_feature)] = 0
        words, distance = vq(X, self.visual_words)
        for w in words:
            test_features[0][w] += 1

        # Perform Tf-Idf vectorization and L2 normalization
        hist_features = preprocessing.normalize(test_features, norm='l2')


        score_hist = np.dot(hist_features, self.db_features.T)  #无序
        rank_ID_hist = np.argsort(-score_hist) #有序
        rank_score_hist = score_hist[0][rank_ID_hist[0]]


        if abs(frama_id - self.frm_idx2id[rank_ID_hist[0][0]]) >= 30 and rank_score_hist[0] > 0.95:
            return self.frm_idx2id[rank_ID_hist[0][0]], rank_score_hist
        else:
            return -1, rank_score_hist

