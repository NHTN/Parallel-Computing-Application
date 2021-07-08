import cv2
import os
from random import *
import numpy as np
import math
import glob
from numba import jit
import time
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from argparse import ArgumentParser
parser = ArgumentParser()


@jit(nopython=True)
def Random_Centroids(k, image, height, width):
    Centroids_R,Centroids_G, Centroids_B = np.array([1] * k), np.array([1] * k), np.array([1] * k)
    
    for i in range(k):
        Centroids_R[i] = image[randrange(width)][randrange(height)][0]
        Centroids_G[i] = image[randrange(width)][randrange(height)][1]
        Centroids_B[i] = image[randrange(width)][randrange(height)][2]
    
    return Centroids_R, Centroids_G, Centroids_B

@jit(nopython=True)
def Mat_3D(k, height, width):
    return np.zeros((width, height, k), np.int64)

@jit(nopython=True)
def Choose_Centroid(Index_Map, Map_Centroids, image, Centroids):
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            pixel = image[row][col]
            index = 0
            temp = 100*len(Centroids)

            for i in range(len(Centroids)):
                distance = (math.sqrt((math.pow(int(Centroids[i][0])-int(pixel[0]),2))+(math.pow(int(Centroids[i][1])-int(pixel[1]),2))+(math.pow(int(Centroids[i][2])-int(pixel[2]),2))))
                if (distance < temp):
                    temp = distance
                    index = i
        
            Index_Map[row][col][0] = index
            Map_Centroids[row][col][0] = Centroids[index][0]
            Map_Centroids[row][col][1] = Centroids[index][1]
            Map_Centroids[row][col][2] = Centroids[index][2]

@jit(nopython=True)
def ReChoose_Centroid(index_map, image, k, centroids):
    
    Centroids_R,Centroids_G, Centroids_B = np.array([1] * k), np.array([1] * k), np.array([1] * k)
    n_Centroids_R, n_Centroids_G, n_Centroids_B = np.array([1] * k), np.array([1] * k), np.array([1] * k)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            
            Centroids_R[index_map[row][col]] = Centroids_R[index_map[row][col]] + image[row][col][0]
            Centroids_G[index_map[row][col]] = Centroids_G[index_map[row][col]] + image[row][col][1]
            Centroids_B[index_map[row][col]] = Centroids_B[index_map[row][col]] + image[row][col][2]
            
            n_Centroids_R[index_map[row][col]] =  n_Centroids_R[index_map[row][col]] + 1
            n_Centroids_G[index_map[row][col]] =  n_Centroids_G[index_map[row][col]] + 1
            n_Centroids_B[index_map[row][col]] =  n_Centroids_B[index_map[row][col]] + 1
    
    for i in range(k):
        if(n_Centroids_R[i] == 0):
            n_Centroids_R[i] = n_Centroids_R[i] + 1
        if(n_Centroids_G[i] == 0):
            n_Centroids_G[i] = n_Centroids_G[i] + 1
        if(n_Centroids_B[i] == 0):
            n_Centroids_B[i] = n_Centroids_B[i] + 1
            
        
        Centroids_R[i] = float(Centroids_R[i]/n_Centroids_R[i])
        Centroids_G[i] = float(Centroids_G[i]/n_Centroids_G[i])
        Centroids_B[i] = float(Centroids_B[i]/n_Centroids_B[i])
        
    return Centroids_R,Centroids_G, Centroids_B

def SegmentationImage(image, k):
    
    timing_choose = 0
    timing_rechoose = 0
    height, width, channels = image.shape
    
    # Choose ramdom centroids
    Centroids_R, Centroids_G, Centroids_B = Random_Centroids(k, image, width, height)
    
    #First Centroids
    Centroids = np.array([[1, 2, 3]] * k, np.int64)
    
    for i in range(k):
        Centroids[i] = [Centroids_R[i], Centroids_G[i], Centroids_B[i]]
        
    #Create map to save centroid indexs
    Map_Centroids = Mat_3D(3, width, height)
    Index_Map = Mat_3D(1, width, height)
    
    #KMeans
    for i in range(10):
        
        start_time1 = time.time()
        Choose_Centroid(Index_Map, Map_Centroids, image, Centroids)
        timing_choose = timing_choose + (time.time() - start_time1)
        
        #Rechoose Centroids
        start_time2 = time.time()
        Centroids_R,Centroids_G, Centroids_B  = ReChoose_Centroid(Index_Map, image, k, Centroids)
        timing_rechoose = timing_rechoose + (time.time() - start_time2)
        
        for i in range(k):
            Centroids[i] = [Centroids_R[i], Centroids_G[i], Centroids_B[i]]
    
#     print("--- Choose Time: %s seconds ---" % timing_choose)
#     print("--- ReChoose Time: %s seconds ---" % timing_rechoose)
    return Map_Centroids

def Calculate_SSE(out_img, img):
    SSE_map  = np.full((img.shape[0], img.shape[1]), 1, np.float)
    
    for row in range(img.shape[0]):
        for col in range(img.shape[1]):
            SSE_map[row][col] = math.sqrt(math.pow((float(img[row][col][0])-float(out_img[row][col][0])),2) + math.pow((float(img[row][col][1])-float(out_img[row][col][1])),2) + math.pow((float(img[row][col][2])-float(out_img[row][col][2])),2))
    
    return np.reshape(SSE_map, (SSE_map.shape[0]*SSE_map.shape[1], 1)).sum()

def Resize_Image(orginal_image):

    orginal_image = cv2.imread(orginal_image)
    if(orginal_image.shape[1] > 500 and orginal_image.shape[0] > 300):
        scale_percent = 500 / orginal_image.shape[1]
        width = int(orginal_image.shape[1] * scale_percent)
        height = int(orginal_image.shape[0] * scale_percent)
        dim = (width, height)

        # resize image
        resized_image = cv2.resize(orginal_image, dim, interpolation = cv2.INTER_AREA)
        return resized_image
    else:
        return orginal_image

def Calc_K(img):
    n = 10
    SSE = [0] * n

    for i in range(n):

        out_img = SegmentationImage(img,i + 1)
        SSE[i] = Calculate_SSE(out_img, img)

        if(i > 2):
            for j in range(len(SSE) - 3):
                if(abs(int(SSE[j] - SSE[j+1]) / int(SSE[j+1] - SSE[j+2])) < 1.4):
                    return j + 1
    
def main():
    
    start_time = time.time()
    parser.add_argument('filename1', help="File Image Input 1")
    args = parser.parse_args()
    
    print(Calc_K(Resize_Image(args.filename1)))
    print("--- %s seconds ---" % (time.time() - start_time))
    
if __name__ == "__main__":
    main()