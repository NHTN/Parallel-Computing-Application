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
    
    print("--- Choose Time: %s seconds ---" % timing_choose)
    print("--- ReChoose Time: %s seconds ---" % timing_rechoose)
    return Map_Centroids

def main():
    
    parser.add_argument('filename', help="File Image Input")
    parser.add_argument('-k' , help="Number of Clusters")
    parser.add_argument('fileout', help="File Image Output")
    
    args = parser.parse_args()
    global img

    if('.mp4' in args.filename):
        count = 0
        vidcap = cv2.VideoCapture(args.filename)
        success,image = vidcap.read()
        success = True

        while success:
            vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*100))    # added this line 
            success,image = vidcap.read()
            try:
                cv2.imwrite("frame%d.jpg" % count, image)     # save frame as JPEG file
                count = count + 1
            except:
                break
        
    if(args.k == 'elbow'):
        n = 10
        SSE = [0] * n 

        print('--- Calculating Elbow Method ----')  
        temp = []
        
        if('.jpg' in args.filename):
            
            for i in range(n):
                print('--- Calculating SSE with k = {} ----'.format(i+1)) 
                SSE[i] = 0
                img = cv2.imread(args.filename)
                out_img = SegmentationImage(img,i + 1)
                img = cv2.imread(args.filename)

                for x in range(len(img)):
                    for y in range(len(img[0])):
                        SSE[i] = SSE[i] + math.sqrt(math.pow((int(img[x][y][0])-int(out_img[x][y][0])),2) + math.pow((int(img[x][y][1])-int(out_img[x][y][1])),2) + math.pow((int(img[x][y][2])-int(out_img[x][y][2])),2))

            for i in range(len(SSE) - 3):
                temp.append(abs(int(SSE[i] - SSE[i+1]) - int(SSE[i+1] - SSE[i+2]) - int(SSE[i+2] - SSE[i+3])))
            k = temp.index(min(temp)) + 1
            
        if('.mp4' in args.filename):
            
            for i in range(n):
                print('--- Calculating SSE with k = {} ----'.format(i+1)) 
                SSE[i] = 0
                img = cv2.imread('frame0.jpg')
                out_img = SegmentationImage(img,i + 1)
                img = cv2.imread('frame0.jpg')
                
                for x in range(len(img)):
                    for y in range(len(img[0])):
                        SSE[i] = SSE[i] + math.sqrt(math.pow((int(img[x][y][0])-int(out_img[x][y][0])),2) + math.pow((int(img[x][y][1])-int(out_img[x][y][1])),2) + math.pow((int(img[x][y][2])-int(out_img[x][y][2])),2))

            for i in range(len(SSE) - 3):
                temp.append(abs(int(SSE[i] - SSE[i+1]) - int(SSE[i+1] - SSE[i+2]) - int(SSE[i+2] - SSE[i+3])))
            k = temp.index(min(temp)) + 1
    
    else:
        k = int(args.k)
            
    print("--- Number of Cluster: {} ----".format(k))
    
    if('.jpg' in args.filename):
        
        print("--- Coverting the image ---")
        start_time = time.time()
        img = cv2.imread(args.filename)
        out_img = SegmentationImage(img, k)
        cv2.imwrite(args.fileout, out_img)

        print("--- %s seconds ---" % (time.time() - start_time))
        
    if('.mp4' in args.filename):
        
        start_time = time.time()
        print("---- Start Coverting ----")
        count = 1
        images = glob.glob("*.jpg")
        for i in images:
            if("frame" in i):
                print("Coverting {}/{}".format(count, len(images)))
                img = cv2.imread(i)
                out_img = SegmentationImage(img, k)
                cv2.imwrite(i.replace('frame',args.fileout), out_img)
                count = count + 1

        print("--- Done in %s seconds ---" % (time.time() - start_time))
        
        
        img_array = []
        height = 0
        width = 0
        images = glob.glob("*.jpg")
        for filename in images:
            if (args.fileout in filename):
                img = cv2.imread(filename)
                height, width, layers = img.shape
                img_array.append(img)

        fourcc = cv2.VideoWriter_fourcc(*'MPEG')
        out = cv2.VideoWriter('output.avi',fourcc, 10, (width,height))
        
        for i in range(len(img_array)):
            out.write(img_array[i])
        out.release()    
        
        for i in glob.glob("*.jpg"):
            os.remove(i)

if __name__ == "__main__":
    main()