import cv2
import os
from random import *
from numpy import *
import numpy as np
import math
import glob
from numba import jit, prange
import time
import warnings
from numba import cuda
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
from numba.typed import List
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

@cuda.jit
def Choose_Centroid(Index_Map, Map_Centroids, image, Centroids):
  
    row, col = cuda.grid(2)
    
    if row < image.shape[0] and col < image.shape[1]:
        pixel = image[row][col]
        index = 0
        temp = 999

        for i in range(len(Centroids)):
            distance = (math.sqrt((math.pow(int(Centroids[i][0])-int(pixel[0]),2))+(math.pow(int(Centroids[i][1])-int(pixel[1]),2))+(math.pow(int(Centroids[i][2])-int(pixel[2]),2))))
            if (distance <= temp):
                temp = distance
                index = i
        
        Index_Map[row][col][0] = index
        Map_Centroids[row][col][0] = Centroids[index][0]
        Map_Centroids[row][col][1] = Centroids[index][1]
        Map_Centroids[row][col][2] = Centroids[index][2]
        
@cuda.jit
def ReChoose_Centroid(index_map, image, k, Centroids_):
    pos = cuda.grid(1)
    
    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            temp = index_map[row][col]
            
            if (pos < 3):
                Centroids_[pos][temp[0]] += image[row][col][pos]
            if (pos < 6 and pos >= 3):
                Centroids_[pos][temp[0]] += 1   
            
def SegmentationImage(image, k):
    
    timing_choose = 0
    timing_rechoose = 0
    img = cv2.imread(image)
    height, width, channels = img.shape

    Centroids_R, Centroids_G, Centroids_B = Random_Centroids(k, img, width, height)

    #First Centroids
    Centroids = np.array([[1, 2, 3]] * k, np.int64)
    
    for i in range(k):
        Centroids[i] = [Centroids_R[i], Centroids_G[i], Centroids_B[i]]

    #KMeans
    for i in range(10):
        
        image_global_mem = cuda.to_device(img)
        Centroids_global_mem = cuda.to_device(Centroids)

        Map_Centroids_global_mem = cuda.device_array((height, width, 3))
        Index_Map_global_mem = cuda.device_array((height, width, 1))

        threadsperblock = (32,32)
        blockspergrid_x = int(math.ceil(height / threadsperblock[0]))
        blockspergrid_y = int(math.ceil(width / threadsperblock[1]))
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        
        start_time1 = time.time()
        Choose_Centroid[blockspergrid, threadsperblock](Index_Map_global_mem, Map_Centroids_global_mem, image_global_mem, Centroids_global_mem)
        timing_choose = timing_choose + (time.time() - start_time1)
        
        Index_Map = Index_Map_global_mem.copy_to_host().astype(int) 
    
        start_time2 = time.time()
        Index_Map_global_mem = cuda.to_device(Index_Map)
        
        Centroids_R,Centroids_G, Centroids_B = np.array([0] * k), np.array([0] * k), np.array([0] * k)
        n_Centroids_R, n_Centroids_G, n_Centroids_B = np.array([0] * k), np.array([0] * k), np.array([0] * k)
        
        Centroids_ = [Centroids_R,Centroids_G, Centroids_B, n_Centroids_R, n_Centroids_G, n_Centroids_B]
        Centroids__global_mem = cuda.to_device(Centroids_)
         
        ReChoose_Centroid[1, 6](Index_Map_global_mem, image_global_mem, k, Centroids__global_mem)
        Centroids_ = Centroids__global_mem.copy_to_host()    
        timing_rechoose = timing_rechoose + (time.time() - start_time2)
           
        Centroids_ = Centroids_.astype(np.float).astype("Int32")
        
        for i in range(k):
            for j in range(3,6):
                if(Centroids_[j][i] == 0):
                    Centroids_[j][i] = 1
          
            Centroids[i] = [Centroids_[0][i]/Centroids_[3][i], Centroids_[1][i]/Centroids_[4][i], Centroids_[2][i]/Centroids_[5][i]]
        
        Map_Centroids = Map_Centroids_global_mem.copy_to_host().astype(int)
        
    print("--- Choose Time: %s seconds ---" % timing_choose)
    print("--- ReChoose Time: %s seconds ---" % timing_rechoose)
    return Map_Centroids

@cuda.jit
def SSE_Map(out_img, img, SSE_map, n):
  
    row, col = cuda.grid(2)
    if row < n and col < len(img[0]):
        SSE_map[row][col] = math.sqrt(math.pow((float(img[row][col][0])-float(out_img[row][col][0])),2) + math.pow((float(img[row][col][1])-float(out_img[row][col][1])),2) + math.pow((float(img[row][col][2])-float(out_img[row][col][2])),2))

def Calculate_SSE(out_img, img):
    number_of_streams = 5

    segment_size = len(img) // number_of_streams

    stream_list = list()
    for i in range (0, number_of_streams):
        stream = cuda.stream()
        stream_list.append(stream)

    threads_per_block = (32,32)
    blockspergrid_x = int(math.ceil(len(img) / threads_per_block[0]))
    blockspergrid_y = int(math.ceil(len(img[0]) / threads_per_block[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    SSE_map  = np.full((segment_size, len(img[0])), 1, np.float)
    SSE_map_global_mem = cuda.to_device(SSE_map)
    SSE_list = []
    for i in range(0, number_of_streams):
        img_global_mem = cuda.to_device(img[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])
        out_img_global_mem = cuda.to_device(out_img[i * segment_size : (i + 1) * segment_size], stream=stream_list[i])

        SSE_Map[blockspergrid, threads_per_block, stream_list[i]](
                img_global_mem, 
                out_img_global_mem, 
                SSE_map_global_mem,
                segment_size)

        SSE_list.append(SSE_map_global_mem.copy_to_host(stream=stream_list[i]))
    cuda.synchronize()
        
    return np.reshape(SSE_list, (len(img)*len(img[0]), 1)).sum()

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

        start_time = time.time() 
        for i in range(n):

            out_img = SegmentationImage('test1.jpg',i + 1)
            img = cv2.imread('test1.jpg')

            SSE[i] = Calculate_SSE(out_img, img)
            print("%s / 10" %(i + 1))

        temp = []
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
        out_img = SegmentationImage(args.filename, k)
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