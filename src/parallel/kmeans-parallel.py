import cv2
from random import *
import numpy as np
import math
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
   '''
    Random initialize for k clusters

    Args:
      input_pixels (int[[]]): The array of pixels of image.
      k (int): Integer value of k clusters.
      height (int): Height of input image.
      width (int): Width of input image.

    Returns:
      int[[][][]]: The vector clusters is initialized.
   '''
   Centroids_R,Centroids_G, Centroids_B = np.array([1] * k), np.array([1] * k), np.array([1] * k)
    
   for i in range(k):
      Centroids_R[i] = image[randrange(width)][randrange(height)][0]
      Centroids_G[i] = image[randrange(width)][randrange(height)][1]
      Centroids_B[i] = image[randrange(width)][randrange(height)][2]
    
   return Centroids_R, Centroids_G, Centroids_B

@cuda.jit
def Choose_Centroid(Index_Map, Map_Centroids, image, Centroids):
   '''
    Choose the index of pixel which have the minimum distance with centroids 

    Args:
      input_pixels (int[[]]): The array of pixels of image.
      k (int): Integer value of k clusters.
      height (int): Height of input image.
      width (int): Width of input image.

    Returns:
      int[[][][]]: The vector clusters is initialized.
   '''
   row, col = cuda.grid(2)
    
   if row < image.shape[0] and col < image.shape[1]:
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
        
@cuda.jit
def ReChoose_Centroid(index_map, image, k, Centroids_):
   '''
    Re choose the centroids which suitable

    Args:
      index_map (int[[]]): The array of pixels of image.
      image (int[][]): Input array 
      k (int): Integer value of k clusters.
      height (int): Height of input image.
      width (int): Width of input image.

    Returns:
      int[[][][]]: The vector of centroids.
   '''
   pos = cuda.grid(1)
    
   for row in range(image.shape[0]):
      for col in range(image.shape[1]):
         temp = index_map[row][col]
            
         if (pos < 3):
            Centroids_[pos][temp[0]] += image[row][col][pos]
         if (pos < 6 and pos >= 3):
            Centroids_[pos][temp[0]] += 1
       
   for i in range(k):
      print(Centroids_[0][i], Centroids_[3][i])
      Centroids_[0][i] = float(Centroids_[0][i]/Centroids_[3][i])
      Centroids_[1][i] = float(Centroids_[1][i]/Centroids_[4][i])
      Centroids_[2][i] = float(Centroids_[2][i]/Centroids_[5][i])
        

def SegmentationImage(image, k):
   '''
   Partitioned image into various subgroups (of pixels) 

   Args:
      image (int[][]): Input array 
      k (int): Integer value of k clusters

   Returns:
      Export output to image or video
   '''
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
#      print(Centroids)  
      ReChoose_Centroid[1, 6](Index_Map_global_mem, image_global_mem, k, Centroids__global_mem)
      Centroids_ = Centroids__global_mem.copy_to_host()    
      timing_rechoose = timing_rechoose + (time.time() - start_time2)
        
        
      for i in range(k):
         Centroids[i] = [Centroids_[0][i], Centroids_[1][i], Centroids_[2][i]]
#      print(Centroids)
        
      Map_Centroids = Map_Centroids_global_mem.copy_to_host().astype(int)
        
   print("--- Choose Time: %s seconds ---" % timing_choose)
   print("--- ReChoose Time: %s seconds ---" % timing_rechoose)
   return Map_Centroids

start_time = time.time()
out_img = SegmentationImage("images.jpg", 3)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imwrite('kmeans_parallel_test.jpg', out_img)