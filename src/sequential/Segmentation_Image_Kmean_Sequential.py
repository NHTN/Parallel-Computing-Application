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
from argparse import ArgumentParser
parser = ArgumentParser()

@jit
def initialize_centroids(input_pixels, k, height, width):
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
    centroids_R, centroids_G,  centroids_B = [1] * k, [1] * k, [1] * k
    
    for i in range(k):
        random_pixel = input_pixels[randint(0, width)][randint(0, height)]
        centroids_R[i] = random_pixel[0]
        centroids_G[i] = random_pixel[1]
        centroids_B[i] = random_pixel[2]
    
    return centroids_R, centroids_G, centroids_B

def Mat_3D(k, height, width):
    return [[[1 for x in range(k)] for y in range(height)] for z in range(width)]

def choose_centroid(image, row, col, centroids):
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
    pixel = image[row][col]
    index = 0
    min_distance = 100*len(centroids)
    
    for i in range(len(centroids)):
        distance = (math.sqrt((math.pow(int(centroids[i][0])-int(pixel[0]),2))+(math.pow(int(centroids[i][1])-int(pixel[1]),2))+(math.pow(int(centroids[i][2])-int(pixel[2]),2))))
        if (distance < min_distance):
            min_distance = distance
            index = i
    return index

def rechoose_centroid(index_map, image, k, height, width):
    Centroids_R,Centroids_G, Centroids_B = [0] * k, [0] * k, [0] * k
    n_Centroids_R, n_Centroids_G, n_Centroids_B = [0] * k, [0] * k, [0] * k
    
    for row in range(height):
        for col in range(width):
            Centroids_R[index_map[row][col]] = Centroids_R[index_map[row][col]] + image[row][col][0]
            n_Centroids_R[index_map[row][col]] =  n_Centroids_R[index_map[row][col]] + 1

            Centroids_G[index_map[row][col]] = Centroids_G[index_map[row][col]] + image[row][col][1]
            n_Centroids_G[index_map[row][col]] =  n_Centroids_G[index_map[row][col]] + 1

            Centroids_B[index_map[row][col]] = Centroids_B[index_map[row][col]] + image[row][col][2]
            n_Centroids_B[index_map[row][col]] =  n_Centroids_B[index_map[row][col]] + 1
    
    for i in range(k):
        if(n_Centroids_R[i] == 0):
            n_Centroids_R[i] = n_Centroids_R[i] + 1
        if(n_Centroids_G[i] == 0):
            n_Centroids_G[i] = n_Centroids_G[i] + 1
        if(n_Centroids_B[i] == 0):
            n_Centroids_B[i] = n_Centroids_B[i] + 1
            
        Centroids_R[i] = Centroids_R[i]/n_Centroids_R[i]
        Centroids_G[i] = Centroids_G[i]/n_Centroids_G[i]
        Centroids_B[i] = Centroids_B[i]/n_Centroids_B[i]
    
    return Centroids_R,Centroids_G, Centroids_B

@jit
def SegmentationImage(image, k):

    height, width, channels = img.shape
    
    # Choose ramdom centroids
    Centroids_R, Centroids_G, Centroids_B = initialize_centroids(k, image, width, height)
    
    #First Centroids
    Centroids = [[1, 2, 3]] * k
    
    for i in range(k):
        Centroids[i] = [Centroids_R[i], Centroids_G[i], Centroids_B[i]]
        
    #Create map to save centroid indexs
    Map_Centroids = Mat_3D(k, width, height)
    Index_Map = Mat_3D(k, width, height)
    
    #KMeans
    for i in range(10):
        for row in range(height):
            for col in range(width):
                Index_Map[row][col] = choose_centroid(row, col, image, Centroids)
                Map_Centroids[row][col] = Centroids[Index_Map[row][col]]
            
        #Rechoose Centroids
        Centroids_R,Centroids_G, Centroids_B  = rechoose_centroid(Index_Map, image, k, height, width)
        
        for i in range(k):
            Centroids[i] = [Centroids_R[i], Centroids_G[i], Centroids_B[i]]
        
    return np.array(Map_Centroids)

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