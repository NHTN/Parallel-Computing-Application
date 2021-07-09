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

def ConverVideoToFrames(video_path):
    count = 0
    n = 1000
    vidcap = cv2.VideoCapture(video_path)
    success,image = vidcap.read()
    success = True
    while success:
        vidcap.set(cv2.CAP_PROP_POS_MSEC,(count*75))    # added this line 
        success,image = vidcap.read()
        try:
            cv2.imwrite("frame%d.jpg" % n, image)     # save frame as JPEG file
            count = count + 1
            n = n + 1
        except:
            break
            
def ssim(img1, img2):
    
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()

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
    
def Find_Elbow():
    list_frame = []
    frame = sorted(glob.glob('*.jpg'))

    for i in frame:
        if('frame' in i):
            list_frame.append(Resize_Image(i))

    list_elbow = []
    for i in range(len(list_frame) - 1):
            if(ssim(list_frame[i], list_frame[i+1]) < 0.65):
                list_elbow.append(i+1)
    return list_elbow

def Merge_Frames_To_Video(file_output, fps=13):
    img_array = []
    height = 0
    width = 0
    images = glob.glob("*.jpg")
    for filename in images:
        if ('output' in filename):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            img_array.append(img)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(file_output, fourcc, fps, (width,height))
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release() 
    
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

def Clean_Images():
    for i in glob.glob("*.jpg"):
        if('frame' in i or 'output' in i):
            os.remove(i)

def main():
    
    parser.add_argument('filename', help="File Image Input")
    parser.add_argument('-k' , help="Number of Clusters")
    parser.add_argument('fileout', help="File Image Output")
    args = parser.parse_args()
    start_time = time.time() 
    
    if('.mp4' in args.filename):
        ConverVideoToFrames(args.filename)
        count = 0
        
        if(args.k == 'elbow'):
            
            k = Calc_K(Resize_Image('frame1000.jpg'))
            temp = Find_Elbow()
            print(temp)
            print("---- Number of Cluster: %s ------" %k)
            
            for i in sorted(glob.glob('*.jpg')):
                if('frame' in i):
                    cv2.imwrite(i.replace('frame','output'),SegmentationImage(cv2.imread(i), k))
                    print("------Complete {}------".format(count))
                    count = count + 1
                    if (count in temp):
                        k = Calc_K(Resize_Image('frame1000.jpg'))
                        print("---- Number of Cluster: %s ------" %k)
        else:
            k = int(args.k)
            for i in sorted(glob.glob('*.jpg')):
                if('frame' in i):
                    cv2.imwrite(i.replace('frame','output'),SegmentationImage(cv2.imread(i), k))
                    print("------Complete {}------".format(count))
                    count = count + 1
                    
        Merge_Frames_To_Video(args.fileout)
        Clean_Images()
        
    if('.jpg' in args.filename): 
        
        if(args.k == 'elbow'):
            k = Calc_K(Resize_Image(args.filename))
            print("---- Number of Cluster: %s ------" %k)
        else:  
            k = int(args.k)
            
        cv2.imwrite(args.fileout,SegmentationImage(cv2.imread(args.filename), k))
    print("--- %s seconds ---" % (time.time() - start_time))

if __name__ == "__main__":
    main()