import cv2
import time
from random import *
import numpy as np

def read_image(image_file, image_type):
  '''
    Read image

    Args: 
      image_file (str): The name of image file.
      image_type (int): The type of image. 0 is Grayscale image. 1 is Color image

    Returns: 
      int[[]]: Array of pixels of image. 
  '''
  pixels_array = cv2.imread('input.jpg', 0)
  return pixels_array

def initialize_centroids(input_pixels, k = 2):
  '''
    Random initialize for k clusters

    Args:
      input_pixels (int[[]]): The array of pixels of image.
      k (int): Integer value of k clusters.

    Returns:
      int[]: The vector clusters is initialized.
  '''
  (width_image, height_image) = input_pixels.shape

  random_column = randint(0, width_image - 1)
  random_row = randint(0, height_image - 1)
  
  centroids = []
  centroids.append(input_pixels[random_column, random_row])

  for i in range(k):
    while True:
      random_column = randint(0, width_image - 1)
      random_row = randint(0, height_image - 1)
      random_pixel = input_pixels[random_column, random_row]

      if (random_pixel not in centroids):
        break
      else:
        continue
  
    centroids.append(random_pixel)

  return centroids

def euclidean_distance(value_1, value_2):
  '''
    Calculate distance between 2 values by Euclidean formula.

    Args:
      value_1 (): The first value
      value_2 (): The second value
    
    Returns:
      (float): The value of distance between 2 values
  '''

  distance = np.linalg.norm(value_1 - value_2)
  return distance

def get_value(a, b):
    '''
      Calculate absolute distance between 2 integers

    Args:
      a (int[]): The first integer
      b (int): The second integer
    
    Returns:
      (int): The  absolute distance between 2 integers
  '''
  return abs(int(a)-int(b))

def get_best_centroid(centroids, pixel):
  '''
    Get the best centrer index which have the min distance to pixel

    Args:
      centroids (int[]): The centroids array
      pixel (int): The pixel of image
    
    Returns:
      (int): The index of the best center
  '''
  best_center_value = get_value(centroids[0], pixel)

  best_center_index = 0

  for i in range(1, len(centroids)):
    current_value = get_value(centroids[i], pixel)

    if (current_value < best_center_value):
      best_center_value = current_value
      best_center_index = i 
    else:
      continue

  return best_center_index

def recompute_centroids(centroid_map, input_pixels, k, centroids):
  '''
    Calculate average value of centroids

    Args:
      centroid_map (int[][]): The centroids array
      input_pixels (int[]): The pixels array of image
      k (int): The pixel of image
      centroids (int[])[]: The centroids array

    Returns:
      (int): The centroids array after recalculte value
  '''
  num_pixels_in_centroids = 0
  sum_pixels = 0

  (width, height) = input_pixels.shape

  for i in range(0, k):
    for row in range(0, height):
      for col in range(0, width):
        if (centroid_map[row][col] == centroids[i]):
          sum_pixels += input_pixels[row][col]
          num_pixels_in_centroids = num_pixels_in_centroids + 1
        else:
          continue

    if (sum_pixels == 0):
      centroids[i] = 0
    else:
      centroids[i] = round(sum_pixels / num_pixels_in_centroids)
  return centroids

def main():
  img = read_image('input.jpg', 1)
  clusters = initialize_centroids(img, 3)
  print(clusters)

  print(euclidean_distance(img[0][0], img[100][1]))

  print(get_best_centroid(clusters, img[0][0]))

main()