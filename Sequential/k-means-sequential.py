import cv2
import time
from random import *

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
    Calculate distance between value of 2 values by Euclidean formula.

    Args:
      value_1 (): The first value
      value_2 (): The second value
    
    Returns:
      (float): The value of distance between 2 values
  '''

  distance = np.linalg.norm(value_1 - value_2)
  return distance

def main():
  img = read_image('input.jpg', 1)
  clusters = initialize_centroids(img, 3)
  print(clusters)

  print(euclidean_distance(img[0][0], img[100][1]))

main()