# SinGAN
Pytorch implementation of "SinGAN: Learning a Generative Model from a Single Natural Image"

## Requirement
  * python 3.6
  * pytorch 1.0.0 or 1.1.0
  * torchvision 0.2.2 or 0.3.0
  * tqdm
  * scipy
  * PIL
  * opencv-python (cv2)
  
## Data Preparation
  * Download "monet2photo" dataset from https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/
  * Extract and rename "trainB" and "testB" to "trainPhoto" and "testPhoto", respectively. Then, place "trainPhoto" and "testPhoto" in "SinGANdata" folder

  * Directory should be like :
  ```
  Project
  |--- data
  |    |--- SinGANdata
  |             |--- trainPhoto
  |             |--- testPhoto
  |--- SinGAN
       |--- models
       |        |--- generator.py
       |        |--- ...
       |--- main.py 
       |--- train.py
       | ...
       
  ```
   * Then, an image in "trainPhoto" will be selected randomly for training.
   
  ## How to Run
  ### Train
   * 
  ### Test
