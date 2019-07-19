# SinGAN: Learning a Generative Model from a Single Natural Image
Pytorch implementation of "SinGAN: Learning a Generative Model from a Single Natural Image" 
([arxiv](https://arxiv.org/abs/1905.01164))

This implementation is based on these repos.
* [Pytorch Official ImageNet Example](https://github.com/pytorch/examples/tree/master/imagenet)
* [Official Repository of " Which Training Methods for GANs do actually Converge?"](https://github.com/LMescheder/GAN_stability)

![structure](./src/structure.png)

## Abstract
We introduce SinGAN, an unconditional generative
model that can be learned from a single natural image.
Our model is trained to capture the internal distribution of
patches within the image, and is then able to generate high
quality, diverse samples that carry the same visual content
as the image. SinGAN contains a pyramid of fully convolu-
tional GANs, each responsible for learning the patch distri-
bution at a different scale of the image. This allows generat-
ing new samples of arbitrary size and aspect ratio, that have
significant variability, yet maintain both the global struc-
ture and the fine textures of the training image. In contrast
to previous single image GAN schemes, our approach is not
limited to texture images, and is not conditional (i.e. it gen-
erates samples from noise). User studies confirm that the
generated samples are commonly confused to be real im-
ages. We illustrate the utility of SinGAN in a wide range of
image manipulation tasks.

## Todo
- [X] Multi-scale GAN with progression
- [X] Initialization via copy
- [X] Scaling noise by the root mean square error between input image and reconstructed one
- [X] Zero padding at the image level (not feature level)
- [X] WGAN-GP loss

### Additional implementation for better quality
- [X] LSGAN loss
- [X] Non-saturating loss with zero-centered gradient penalty

## Notes
  * GAN with Zero-centered GP and larger weight of reconstruction loss exhibits better quality.

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
### Arguments
   * data_dir
    ** dd
   * dataset
   * gantype
   * model_name
   * workers
   * batch_size
   * val_batch
   * img_size_max
   * img_size_min
   * img_to_use
   * load_model
   * validation
   * test
   * world-size
   * rank
   * gpu
   * multiprocessing-distributed
   * port
   
### Train
   * dd
### Test

## Results
   * dd
