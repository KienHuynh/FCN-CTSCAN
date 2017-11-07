# TensorFun

A small TensorFlow project created to test some machine learning problems

Current problem(s) that I'm working on:
* Using fcn to segment CT images

Requirement:

* python 2.7
* tensorflow
* numpy
* matplotlib
* h5py (if you want to use my preprocessed data)


================================================
## Abdominal CT image segmentation using fully convolutional networks
(Original method proposed in [Jonathan Long, 2015](https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf))

![](https://lh3.googleusercontent.com/-ngeNapT4Coo/WPBubqpvy6I/AAAAAAAAN-Q/SsdttF6ZV7YnMHNNmfOLMpQCLW7fh0WHwCJoC/w530-h176/file1.gif)

* Blue: air
* Light blue: bone
* Orange: liver
* Teal/Green (depending on your screen): kidney
* Red: everything else

Dataset: http://www.ircad.fr/research/computer/

Input construction:
* Instead of using a single CT scan image, I combined 3 adjacent images into one stack. The input of the network has the shape of [batch_size, 480, 480, 3].
* Two zero images will be padded before and after the top/bottom CT scan image.

Preprocessing:
* Contrast limited histogram equilization was performed on each CT image (CLAHE)).
* Computing mean and std (shape is [1,1,1,3]) on train data. Subtract each stack of 3 iamge to the mean and divide them with the std.

Network architecture:
* The classification network is very similar to VGG. However, I reduced the number of params and layers since the number of train/test samples and classes are not as large as PASCAL VOC or imagenet.
* The rest of the fcn architecture is generally the same. Bilinear upsampling was used instead of tranposed convolution to avoid overfitting.

Training procedure:
* Before traing the actual fcnn, I had to pretrain the classification network first (as you can see in fcn_pretrain.py). Training samples for this part was generated randomly from the preprocessed data.
* The training method and hyper params can be seen in fcn_ctscan.py.

**Note: my current PC is very limited in computing power, therefore for each patient I only sample half of the images. This resulted in a somewhat watered down precision for each class. I will redo this experiment later on a high-end computer, my expectation is that the overall accuracy will be higher.**

Some preliminary results:
Confusion matrix (except for precision and recall, the unit is pixel):

|| Bone         |Kidney  |Liver  |Other    |Recall   |
|:-------------:|-------:|------:| -------:|--------:|------:|
|**Bone**       |528990  |38     |70       |14171    |0.973  |
|**Kidney**     |133     |74522  |0        |699      |0.989  |
|**Liver**      |675     |10     |1177178  |27398    |0.977  |
|**Other**      |208497  |52340  |181059   |17539260 |0.975  |
|**Precision**  |0.717   |0.587  |0.867    |0.998    |

Other metric:
* Pixel accuracy: 0.976
* Mean accuracy: 0.979
* Mean IU (intersection over union): 0.777
* Frequency weighted IU: 0.957
