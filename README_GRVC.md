# Piloting - segmentation

## Dependencies

 Follow this https://hangzhang.org/PyTorch-Encoding/notes/compile.html , but taken into account that some changes have been made to that repository, so intead ofusing Hang version, use my fork:

### PyTorch

 Install PyTorch 1.4.0 according to: https://pytorch.org/

### PyTorch-Encoding

```bash
git clone https://github.com/JJavierga/PyTorch-Encoding.git && cd PyTorch-Encoding
python setup.py install
```


## Running:

 Take a look at pathtoPyTorch-Encoding/experiments/segmentation/demo_working
 
 It is still to be understood how to free up memory because when you feed many images sequentially it crashes due to lack of available GPU memory.
 

## Training ADE20K-style:

### ADE20K

 It is important to notice that, I don't know why, it does not work properly with class 0. So start at 1. ADE20k style is a grayscale image with a label in each pixel. According to the dataset, labels and classes are: https://github.com/CSAILVision/sceneparsing/blob/master/objectInfo150.csv

 *Attention!* The images have to be in format nameabcd.**jpg** and the labels in format nameabcd.**png** .


### Where and do I move my images?

 Go to homedirectory/.encoding/data/ADEChallengeData2016 and create two folders: annotations and images. Inside of each one, create training and validation directories. Paste in those directories the images you need for training/validation.

 *Attention!* The images have to be in format nameabcd.**jpg** and the labels in format nameabcd.**png** .

 :star: :eye: :eye: :star: Original repository checks that the number of images for training is that of ADE20K dataset, but we don't want that, so I have changed pathtoPyTorch-Encoding/encoding/datasets/ade20k.py so that it only checks that there are images.


### Changing number of classes:

 It looks like it does not work when you use only a couple of the 150 classes. I have manually set the number of classes for deeplab and fcn to be 2 at pathtoPyTorch-Encoding/encoding/models/sseg/fcn (or deeplab). Required change is indicated in the corresponding line.


### Training

 This is the easiest part. Just run the following command with desired flags:

```bash
python3 experiments/segmentation/train_dist.py --model fcn --backbone resnest50 --aux --lr=0.001
```

 :star: :eye: :eye: :star: The value of miou given during training is not trustful,so you will have to check the images by yourself.

 Results obtained are in folder pathtoPyTorch-Encoding/runs/ade20k/model/backbone/default/checkpoint.pth.tar


