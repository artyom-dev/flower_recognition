# Image classification using deep learning 


[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

In the framework of this project where trained two famous NN's - Inceptionv3 and EfficientNetb1. Project include following sections

- Explanatory Data Analasys
- Training pipeline

## Data

- Flowers Recognition Dataset with 5 classes of flowers
- Total images: 4317 images
- Daisy: 764 images
- Dandelion: 1052 images
- Rose: 784 images
- Sunflower: 733 images 
- Tulip: 984 images 


## Result
- EfficientNet show pretty good results with:
- Train Accuracy: 94%
- Validation Accuracy: 82%
- Precision/recall: 91%, 86%
- Matthews Corr. coef.: 0.86
- Cohen Kappa: 0.86
- 



## Used libraries and frameworks

During project where used following libraries:

- [PyTorch] - Deep Learning Framework
- [Pandas] - awesome data processing library
- [torchvision] - Tools for working with PyTorch CV related problems.
- [Wandb] - Experiment Tracker
- [torchmetrics] - PyTorch metrics
- [sklearn] - Machine Learning library

## Installation

Project requires [Pytorch](https://pytorch.org/)  1.11.0.

Install the dependencies via pip or conda install managers.

```sh
conda install pytorch torchvision torchaudio cpuonly -c pytorch
conda install -c pytorch torchvision
conda install -c conda-forge torchmetrics
conda install -c conda-forge wandb
```



[//]: # (These are reference links used in the body of this note and get stripped out when the markdown processor does its job. There is no need to format nicely because it shouldn't be seen. Thanks SO - http://stackoverflow.com/questions/4823468/store-comments-in-markdown-syntax)

   [Pytorch]: <https://pytorch.org/>
   [Pandas]: <https://pandas.pydata.org/>
   [torchvision]: <https://pytorch.org/vision/stable/index.html>
   [Wandb]: <https://wandb.ai/home>
   [torchmetrics]: <https://torchmetrics.readthedocs.io/en/stable/>
   [sklearn]: <https://scikit-learn.org/stable/>


   [PlDb]: <https://github.com/joemccann/dillinger/tree/master/plugins/dropbox/README.md>
   [PlGh]: <https://github.com/joemccann/dillinger/tree/master/plugins/github/README.md>
   [PlGd]: <https://github.com/joemccann/dillinger/tree/master/plugins/googledrive/README.md>
   [PlOd]: <https://github.com/joemccann/dillinger/tree/master/plugins/onedrive/README.md>
   [PlMe]: <https://github.com/joemccann/dillinger/tree/master/plugins/medium/README.md>
   [PlGa]: <https://github.com/RahulHP/dillinger/blob/master/plugins/googleanalytics/README.md>
