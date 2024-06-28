# Recurrent neural networks for the dynamics of the membrane's displacement in the micromachined fixed-fixed beam 

This repository is the official implementation of "Efficient machine learning methods to predict the stationary
electron density profile in a n+/n/n+ silicon diode"

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

>📋  The available code was developed using Python 3.12.1.
>📋  To run the code the libraries available in the "requirements.txt" file were used, with the referred version.
>📋  It is recommended to install them with the given command and with the same version of each library, although other versions might still work.

## Training

To train the model(s) in the paper, run this command:

```train
python Diode.py -epochs 2 -interpolations 2 -reservoir_size 3500 -sparsity 0.5 -savepath <path where to save the results to/> -plots True/False
```
>📋 The input parameters for the training script are:
    * -epochs: The number of epochs to train the model
    * -interpolations: The number of points to interpolate between consecutive points
    * -reservoir_size: Size of the reservoir
    * -sparsity: Proportion of the reservoir matrix weights forced to be zero.
    * -savepath: The path where to save the weights, model and training history
    * -plots: Whether to plot the inputs and outputs from the dataset or not


