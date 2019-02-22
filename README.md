# Extreme Pseudo-Sampler
EPS is a feature selection and feature ranking method. 
We have described the technique and used it to extract gene rankings in 12 case-control RNA-Seq data sets ranging from 323 to 1,210 samples in a [paper published in Frontiers in Genetics](https://www.frontiersin.org/articles/10.3389/fgene.2018.00297/full). 

## How it works
This library uses [TensorFlow](https://www.tensorflow.org/) in 4 steps: 
1. It first creates a Variational Auto-Encoder (VAE) to map each point from feature space to a distribution in the latent space. You can read more about VAEs [here](https://arxiv.org/abs/1312.6114).
2. It then uses a regression model to classify samples in the latent space with good accuracy using a simple line. It then finds the furthest points to that line on both sides. We call these *extreme samples*. 
3. It then randomly generates new samples around the extreme samples using a normal distribution, called Extreme Pseudo-Samples. Using the same trained VAE, these newly generated samples are mapped back into the feature space. 
4. A new regression model is trained to classify the generated Pseudo-Samples. We then use the regression line to rank the most important features in these Extreme Pseudo-Samples. 

## Installation
Using setup_tools, install the package using the following command:

`python3 -m pip install pseudo_sampler`

Or you can download the code and import it to your project manually. Or use virtual environments.

## Usage
Import the main class called EPS from the package:

`from pseudo_sampler.eps import EPS`

You can then create an EPS instant using the following snippet:

`eps = EPS(data, labels, layers, learning_rate = 1e-4, batch_size = 100, VAE_activation=tf.nn.relu, normalize=True) `

- *data* should be a float numpy array with N\*D dimensions, where N is the total number of samples and D is the number of features in the feature space. 
- *labels* should be a numpy array with length N containing 1s and 0s for cases and controls. 
- *layers* is an integer python list containing the number of perceptrons in every layer of your Deep Variational Auto-Encoder; only in the encoder side (the decoder side will be mirrored).

   For example if you want to have a Deep Network with the following structure:
`Input -> 250 -> 120 -> 60 -> Latent Space with 30 dimensions -> 60 -> 120 -> 250 -> Output`, you can represent it by passing the following as your layers argument: `layers = [250,120,60,30]`

- *learning_rate*, *batch_size* and *VAE_activation* (the activation function) are used to create the VAE. 

The input data by default will be normalized to be between 0 and 1. In case the input data is already in that interval, the normalization flag (*normalize*) can be set to off. 

After creating an EPS instance, you can run your EPS experiment with the following command:

`feature_ranks = eps.run(vae_epochs=50,regression_epochs=500,vae_address=‘./vae_mode.ckpt’)`

You can set the number of epochs for VAE and Linear Regression models separately. 

Trained VAE models are saved mid-operation as back-up. 
You can set the backup address using the *vae_address* argument.
The *EPS.run* function returns a list of indices of features sorted based on their importance in classification of EPS.

## Future Work
In the near future, We plan to add the ability of using VAE classifiers for classification purposes and not just feature ranking. we also plan to add more customization options for models, such as the number of distributions available for generating EPS. 

## Citation

**Wenric S and Shemirani R (2018) Using Supervised Learning Methods for Gene Selection in RNA-Seq Case-Control Studies.** *Front. Genet. 9:297.* doi: [10.3389/fgene.2018.00297](https://doi.org/10.3389/fgene.2018.00297)