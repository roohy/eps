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

Create an EPS instance using the following snippet:

`eps = EPS()`

Train the Variational AutoEncoder (VAE):

`train(data,labels,vae_epochs, learning_rate, batch_size,VAE_activation, normalize,vae_address,layers)`

- *data* should be a float numpy array with N\*D dimensions, where N is the total number of samples and D is the number of features in the feature space. Data from multiple case/control datasets can be merged to enhance the training process and prevent NaN errors.
- *labels* should be a numpy array with length N containing 1s and 0s for cases and controls. When combining multiple datasets, there is no need to have different label numbers for various datasets cases.
- *vae_epochs* is the number of traning epochs for the VAE traning (*default=50*).
- *learning_rate* is the learning rate for the RMSProp optimizer that fits the VAE model (*default=1e-4*).
- *batch_size* Number of samples in each batch (*default=100*).
- *VAE_activation* sets the activation functions of the nodes in the VAE neural network (*default=tf.nn.relu*).
- If the *normalize* is set to true, EPS will normalize the data to be between 0 and 1. They should be normalized, thus the default value for this option is set to true. If the data is already normalized, a false flag can be passed.
- *vae_address* sets to address that EPS uses to save the VAE model. EPS has to retrieve the model later to use the decoder, or to train regressors for other case/control groups (*default=./vae_mode.ckpt*).
- *layers* is an integer python list containing the number of perceptrons in every layer of your Deep Variational Auto-Encoder; only in the encoder side; the decoder side will be mirrored (*default=None*; The default value should only be used if the the layers have been setup using `set_layers` prior to training).

   For example if you want to have a Deep Network with the following structure:
`Input -> 250 -> 120 -> 60 -> Latent Space with 30 dimensions -> 60 -> 120 -> 250 -> Output`, you can represent it by passing the following as your layers argument: `layers = [250,120,60,30]`

The `train` function returns the EPS instance object.

After training the VAE, you can generate extreme psuedo-samples by calling the `generate` function:

`eps.generate(count,regression_epochs,learning_rate,regression_index,variance)`

- *count* parameter sets the number of extreme pseudo-samples generated (*default=200*).
- *regression_epochs* sets the number of epochs for the logistic regression training (*default=500*).
- *learning_rate* sets the learning rate parameter for the Adam optimizer that fits the logitstic regression model (*default=1e-4*).
- If you only want to use a subset of the data to train the regressor model, you pass the list of indices as a numpy array  in *regression_index* parameter. This is useful for adopting multiple case/control study data to train a VAE and performing the rest of the feature selection steps separately for each dataset.
- The *variance* parameter is used in the process of generating new extreme pseudo-samples around the real extreme samples (*default=0.2*).

The `generate` function returns the extreme pseudo-samples and their labels.

After calling the generate function, the EPS object would also have the feature rankings. The `rank` function returns these rankings as a list of sorted indices based on the original order of features.

`eps.rank()`

## Future Work
In the near future, we plan to add more customization options for models, such as the number of distributions available for generating EPS, separate activation function options for each layer, and a variety of optimizers for each model. 

## Citation

**Wenric S and Shemirani R (2018) Using Supervised Learning Methods for Gene Selection in RNA-Seq Case-Control Studies.** *Front. Genet. 9:297.* doi: [10.3389/fgene.2018.00297](https://doi.org/10.3389/fgene.2018.00297)

**Shemirani R, Wenric S, Kenny E, and Ambite JL (2021) EPS: Automated Feature Selection in Case-Control Studies using Extreme Pseudo-Sampling.** *Bioinformatics (Oxford, England), btab214.* doi: [10.1093/bioinformatics/btab214](https://doi.org/10.1093/bioinformatics/btab214)