# Extreme Pseudo-Sampler
EPS is a feature selection and feature ranking method. We have described the technique and used to extract a gene selection for and RNA-Seq Case-Control study in a [paper](https://www.frontiersin.org/articles/10.3389/fgene.2018.00297/full). 
## How it works
This library uses TensorFlow in 4 steps. 
1. It first creates a Variational Auto-Encoder (VAE) to map each point from feature space to a distribution in the latent space. You can read more about VAEs [here](https://arxiv.org/abs/1312.6114).
2. It then uses a regression model to classify samples in the latent space with good accuracy using a simple line. It then finds the furthest points to that line on both sides of it. We call these extreme samples. 
3. It then randomly generates new samples around our extreme samples using a normal distribution, called Extreme Pseudo-Samples. Using the same trained VAE, these newly generated samples are mapped back into the feature space. 
4. A new regression model is trained to classify our generated Pseudo-Samples. We then use the regression line to rank the most important features in these Extreme Pseudo-Samples. 

## Installation
Using setup_tools, install the package using the following command:
`python3 -m pip install pseudo-sampler`
Or you can download the code and import it to your project manually. Or use virtual environments.
## Usage
Import the main class called EPS from the package:
`from pseudo-sampler.eps import EPS`
You can then create an EPS instant using the following snippet:
`eps = EPS(data, labels, layers, learning_rate = 1e-4, batch_size = 100, VAE_activation=tf.nn.relu, normalize=True) `
Data should be a float numpy array with N*D dimensions. Where N is the total number of samples and D is the number of features in the feature space. 
Labels should be a numpy array with length N containing 1s and 0s for cases and controls. 
Layers is an integer python list containing the number of perceptrons in every layer of your Deep Variational Auto-Encoder; only in the encoder side (decoder side will be mirrored) . For example if you want to have a Deep Network with the following structure:
Input -> 250 -> 120 -> 60 -> Latent Space with 30 dimensions -> 60 -> 120 -> 250 -> Output 
You can represent it by passing the following as your layers argument. 
`layers = [250,120,60,30]`
Learning rate, batch size and activation function are used to create the VAE. 
The input data by default will be normalized to be between 0 and 1. In case they already are in that interval, you can set the normalization flag off. 
After creating an EPS instance, you can run your EPS experiment with the following command:
`feature_ranks = eps.run(vae_epochs=50,regression_epochs=500,vae_address=‘./vae_mode.ckpt’)`
You can set the number of epochs for VAE and Linear Regression models separately. Trained VAE models are saved mid-operation as back-up. You can set the backup address using vae_address argument.
EPS.run function returns a list of indices of features sorted based on their importance in classification of EPS.

## Future Work
In the near future, I plan to add the ability of using VAE classifier for classification purposes and not just feature ranking. I also plan to add more customization options for models. Such as the number of distributions available for generating EPS. 





 [Github-flavored Markdown](https://guides.github.com/features/mastering-markdown/)