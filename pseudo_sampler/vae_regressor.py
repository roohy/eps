import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.contrib.slim import fully_connected as fc
from .regressor import LogisticRegressor
from .data_handling import BatchMaker



vindim = None
class VariantionalAutoencoder(object):

    def __init__(self,vindim,layers, learning_rate=1e-3, batch_size=100,activation=tf.nn.relu):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vindim = vindim
        self.layers = layers
        self.n_z = layers[-1]
        self.activation_func = activation
        self.build()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        

    # Build the netowrk and the loss functions
    def build(self):
        self.tf_layers = []
        self.x = tf.placeholder(name='x', dtype=tf.float32, shape=[None, self.vindim])
        current_layer = self.x
        for layer in self.layers[:-1]:
            current_layer = fc(current_layer,layer,scope='enc_'+str(layer),activation_fn=self.activation_func)
            self.tf_layers.append(current_layer)
        self.z_mu = fc(current_layer,self.n_z,scope='enc_fc_mu',activation_fn=None)
        self.z_log_sigma_sq = fc(current_layer,self.n_z,scope='enc_fc_sigma',activation_fn=None)
        eps = tf.random_normal(shape=tf.shape(self.z_log_sigma_sq),mean=0,stddev=1,dtype=tf.float32)
        self.z = self.z_mu + tf.sqrt(tf.exp(self.z_log_sigma_sq))*eps
        current_layer = self.z
        for layer in self.layers[:-1][::-1]:
            current_layer = fc(current_layer,layer,scope='dec_'+str(layer),activation_fn=self.activation_func)
            self.tf_layers.append(current_layer)
        self.x_hat = fc(current_layer,self.vindim,scope="dec_x_hat",activation_fn=tf.sigmoid)
        # Loss
        # Reconstruction loss
        # Minimize the cross-entropy loss
        # H(x, x_hat) = -\Sigma x*log(x_hat) + (1-x)*log(1-x_hat)
        epsilon = 1e-9
        recon_loss = -tf.reduce_sum(
            self.x * tf.log(epsilon+self.x_hat) + (1-self.x) * tf.log(epsilon+ (1-self.x_hat)), 
            axis=1
        )
        #recon_loss = tf.reduce_sum((self.x_hat-self.x)**2,axis=1)

        #recon_loss = tf.nn.l2_loss(self.x_hat-self.x)
        self.recon_loss = tf.reduce_mean(recon_loss)

        # Latent loss
        # Kullback Leibler divergence: measure the difference between two distributions
        # Here we measure the divergence between the latent distribution and N(0, 1)
        latent_loss = -0.5 * tf.reduce_sum(
            1 + self.z_log_sigma_sq - tf.square(self.z_mu) - tf.exp(self.z_log_sigma_sq), axis=1)
        self.latent_loss = tf.reduce_mean(latent_loss)

        self.total_loss = tf.reduce_mean(latent_loss + recon_loss)
        self.train_op = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.total_loss)
        #self.train_op = tf.train.AdamOptimizer(
        #    learning_rate=self.learning_rate).minimize(self.total_loss)
        return

    # Execute the forward and the backward pass
    def run_single_step(self, x):
        _, loss, recon_loss, latent_loss = self.sess.run(
            [self.train_op, self.total_loss, self.recon_loss, self.latent_loss],
            feed_dict={self.x: x}
        )

        return loss, recon_loss, latent_loss

    # x -> x_hat
    def reconstructor(self, x):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.x: x})
        return x_hat

    # z -> x
    def generator(self, z):
        x_hat = self.sess.run(self.x_hat, feed_dict={self.z: z})
        return x_hat
    
    # x -> z
    def transformer(self, x):
        z = self.sess.run(self.z, feed_dict={self.x: x})
        return z



def do_regression(data,labels,epochs,batch_size=300,learning_rate=1e-4):
    tf.reset_default_graph()
    model = LogisticRegressor(learning_rate=learning_rate,input_dim=data.shape[1])
    print('TRAINIG LATENT SPACE REGRESSOR:')
    data_handler = BatchMaker()
    data_handler.load_data(data.shape[0])
    loss_list = []
    while data_handler.batch_number < epochs:
        index_list = data_handler.get_batch(batch_size)
        loss = model.run_single_step(data[index_list,:],labels[index_list])
        loss_list.append(loss) 
        if data_handler.batch_number % 5 == 0:
            print('[Epoch {}] Loss: {}'.format(data_handler.batch_number, loss))
    predictions = model.classifier(data)
    latent_reg_acc = np.sum(predictions[:] == labels[:,0])/labels.shape[0]
    print('Latent Regressor Accuracy is :',latent_reg_acc)
    return model
