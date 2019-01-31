from vae_regressor import  VariantionalAutoencoder
import tensorflow as tf
import numpy as np
from data_handling import BatchMaker

class EPS(object):
    def __init__(self,data, labels,layers,learning_rate = 1e-4, batch_size = 100,VAE_activation=tf.nn.relu):
        self.data = data
        self.labels = labels
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.VAE_activation = VAE_activation
        self.batchMaker = BatchMaker()
        self.batchMaker.load_data(self.data.shape[0])

    
    def create_and_train_vae(self,epochs=50):
    
        tf.reset_default_graph()
        vindim = self.data.shape[1]
        loss_list = []
        model = VariantionalAutoencoder(self.data.shape[1],self.layers,learning_rate=self.learning_rate,
            batch_size=self.batch_size,activation=self.VAE_activation)
        print("TRAINING VARIATIONAL:")
        while(self.batchMaker.batch_number < epochs):
            index_list = self.batchMaker.get_batch(self.batch_size)
            loss = model.run_single_step(self.data[index_list])
            loss_list.append(loss)
        self.VAE_model = model
        
        return model
    def save_model(self,address='./vae_mode.ckpt'):
        if self.VAE_model is None:
            print("There is no model")
            raise Exception("There is no Model")
        self.address = address
        saver = tf.train.Saver()
        saver.save(self.VAE_model.sess,self.address)

        