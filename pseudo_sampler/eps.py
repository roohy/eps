from vae_regressor import  VariantionalAutoencoder,do_regression
import tensorflow as tf
import numpy as np
from data_handling import BatchMaker,normalizer
import gc


class EPS(object):
    def __init__(self,data, labels,layers,learning_rate = 1e-4, batch_size = 100,VAE_activation=tf.nn.relu,normalize=True):
        if normalize:
            self.data = normalizer(data)
        else:
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
            loss = model.run_single_step(self.data[index_list,:])
            loss_list.append(loss)
        self.VAE_model = model
        
        return model

    def save_model(self,address):
        if self.VAE_model is None:
            print("There is no model")
            raise Exception("There is no Model")
        self.address = address
        saver = tf.train.Saver()
        saver.save(self.VAE_model.sess,self.address)

    
    def run(self,vae_epochs=50,regression_epochs=500,vae_address='./vae_mode.ckpt'):
        self.create_and_train_vae(vae_epochs)
        self.save_model(vae_address)
        transformed_data = self.VAE_model.transformer(self.data)
        self.VAE_model.sess.close()
        gc.collect()

        model = do_regression(transformed_data,self.labels,regression_epochs)
        # print('SAVING REGRESSOR')
        # saver = tf.train.Saver()
        # saver.save(model.sess,model_address+'latent_reg.ckpt')
        w = model.sess.run(model.W)
        b = model.sess.run(model.b)
        
        dists = transformed_data.dot(w) + b

        max_point = transformed_data[np.argmax(dists),:]
        min_point = transformed_data[np.argmin(dists),:]

        cov = np.eye(500)
        cov = cov*0.2

        max_rand = np.random.multivariate_normal(max_point,cov,200)

        min_rand = np.random.multivariate_normal(min_point,cov,200)
            
        tf.reset_default_graph()
        model = VariantionalAutoencoder(self.data.shape[1],self.layers,learning_rate=self.learning_rate,
            batch_size=self.batch_size,activation=self.VAE_activation)
        saver = tf.train.Saver()
        saver.restore(model.sess,vae_address)
        max_generated = model.generator(max_rand)
        min_generated = model.generator(min_rand)
        model.sess.close()
        gc.collect()
        ex_data = np.concatenate((min_generated,max_generated),axis=0)
        fullbool = np.zeros((400,1))
        fullbool[200:400,0]+=1
        
        tf.reset_default_graph()
        
        print("INITIATING EXAGGERATED REGRESSOR...")
        model = do_regression(ex_data,fullbool,regression_epochs)
        
        
        orig_w = model.sess.run(model.W)
    
        sortedargs = np.argsort(-np.fabs(orig_w[:,0]))
        
        return sortedargs
        

            