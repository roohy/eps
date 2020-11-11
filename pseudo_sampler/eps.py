from .vae_regressor import VariantionalAutoencoder,do_regression 
import tensorflow as tf
import numpy as np
from .data_handling import BatchMaker,normalizer
import gc
VAE_STATE_EMPTY = 0
VAE_STATE_EMPTY_MODEL = 3
VAE_STATE_FITTED = 1
VAE_STATE_SAVED = 2
REGRESSOR_STATE_EMPTY = 0
REGRESSOR_STATE_TRAINED = 1
MODEL_LOADED  = True
MODEL_NOT_LOADED = False


class EPS(object):
    def __init__(self):
        self.layers = None
        self.vae_state = VAE_STATE_EMPTY 
        self.regressor_state = REGRESSOR_STATE_EMPTY
        self.ex_data = None
        self.fullbool = None
        self.model_state = MODEL_NOT_LOADED
    
    def set_layers(self,layers,activation_function=tf.nn.relu):
        self.layers = layers
        self.VAE_activation = activation_function

    def create_VAE(self,layers,activation_func):
        self.set_layers(layers)
        self.VAE_activation = activation_func
        tf.reset_default_graph()
        self.VAE_model = VariantionalAutoencoder(self.layers[0],self.layers,learning_rate=self.learning_rate,
            batch_size=self.batch_size,activation=self.VAE_activation)
        self.model_state = MODEL_LOADED
        self.vae_state = VAE_STATE_EMPTY_MODEL
        return self.VAE_model
    
    def save_VAE(self,address):
        if self.VAE_model is None:
            print("There is no model to save!")
            raise Exception("No VAE Model")
        saver = tf.train.Saver()
        saver.save(self.VAE_model.sess,address)

    def load_VAE(self,address):
        if self.model_state and self.vae_state != VAE_STATE_EMPTY_MODEL:
            tf.reset_default_graph()
        if self.layers is None:
            raise Exception('Please set a layer architecture and activation function first')
        if self.vae_state != VAE_STATE_EMPTY_MODEL:
            self.VAE_model = VariantionalAutoencoder(self.layers[0],self.layers,learning_rate=self.learning_rate,
                batch_size=self.batch_size,activation=self.VAE_activation)
        saver = tf.train.Saver()
        saver.restore(self.VAE_model.sess,address)
        self.vae_state = VAE_STATE_FITTED
        self.model_state = MODEL_LOADED
    
    def train_VAE(self,epochs,data):
        if self.VAE_model is None:
            raise Exception("No VAE Model!")
        loss_list = []
        batchMaker = BatchMaker()
        batchMaker.load_data(data)
        print("TRAINING VARIATIONAL:")
        while(batchMaker.batch_number < epochs):
            data_batch = batchMaker.get_batch(self.batch_size)
            
            loss = self.VAE_model.run_single_step(data_batch)
            loss_list.append(loss)
        self.vae_state = VAE_STATE_FITTED
    def transform_data(self,data):
        return self.VAE_model.transformer(data)

    '''def __init__(self,data, labels,layers,learning_rate = 1e-4, batch_size = 100,VAE_activation=tf.nn.relu,normalize=True):
        if normalize:
            self.data = normalizer(data)
        else:
            self.data = data
        self.labels = labels
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape(self.labels.shape[0],1)
        self.layers = layers
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.VAE_activation = VAE_activation
        self.batchMaker = BatchMaker()
        self.batchMaker.load_data(self.data.shape[0])'''

    def create_and_train_vae(self,data,activation_func,epochs=50):
        self.create_VAE(self.layers,activation_func)
        self.train_VAE(epochs,data)
        return self.VAE_model


    
    def train(self,data,labels,vae_epochs=50,
        learning_rate=1e-4, batch_size = 100,VAE_activation=tf.nn.relu,
        normalize=True,vae_address='./vae_mode.ckpt',layers = None):
        if normalize:
            self.data = normalizer(data)
        else:
            self.data = data
        
        self.labels = labels
        if len(self.labels.shape) == 1:
            self.labels = self.labels.reshape(self.labels.shape[0],1)
        if self.layers is None:
            if layers is None:
                raise Exception('No layers were provided anywere.')
            else:
                self.set_layers([data.shape[1]]+layers)
        
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.vae_address = vae_address
        
        self.create_and_train_vae(data,VAE_activation,vae_epochs)
        self.save_VAE(vae_address)
        self.vae_state = VAE_STATE_SAVED
        self.transformed_data = self.transform_data(self.data)
        self.VAE_model.sess.close()
        gc.collect()
        
        return self
    
    '''def train_classifier(self,regression_epochs=200,regression_index=None):
        if not self.vae_state == VAE_STATE_SAVED:
            raise Exception('No VAE trained. Call the "train" function first.')
        if regression_index is not None:
            self.transformed_data = self.transform_data(self.data[regression_index])
        else:
            self.transformed_data = self.transform_data(self.data)
        
        self.temp_labels = self.labels if regression_index is None else self.labels[regression_index]
        model = do_regression(self.transformed_data,self.temp_labels,regression_epochs)
        self.regressor_state = REGRESSOR_STATE_TRAINED'''
    def generate(self,count=200,regression_epochs=500,learning_rate=1e-4,regression_index=None,variance=0.2,seed_count=1):
        if not self.vae_state == VAE_STATE_SAVED:
            raise Exception('No VAE trained. Call the "train" function first.')
        self.regression_epochs = regression_epochs
        self.regressor_learning_rate = learning_rate
        # if not self.regressor_state == REGRESSOR_STATE_TRAINED:
        #     raise Exception('No Regressors available. Call the "train" function first.')
        if regression_index is not None:
            self.transformed_data = self.transform_data(self.data[regression_index])
        else:
            self.transformed_data = self.transform_data(self.data)
        
        self.temp_labels = self.labels if regression_index is None else self.labels[regression_index]
        model = do_regression(self.transformed_data,self.temp_labels,regression_epochs,learning_rate=self.regressor_learning_rate)
        self.regressor_state = REGRESSOR_STATE_TRAINED

        # print('SAVING REGRESSOR')
        # saver = tf.train.Saver()
        # saver.save(model.sess,model_address+'latent_reg.ckpt')
        self.w = model.sess.run(model.W)
        self.b = model.sess.run(model.b)
        dists = self.transformed_data.dot(self.w) + self.b

        # max_point = self.transformed_data[np.argmax(dists),:]
        # min_point = self.transformed_data[np.argmin(dists),:]
        max_points = self.transformed_data[np.argsort(dists)[-seed_count:],:]
        min_points = self.transformed_data[np.argsort(dists)[:seed_count],:]

        
        cov = np.eye(self.transformed_data.shape[1])
        cov = cov*variance
        max_rand = []
        min_rand = []
        for i in range(seed_count):
            max_rand.append(np.random.multivariate_normal(max_points[i],cov,count))
            min_rand.append(np.random.multivariate_normal(min_points[i],cov,count))
        # max_rand = np.random.multivariate_normal(max_point,cov,count)

        # min_rand = np.random.multivariate_normal(min_point,cov,count)
        max_rand = np.concatenate(max_rand,axis=0)
        min_rand = np.concatenate(min_rand,axis=0)
        tf.reset_default_graph()
        model = VariantionalAutoencoder(self.data.shape[1],self.layers,learning_rate=self.learning_rate,
            batch_size=self.batch_size,activation=self.VAE_activation)
        saver = tf.train.Saver()
        saver.restore(model.sess,self.vae_address)  
        max_generated = model.generator(max_rand)
        min_generated = model.generator(min_rand)
        model.sess.close()
        gc.collect()
        ex_data = np.concatenate((min_generated,max_generated),axis=0)
        fullbool = np.zeros((count*2,1))
        fullbool[count:count*2,0]+=1
        
        tf.reset_default_graph()
        self.ex_data = ex_data
        self.fullbool = fullbool
        return ex_data,fullbool
    
    def rank(self):
        if self.ex_data is None:
            raise Exception('No generated data available. Try training a model and generating data first.')
        #print("INITIATING EXAGGERATED REGRESSOR...")
        model = do_regression(self.ex_data,self.fullbool,self.regression_epochs,learning_rate=self.regressor_learning_rate)
        
        
        orig_w = model.sess.run(model.W)
    
        sortedargs = np.argsort(-np.fabs(orig_w[:,0]))
        
        return sortedargs
        
            