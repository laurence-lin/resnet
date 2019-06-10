import resnet as resnet
import tensorflow as tf
import cifar_10 as cifar
import numpy as np
import os
import pandas as pd
from datetime import datetime  

'''Hyperparamaters at the beginning'''
'''About data'''
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNEL = 3
num_class = 10
'''About training''' 
total_data = 50000 
test_size = 10000
train_epoch = 50
train_batch_size = 125
num_of_train_batch = int(total_data/train_batch_size)
validate_batch_size = 250
test_batch_size = 250
num_of_val_batch = int(test_size/test_batch_size)
report_freq = 1
'''About network configuration'''
n_residual_block = 5
padding_size = 2
'''About model saving and tensorboard showing result'''
model_save_path = './checkpoint/model'
model_file_path = './checkpoint/'
tensorboard_save_path = './tensorboard/'
# Not sure if needed
version = 'test110'
load_ckpt = False 

class Train(object):
    def __init__(self):
        self.learn_rate = 0.001
        self.placeholder()
    
    def placeholder(self):
        self.x_data = tf.placeholder(tf.float32, [train_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        self.y_data = tf.placeholder(tf.int32, [train_batch_size])
        self.x_valid = tf.placeholder(tf.float32, [test_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        self.y_valid = tf.placeholder(tf.int32, [test_batch_size])
        
    def train_and_valid_graph(self):
        '''
        Built train & valid graph
        '''
        # We compute train data and validate data on same session graph. By setting
        # variable_scope reuse = True, we could share the weight variables for the 
        # training and validating
        global_step = tf.Variable(0, trainable = False) # this is for compute decay of moving average
        logits = resnet.inference(self.x_data, n_residual_block, False)
        valid_logits = resnet.inference(self.x_valid, n_residual_block, True)  # while reuse = True, share same variables with training model        
        
        train_loss = self.loss(logits, self.y_data)
        #regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        '''weight = tf.trainable_variables()
        l2_regularizer = tf.contrib.layers.l2_regularizer(0.002)
        penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weight)'''
        self.total_loss = train_loss 
        prediction = tf.nn.softmax(logits)
        self.train_top_k_err = self.top_k_error(prediction, self.y_data, 1) # tensor type error
        self.train_op, self.train_ema_op = self.train_operation(self.total_loss, self.train_top_k_err, global_step)

        self.valid_loss = self.loss(valid_logits, self.y_valid)
        valid_prediction = tf.nn.softmax(valid_logits)
        self.valid_top_k_err = self.top_k_error(valid_prediction, self.y_valid, 1)
        
        # Add train loss & error summary to tensorboard
        tf.summary.scalar('Training loss', self.total_loss)
        tf.summary.scalar('Top-1 error', self.train_top_k_err)
        tf.summary.scalar('Validate loss', self.valid_loss)
        tf.summary.scalar('Validate top-1 error', self.valid_top_k_err)
        
    def train_operation(self, total_loss, top_1_err, global_step):
        '''
        Define optimizer
        '''
        ema_decay = 0.95
        ema = tf.train.ExponentialMovingAverage(ema_decay, global_step) # global step is to make actual decay increase to its limit
        train_ema_op = ema.apply([total_loss, top_1_err])
        train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(total_loss, global_step = global_step) # global step to add 1 after loss is updated
   
        return train_op, train_ema_op     
    
    def top_k_error(self, prediction, label, k):
        '''
        Generate top-k error rate
        predictions: 2D tensor with shape[batch, num_labels]
        label: 1D tensor with shape [ num_labels, 1]
        k: int
        return: tensor with shape [1]
        '''
        batch_size = prediction.get_shape().as_list()[0]
        in_top1 = tf.cast(tf.nn.in_top_k(prediction, label, k=1), tf.float32) # use top-1 error, return 1D boolean classify result, turn to float
        num_correct = tf.reduce_sum(in_top1)
        # here use subtraction for tensor, so return error rate is tensor
        return (batch_size - num_correct)/float(batch_size)
        
    def loss(self, logits, labels):
        '''
        Calculate cross entropy loss btw logits and labels
        logits: 2D output array [batch_size, num_of_class]
        labels: 1D array [batch_size]
        return: loss tensor with shape[1]
        '''
        labels = tf.cast(labels, tf.int64) # convert to tensor type int64
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = logits, labels = labels) # sparse softmax loss: input labels have shape [batch_size]
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        return cross_entropy_mean
    
    def generate_augment_train_batch(self, train_data, train_labels, train_batch_size):
        '''
        Help generate augmented train data by random crop, flips, whitening on whole dataset
        train_data: 4D numpy array
        train_labels: 1D array
        return: 4D augmented array, 1D augmented array
        '''
        # Randomly do crop and flipping for whole training set 
        batch_data = cifar.random_crop_flip(train_data, padding_size) 
        whitened_data = cifar.whiten_img(batch_data) # augmented data batch
        
        return whitened_data, train_labels
    
    def generate_vali_batch(self, val_data, val_label, val_batch_size):
        '''
        Generate batch of validate data if we don't want to input whole validate data
        '''
        offset = np.random.choice(test_size - test_batch_size)
        batch_data = val_data[offset:offset + test_batch_size, :]
        batch_label = val_label[offset:offset + test_batch_size]
        
        return batch_data, batch_label
    
    def train(self):
        '''
        main function of training
        '''
        all_data, all_labels = cifar.train(pad_size = padding_size)
        valid_data, valid_labels = cifar.test()
        
        self.train_and_valid_graph()  # define training loss, optimizer, error rate
        
        # Initialize saver to save checkpoints, merge all summary, set up a session graph
        saver = tf.train.Saver(tf.global_variables()) # specify all global variables to be save or restore
        summary_op = tf.summary.merge_all()
        init = tf.global_variables_initializer()
        sess = tf.Session() # session graph to run later
        
        # Determine whether load past checkpoint or not
        if load_ckpt is True:
            files = os.listdir(model_file_path)
            meta_graph = [f for f in files if 'meta' in f]
            meta_graph = os.path.join(model_file_path, meta_graph[0])
            saver = tf.train.import_meta_graph(meta_graph) # if load past model, don't need initialize all global variables
            saver.restore(sess, tf.train.latest_checkpoint(model_file_path))
        else:
            sess.run(init)
            
        # Writer to write on tensorboard
        summary_writer = tf.summary.FileWriter(tensorboard_save_path, sess.graph)
        
        # save error as csv file list
        train_err = []
        step_list = []
        top_1_list = []
        val_err_list = []
        
        print('Start training...')
        print('------------------------------')
        
        for iterate in range(train_epoch):
            
            # data augmentation on all data randomly
            train_data, train_labels = self.generate_augment_train_batch(
                        all_data, all_labels, train_batch_size)
            #val_data, val_labels = self.generate_vali_batch(valid_data, valid_labels, test_batch_size)
            
            # Do validation before training, use test data here. This validate result could perform as test performance
            if iterate % report_freq == 0:
                err = 0
                val_loss = 0
                for batch in range(num_of_val_batch):
                    offset = batch * test_batch_size
                    valid_error, valid_loss = sess.run([self.valid_top_k_err, self.valid_loss],
                                                   {self.x_valid: valid_data[offset:offset + test_batch_size, :],
                                                    self.y_valid:valid_labels[offset:offset + test_batch_size]})
                    err += valid_error
                    val_loss += valid_loss
                
                valid_error  = err/num_of_val_batch
                valid_loss = val_loss/num_of_val_batch
                val_err_list.append(valid_error)
                
            # train several batches
            for batch in range(num_of_train_batch):
                offset = batch * train_batch_size
                x_train = train_data[offset:offset + train_batch_size, :]
                y_train = train_labels[offset:offset + train_batch_size]
                _, training_loss, top_1_err_rate = sess.run([self.train_op, self.total_loss, self.train_top_k_err], 
                                                        {self.x_data: x_train,
                                                         self.y_data: y_train})
                
                if batch % 100 == 0: # show training performance for each 100 batches
                    print('Batch:', batch)
                    print('Training loss: ', training_loss, 'Train top-1 err: ', top_1_err_rate)
    
            if iterate % report_freq == 0:
            # Summary for tensorboard result
                summary_str = sess.run(summary_op, {self.x_data: x_train,
                                                self.y_data: y_train,
                                                self.x_valid: valid_data[0:test_batch_size, :],
                                                self.y_valid: valid_labels[0:test_batch_size]})
                summary_writer.add_summary(summary_str, iterate)
            
                
                print('%s: Step: %d,  loss: %.4f'%(datetime.now(), iterate, training_loss))
                print('Train top-1 error rate: ', top_1_err_rate)
                print('Validation loss: ', valid_loss)
                print('Validation top-1 error rate:', valid_error)
                one_batch_val_err = sess.run(self.valid_top_k_err, {self.x_valid: valid_data[0:test_batch_size, :],
                                                    self.y_valid:valid_labels[0:test_batch_size]})
                print('Validate top-1 error one batch: ', one_batch_val_err)
                
            
                step_list.append(iterate)
                train_err.append(training_loss)
                top_1_list.append(top_1_err_rate)
            
            # Save model checkpoints every 100 steps
            if iterate % 5 == 0 or (iterate + 1) == train_epoch:
                saver.save(sess, model_save_path, global_step = iterate) # checkpoint name for every (iterate) steps
                
                # save train error dataframe
                df = pd.DataFrame(data = {'steps: ':step_list, 'training loss:': 
                                          train_err, 'top_1_err:':top_1_list, 'valid error: ':valid_error})
    
                df.to_csv(model_save_path + '_error_list.csv')
        
train = Train()
train.train()

            
        
        
        
        
        
    
    
    
    
    
    