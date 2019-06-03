import resnet as resnet
import tensorflow as tf
import cifar_10 as cifar
import numpy as np
import time
import os
import pandas as pd
import csv    
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
train_epoch = 3000
train_batch_size = 128
validate_batch_size = 250
test_batch_size = 250
report_freq = 5
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
        self.learn_rate = 0.1
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
        global_step = tf.Variable(0, trainable = False)
        validate_step = tf.Variable(0, trainable = False)
        logits = resnet.inference(self.x_data, n_residual_block, False)
        valid_logits = resnet.inference(self.x_valid, n_residual_block, True)  # while reuse = True, share same variables with training model        
        
        train_loss = self.loss(logits, self.y_data)
        regu_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.total_loss = tf.add_n([train_loss] + regu_loss)
        prediction = tf.nn.softmax(logits)
        self.train_top_k_err = self.top_k_error(prediction, self.y_data, 1) # tensor type error
        self.train_op, self.train_ema_op = self.train_operation(self.total_loss, self.train_top_k_err, global_step)

        self.valid_loss = self.loss(valid_logits, self.y_valid)
        valid_prediction = tf.nn.softmax(valid_logits)
        self.valid_top_k_err = self.top_k_error(valid_prediction, self.y_valid, 1)
        self.valid_op = self.val_op(validate_step, self.valid_loss, self.valid_top_k_err)
        
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
        ema = tf.train.ExponentialMovingAverage(ema_decay, global_step)
        train_ema_op = ema.apply([total_loss, top_1_err])
        train_op = tf.train.AdamOptimizer(self.learn_rate).minimize(total_loss, global_step = global_step)
   
        return train_op, train_ema_op     
    
    def val_op(self, val_step, val_loss, val_top_err):
        '''
        Define validation operations
        val_step: validation step, size [1] tensor
        val_loss: validation loss, size[1] tensor
        val_top_err: validation top-1 error, size [1] tensor
        return: validation operation
        '''
        # ema help calculate moving average of valid loss
        ema = tf.train.ExponentialMovingAverage(0.0, val_step)
        ema2 = tf.train.ExponentialMovingAverage(0.95, val_step)
        
        valid_op = tf.group(val_step.assign_add(1), ema.apply([val_top_err, val_loss]),
                          ema2.apply([val_top_err, val_loss]))
        
        return valid_op
    
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
        Help generate augmented train data by random crop, flips, whitening
        train_data: 4D numpy array
        train_labels: 1D array
        return: 4D augmented array, 1D augmented array
        '''
        # randomly choose a batch to augment
        offset = np.random.choice(total_data - train_batch_size) # random choose a start point to count batch, this batch would be augmented
        batch_data = train_data[offset:offset + train_batch_size, :]
        batch_data = cifar.random_crop_flip(batch_data, padding_size) 
        whitened_data = cifar.whiten_img(batch_data) # augmented data batch
        batch_label = train_labels[offset:offset + train_batch_size]
        
        return whitened_data, batch_label
    
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
        saver = tf.train.Saver(tf.global_variables())
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
            train_data, train_labels = self.generate_augment_train_batch(all_data, all_labels, train_batch_size)
            val_data, val_labels = self.generate_vali_batch(valid_data, valid_labels, test_batch_size)
            '''In each epoch, only train with one single batch?? '''
            
            # Do validation before training, use test data here. This validate result could perform as test performance
            if iterate % report_freq == 0:
                _, valid_error, valid_loss = sess.run([self.valid_op, self.valid_top_k_err, self.valid_loss],
                                                   {self.x_valid: val_data, self.y_valid:val_labels})
                
                val_err_list.append(valid_error)
            
            
            start_time = time.time()
        
            _, _, training_loss, top_1_err_rate = sess.run([self.train_op, self.train_ema_op, self.total_loss, self.train_top_k_err], 
                                                        {self.x_data: train_data,
                                                         self.y_data: train_labels})
    
            duration = time.time() - start_time # duration for one training epoch
            
            if iterate % report_freq == 0:
            # Summary for tensorboard result
                summary_str = sess.run(summary_op, {self.x_data: train_data,
                                                self.y_data: train_labels,
                                                self.x_valid: val_data,
                                                self.y_valid: val_labels})
                summary_writer.add_summary(summary_str, iterate)
            
                
                print('%s: Step: %d,  loss: %.4f'%(datetime.now(), iterate, training_loss))
                print('Train top-1 error rate: ', top_1_err_rate)
                print('Validation loss: ', valid_loss)
                print('Validation top-1 error rate:', valid_error)
            
                step_list.append(iterate)
                train_err.append(training_loss)
                top_1_list.append(top_1_err_rate)
            
            # Save model checkpoints every 100 steps
            if iterate % 50 == 0 or (iterate + 1) == train_epoch:
                saver.save(sess, model_save_path, global_step = iterate)
                
                # save train error dataframe
                df = pd.DataFrame(data = {'steps: ':step_list, 'training loss:': 
                                          train_err, 'top_1_err:':top_1_list, 'valid error: ':valid_error})
    
                df.to_csv(model_save_path + '_error_list.csv')
       
    def test(self, test_data):
        '''
        Show testing performance, finish preprocessing in advance
        test_data: 4D input image array
        return: softmax probability with shape [total test samples, num_labels]
        '''
        # Use 10000 test batch to do testing
        num_test = len(test_data)
        num_batches = num_test // test_batch_size
        remain_img = num_test - num_batches*test_batch_size
        print('Total test images: ', num_batches*test_batch_size)
        
        # test image and label placeholder
        self.img_test_batch = tf.placeholder(tf.float32, [test_batch_size, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
        #label_test = tf.placeholder(tf.int32, [None, num_class])
        
        logits = resnet.inference(self.img_test_batch, n_residual_block, True)
        prediction = tf.nn.softmax(logits)
        
        '''Load in the pre-trained weights'''
        sess = tf.Session()
        saver = tf.train.import_meta_graph('./checkpoint/model.meta') # load model architecture
        model_file = tf.train.latest_checkpoint(model_save_path)  # load weights
        saver.restore(sess, model_file)
        
        print('Model restored from: ', model_save_path)
        
        predict_array = np.array([]).reshape([-1, num_class]) # to store batch of prediction, final size = [batch size, num of class]
        # test by batch
        for step in range(num_batches):
            if step % 10 == 0:
                print('{} batches finished!'.format(step))
            
            offset = step*test_batch_size # start offset to get test batch
            test_img_batch = test_data[offset:offset + test_batch_size, :]
            batch_prediction = sess.run(prediction, {self.img_test_batch: test_img_batch})
            predict_array = np.concatenate((predict_array, batch_prediction))
        
        # If test_batch_size is not a divisor of num_test_images
        if remain_img != 0:
            self.img_test_batch = tf.placeholder(tf.float32, [remain_img, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL])
            
            logits = resnet.inference(self.img_test_batch, n_residual_block, True)
            prediction = tf.nn.softmax(logits)
            remain_batch = test_data[-remain_img:, :] # the left images
            batch_predict = sess.run(prediction, {self.img_test_batch: remain_batch})
            
            predict_array = np.concatenate((predict_array, batch_predict))
            
            
        return predict_array
        
        
        

train = Train()
train.train()

            
        
        
        
        
        
    
    
    
    
    
    