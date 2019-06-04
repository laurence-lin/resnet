import numpy as np
import pickle
import os
import math
import imgaug.augmenters as iaa

# floyd data name be called: my_data
data_dir = '/my_data/'  # contains 5 batches of 50000 training data & 1 test batch of 10000 testing data
IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNEL = 3
total_sample = 50000

def horizontal_flip(img, axis):
    '''
    A single image flip horizontally or vertically by 50% probability
    img: 3D image array
    axis: 0 for vertical flip, 1 for horizontal flip
    return: 3D image array after flip
    '''
    if axis == 0:
       flip = iaa.Flipud(1)     # vertical flip
    elif axis == 1:
       flip = iaa.Fliplr(1)     # horizontal flip
    possibility = np.random.randint(low = 0, high = 2) # generate random integer from 0, 1
    if possibility == 0: # for 0.5 possibility
        img = flip.augment_image(img)
    return img

def whiten_img(img_batch):
    '''
    perform image whitening: abandon unnecessary features; preserve important features
    img: 4D batches of image array
    return: images after whitened
    '''
    for i in range(len(img_batch)):
        mean = np.mean(img_batch[i, :])
        #  in case std = 0, we should adjust the std value, so give minimum value for it (give std an minimal value in case)
        std = np.max( [np.std(img_batch[i, :]), 1.0/np.sqrt(IMG_HEIGHT*IMG_WIDTH*IMG_CHANNEL)] ) 
        img_batch[i, :] = (img_batch[i, :] - mean)/std
    return img_batch

def transform_image(img_batch): # convert to batch*32*32*3 size
    '''
    Reshape 2D batch data to 4D image 
    img_batch: 2D array [batch, 3072]
    return: 4D image array [batch, height, width, channel]
    '''
    x = []
    for i in range(img_batch.shape[0]):
        ch1 = img_batch[i, 0:1024].reshape((32, 32))
        ch2 = img_batch[i, 1024:2048].reshape((32, 32))
        ch3 = img_batch[i, 2048:3072].reshape((32, 32))
        img = np.dstack((ch1, ch2, ch3)) # stack array in depth wise(third dimension)
        x.append(img) # insert each image of size: [1, 32, 32, 3]
    
    x = np.array(x)
    return x/255 # image normalization

def random_crop_flip(batch_data, padding_size):
    '''
    Randomly crop and flip a batch of images
    batch_data: 4D batch image data, which is a list with len() = batch size
    padding_size:int., number of zero padding layer to each side
    return: randomly cropped and flipped image batch
    '''
    crop_batch = np.zeros(len(batch_data)*IMG_HEIGHT*IMG_WIDTH*IMG_CHANNEL).reshape(
            len(batch_data), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNEL) 
    # Do cropping: after cropping, the left that is not cropped in crop_batch is all zero padding
    for i in range(len(batch_data)):
        x_offset = np.random.randint(low = 0, high = 2*padding_size) # random index to crop from image
        y_offset = np.random.randint(low = 0, high = 2*padding_size)
        crop_batch[i, :] = batch_data[i, :][x_offset:x_offset+IMG_HEIGHT, y_offset:y_offset+IMG_WIDTH, :]
        # flip by possibility
        crop_batch[i, :] = horizontal_flip(crop_batch[i, :], axis = 1)
        
    return crop_batch

def cifar(data_path, total_sample):
    '''
    Create cifar dataset from data path
    data_path: path list of the data batches
    return: 4D array image, 1D array labels
    '''
    # create data batch paths
    # input data_path should be directly encode and extract data here
    data = []
    label = []
    for path in data_path:  
       with open(path, 'rb') as file: # open bytes encoded file: cifar
           d = pickle.load(file, encoding = 'bytes')  # retrieve content of file by loading, output an dictionary
           # extract data & labels from dictionary
       images = np.array(d[b'data'])  # batches*3072 
       labels = np.array(d[b'labels']) # batches
       
       images = transform_image(images) # convert to batch*32*32*3
       data.extend(images)
       label.extend(labels)
    
    data = np.array(data)
    label = np.array(label)
    
    # Do shuffling for all training data
    shuffle = np.random.permutation(data.shape[0])
    data = data[shuffle, :].astype(np.float32)
    label = label[shuffle]
       
    return data, label
    
def get_batch(data, label, batch_size):
    '''
    create batches of data from whole training set

    '''
    total_sample = 50000 # for cifar-10 training data samples
    data_batch = []
    label_batch = []
    for i in range(math.ceil(total_sample/batch_size)): # if left sample is not enough one batch, train the left samples
        if label[i*batch_size:].shape[0] < batch_size:  # if left sample is not enough
           data_batch.append(data[i*batch_size:]) # append the left sample
           label_batch.append(label[i*batch_size:])
        else:
            data_batch.append(data[i*batch_size:(i+1)*batch_size])
            label_batch.append(label[i*batch_size:(i+1)*batch_size])
    
    return data_batch, label_batch
    
def train(pad_size): 
    '''
    Function to create train data, get from 50000 training set
    return: 4D numpy array training data, 1D array training labels
    '''  
    file_path = os.listdir(data_dir)
    batch_list = [file for file in file_path if 'data_batch' in file] # 5 batches data list 
    data_path = []
    for batch in batch_list:
        data_path.append(os.path.join(data_dir, batch)) #combine file directory & batch name, create a complete batch name list
   
    train_data, train_labels = cifar(data_path, total_sample)
    padding = ((0, 0), (pad_size, pad_size), (pad_size, pad_size), (0, 0))  # do padding for later cropping
    train_data = np.pad(train_data, pad_width = padding, mode = 'constant', constant_values = 0)
    
    return train_data, train_labels
    
def test(): 
    '''
    Read in the test data, whitening at the same time
    return: 4D numpy array test set, 1D array test label
    '''
    file_path = os.listdir(data_dir)
    test_file = [f for f in file_path if 'test' in f]
    test = os.path.join(data_dir, test_file[0]) # fully test data path name

    # don't need to loop over several batches, so don't go into cifar function
    with open(test, 'rb') as file:
        test_data = pickle.load(file, encoding = 'bytes')
    
    image = np.array(test_data[b'data'])
    test_labels = np.array(test_data[b'labels'])
    test_data = transform_image(image)
    test_data = whiten_img(test_data)
    
    return test_data, test_labels
    
    
    
    
    
    
    