import tensorflow as tf
import numpy as np

# This model is for cifar-10 data
num_class = 10

def activation_summary(x):
    '''
    :param x: A Tensor
    :return: Add histogram summary and scalar summary of the sparsity of the tensor
    '''
    tensor_name = x.op.name
    tf.summary.histogram(tensor_name + '/activations', x)
    tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

def create_variable(name, shape, initializer = None):
    '''
    By this function, we don't need to redefine when create a new variable
    name: variable name
    shape: new created variable shape
    initializer: defined for get_variable
    '''
    if initializer == None:
        initializer = tf.truncated_normal(shape, stddev = 0.01)
    new_variable = tf.get_variable(name, shape = shape, initializer = initializer)
    return new_variable

def output_layer(input_layer, num_class):
    '''
    input: 2D tensor, the input to the final output layer
    output: Y = wX + b
    '''
    input_dim = input_layer.get_shape().aslist()[-1]
    w_out = create_variable('w_out',[input_dim, num_class])
    b_out = create_variable('b_out', [num_class], tf.zeros_initializer())
    output = tf.matmul(input_layer, w_out) + b_out
    
    return output

BN_EPSILON = 0.001
def batch_norm(input_layer, dim):
    '''
    Define batch normalization for input of a layer
    input: layer to input after BN
    '''
    #dimension = input_layer.get_shape().aslist()[-1]
    mean, variance = tf.nn.moments(input_layer, axes = [0, 1, 2]) 
    # calculate mean  & variance over batch, width, height for BN (don't do along channels)
    #beta = tf.Variable('beta', dimension, tf.float32, initializer = tf.constant_initializer(0.0, tf.float32))
    #gamma = tf.constant(1.0)
    bn_layer = tf.nn.batch_normalization(input_layer, mean, variance, None, None, tf.constant(BN_EPSILON))
    
    return bn_layer

def conv_bn_relu(input_layer, filter_size, stride):
    '''
    Do convolution -> batch normalization -> relu activation
    filter_size: [input_width, input_height, input_channel, output_channel]
    stride: moving step in width and height objection for filter window
    '''
    out_channel = filter_size[-1]
    w_filter = create_variable('conv', filter_size)
    #bias = tf.Variable(tf.zeros(output_dim))
    conv_out = tf.nn.conv2d(input_layer, w_filter, [1, stride, stride, 1], padding = 'SAME') 
    bn_out = batch_norm(conv_out, out_channel)
    activation = tf.nn.relu(bn_out)
    
    return activation
    
def bn_relu_conv(input_layer, filter_size, stride):
    '''
    Do batch_norm -> relu function -> convolution sequentially
    '''
    out_channel = filter_size[-1]
    bn_out = batch_norm(input_layer, out_channel)
    activation = tf.nn.relu(bn_out)
    w_filter = create_variable('conv', filter_size)
    #bias = tf.Variable(tf.zeros(output_dim))
    conv_out = tf.nn.conv2d(activation, w_filter, [1, stride, stride, 1], padding = 'SAME') 
    
    return conv_out

def residual_block(input_layer, output_channel, first_block = False):
    '''
    Define redisual block in ResNet, hear configure 2 conv. layers in a residual block
    input: 4D tensor
    output_channel: output channel of residual block, may shrink the channel or remain the same
    first_block: True is this is the first block of whole network
    return: last layer output of the block
    '''    
    in_channel = input_layer.get_shape().aslist()[-1]
    # along residual block, channel doublely increased. We reduce the image size to maintain data size
    # during coding, use output channel to determine if output feature map size changed
    if in_channel * 2 == output_channel:
        increase_dim = True
        stride = 2  # designe rule 1: channel doubled, indicate that feature map is halved
    elif in_channel == output_channel:
        increase_dim = False
        stride = 1  # channel remain the same, means that output feature map is the same
    else:
        raise ValueError('Output and Input channel does not match in Residual block!')
    
    # first conv1 layer in block    
    with tf.variable_scope('conv1_in_block'):
        if first_block: # first conv1 layer in first block doen't need normalization & relu
           filter = create_variable('conv', [3, 3, in_channel, output_channel])
           conv1 = tf.nn.conv2d(input_layer, filter, [1, 1, 1, 1], padding = 'SAME') # before first block, max pooling have halved the feature map size, so first output map size remain same
        else:
            conv1 = bn_relu_conv(input_layer, [3, 3, in_channel, output_channel], stride) # output of first conv. layer in block
            
    with tf.variable_scope('conv2_in_block'):
        conv2 = bn_relu_conv(conv1, [3, 3, output_channel, output_channel], 1) # design rule 2: same filters in same block

    # Add short cut connection, check channel if it is changed
    # When input channel & output channel don't match, we do zero padding to match
    if increase_dim is True:
        # when channel *2, feature map size divided by 2 for output of block, thus do pooling do decrease map size
        pool_input = tf.nn.avg_pool(input_layer, ksize = [1, 2, 2, 1], stride = [1, stride, stride, 1], padding = 'VALID') # halved the feature map
        padded_input = tf.pad(pool_input, [[0,0], [0,0], [0,0], [in_channel//2, in_channel//2]])  # doubled input x channels by zero padding(channels are all even number)
    else:
        padded_input = input_layer

    return conv2 + padded_input

def inference(input_data_batch, n_blocks, reuse): 
    '''
    Define whole ResNet architecture
    n_blocks: number of residual blocks
    reuse: if build train graph, reuse = True. if build validate or test graph, reuse = False.
    return: output of network, logits.
    '''
    layers = [] # Whole stacked residual block + output layer is stacked in layers
    with tf.variable_scope('conv0', reuse = reuse): # reuse these variables when call the pre-trained model
        conv0 = conv_bn_relu(input_data_batch, [3, 3, 3, 16], 1)
        activation_summary(conv0)
        layers.append(conv0)
        
    # Here we use same blockes for each conv. residual block, assume each conv. blockes have same number
    for i in range(n_blocks):
        with tf.variable_scope('conv1_%d'%i, reuse = reuse): # different name for different conv. layer in a conv. residual block, this is for we to distinguish get_variable() variable
            if i == 0:
                conv1 = residual_block(layers[-1], 16, first_block = True)
            else:
                conv1 = residual_block(layers[-1], 16)
            activation_summary(conv1)
            layers.append(conv1)
    
    # conv2 block(filter number might change)
    for i in range(n_blocks):
        with tf.variable_scope('conv2_%d'%i, reuse = reuse):
            conv2 = residual_block(layers[-1], 32)
            activation_summary(conv2)
            layers.append(conv2)
            
    # conv3 block
    for i in range(n_blocks):
        with tf.variable_scope('conv3_%d'%i, reuse = reuse):
            conv3 = residual_block(layers[-1], 64)
            activation_summary(conv3)
            layers.append(conv3)
        assert conv3.get_shape().as_list()[1:] == [8, 8, 64] # add an assertion if output shape is abnormal
    
    with tf.variable_scope('fc', reuse = reuse):
        in_channel = layers[-1].get_shape().aslist()[-1]
        bn_layer = batch_norm(layers[-1], in_channel)
        relu_layer = tf.nn.relu(bn_layer)
        global_pool = tf.reduce_mean(relu_layer, [1, 2]) # global pooling: pool along whole feature map
        assert global_pool.get_shape().as_list()[-1] == 64 # assert output channel correct
        output = output_layer(global_pool, num_class)
        layers.append(output)
        
    return layers[-1] 
    

def test_graph(train_dir='logs'):
    '''
    Run this function to look at the graph structure on tensorboard. A fast way!
    :param train_dir:
    '''
    input_tensor = tf.constant(np.ones([128, 32, 32, 3]), dtype=tf.float32)
    result = inference(input_tensor, 2, reuse=False)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)
    
    
    
    
    
    
    
    






