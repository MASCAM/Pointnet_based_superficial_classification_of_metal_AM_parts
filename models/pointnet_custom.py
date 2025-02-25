import tensorflow as tf
import numpy as np
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size,))
    return pointclouds_pl, labels_pl

def get_model(point_cloud, is_training, bn_decay=None):
    """ Classification PointNet for 3-class problem, input is BxNx3, output Bx3 """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    
    # Normalize point cloud to unit sphere
    point_cloud_normalized = tf.divide(
        point_cloud - tf.reduce_mean(point_cloud, axis=1, keepdims=True),
        tf.maximum(
            tf.sqrt(tf.reduce_sum(tf.square(point_cloud), axis=2, keepdims=True)),
            1e-8
        )
    )
    
    print(f"Input point cloud shape: {point_cloud.shape}")
    
    # Input transformation
    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_cloud_normalized, is_training, bn_decay, K=3)
    point_cloud_transformed = tf.matmul(point_cloud_normalized, transform)
    
    # Point functions (MLP implemented as conv2d)
    net = tf.expand_dims(point_cloud_transformed, -1)
    
    # First set of layers with gradient clipping
    net = tf_util.conv2d(net, 64, [1,3], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
    
    # Feature transform with gradient clipping
    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    
    # Second set of layers with gradient clipping
    net = tf_util.conv2d(point_feat, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    
    # Global features
    global_feat = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
    
    # Fully connected layers with gradient clipping and stronger regularization
    net = tf.reshape(global_feat, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp2')
    
    # Final layer with proper initialization
    with tf.compat.v1.variable_scope('fc3') as sc:
        weights = tf.compat.v1.get_variable('weights', [256, 3],
                                          initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.01))
        biases = tf.compat.v1.get_variable('biases', [3], 
                                         initializer=tf.compat.v1.constant_initializer(0.0))
        net = tf.matmul(net, weights) + biases
    
    print(f"Output prediction shape: {net.shape}")
    
    return net, end_points

def get_loss(pred, label, end_points, reg_weight=0.001):
    """ 
    pred: B*NUM_CLASSES - predictions for each point cloud (batch_size, 3)
    label: B - labels for each point cloud (batch_size,)
    """
    # Add epsilon to avoid numerical instability
    epsilon = 1e-10
    pred = tf.clip_by_value(pred, -1e3, 1e3)  # Clip predictions to avoid extreme values
    
    # Calculate classification loss with label smoothing
    smooth_labels = tf.one_hot(label, depth=3, on_value=0.9, off_value=0.1/2)
    logits = tf.nn.softmax(pred + epsilon)
    classify_loss = -tf.reduce_mean(tf.reduce_sum(smooth_labels * tf.math.log(logits + epsilon), axis=1))
    
    # Enforce the transformation as orthogonal matrix with gradient clipping
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1]
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff) 
    
    # Combine losses with gradient clipping
    total_loss = classify_loss + mat_diff_loss * reg_weight
    total_loss = tf.clip_by_value(total_loss, -1e3, 1e3)
    
    tf.compat.v1.summary.scalar('classify_loss', classify_loss)
    tf.compat.v1.summary.scalar('mat_diff_loss', mat_diff_loss)
    tf.compat.v1.summary.scalar('total_loss', total_loss)
    
    return total_loss

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (XYZ) Transform Net, input is BxNx3 gray image
        Return:
            Transformation matrix of size 3xK """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,3],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_XYZ') as sc:
        # Initialize weights and biases
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
        
        # Create identity matrix
        identity = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        
        # Compute transform
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)
        transform = transform + identity  # Add identity instead of using +=

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0]
    num_point = inputs.get_shape()[1]

    net = tf_util.conv2d(inputs, 64, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1],
                             padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        # Initialize weights and biases
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                          initializer=tf.zeros_initializer(),
                                          dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                         initializer=tf.zeros_initializer(),
                                         dtype=tf.float32)
        
        # Create identity matrix
        identity = tf.constant(np.eye(K).flatten(), dtype=tf.float32)
        
        # Compute transform
        transform = tf.matmul(net, weights)
        transform = tf.nn.bias_add(transform, biases)
        transform = transform + identity  # Add identity instead of using +=

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform 