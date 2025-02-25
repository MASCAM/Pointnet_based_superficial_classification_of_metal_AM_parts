import tensorflow as tf
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

def placeholder_inputs(batch_size, num_point):
    """ Return placeholders for inputs and labels """
    pointclouds_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    normals_pl = tf.compat.v1.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.compat.v1.placeholder(tf.int32, shape=(batch_size,))
    return pointclouds_pl, normals_pl, labels_pl

def get_model(point_cloud, normals, is_training, bn_decay=None):
    """ Classification PointNet with normals, input is BxNx3 points and BxNx3 normals, output Bx5 """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]
    end_points = {}
    
    # Normalize point cloud to unit sphere with improved numerical stability
    point_cloud_centered = point_cloud - tf.reduce_mean(point_cloud, axis=1, keepdims=True)
    point_cloud_normalized = tf.divide(
        point_cloud_centered,
        tf.maximum(
            tf.sqrt(tf.reduce_sum(tf.square(point_cloud_centered), axis=2, keepdims=True)),
            1e-8
        )
    )
    
    # Normalize normal vectors with improved numerical stability
    normals_normalized = tf.nn.l2_normalize(normals, axis=2)
    
    # Concatenate points and normals
    point_normal_cloud = tf.concat([point_cloud_normalized, normals_normalized], axis=2)
    
    print(f"Input combined shape: {point_normal_cloud.shape}")
    
    # Input transformation (6D: 3D points + 3D normals)
    with tf.compat.v1.variable_scope('transform_net1') as sc:
        transform = input_transform_net(point_normal_cloud, is_training, bn_decay, K=6)
    point_normal_transformed = tf.matmul(point_normal_cloud, transform)
    
    # Point functions (MLP implemented as conv2d)
    net = tf.expand_dims(point_normal_transformed, -1)
    
    # First set of layers with gradient clipping and increased regularization
    net = tf_util.conv2d(net, 64, [1,6], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay,
                         weight_decay=0.01)
    net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay,
                         weight_decay=0.01)
    
    # Feature transform with orthogonality regularization
    with tf.compat.v1.variable_scope('transform_net2') as sc:
        transform = feature_transform_net(net, is_training, bn_decay, K=64)
        # Add orthogonality regularization
        transform_transpose = tf.transpose(transform, perm=[0,2,1])
        transform_matmul = tf.matmul(transform, transform_transpose)
        transform_identity = tf.eye(64, batch_shape=[batch_size])
        orthogonality_loss = tf.nn.l2_loss(transform_matmul - transform_identity) * 0.001
        tf.compat.v1.add_to_collection('losses', orthogonality_loss)
    
    end_points['transform'] = transform
    net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    point_feat = tf.expand_dims(net_transformed, [2])
    
    # Second set of layers with increased width and regularization
    net = tf_util.conv2d(point_feat, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay,
                         weight_decay=0.01)
    net = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay,
                         weight_decay=0.01)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay,
                         weight_decay=0.01)
    
    # Global features with improved pooling
    global_feat = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='maxpool')
    end_points['global_feat'] = tf.squeeze(global_feat)
    
    # Fully connected layers with stronger regularization and normalization
    net = tf.reshape(global_feat, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay,
                                 weight_decay=0.01)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay,
                                 weight_decay=0.01)
    net = tf_util.batch_norm_for_fc(net, is_training=is_training, scope='bn_fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
    
    # Temperature scaling for confidence calibration
    temperature = 2.0
    logits = tf_util.fully_connected(net, 5, activation_fn=None, scope='fc3') / temperature
    
    print(f"Output prediction shape: {logits.shape}")
    
    return logits, end_points

def get_loss(pred, label, end_points):
    """ Compute classification loss and regularization losses """
    # Classification loss
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.compat.v1.summary.scalar('classify loss', classify_loss)
    
    # Regularization loss is automatically collected
    tf.compat.v1.add_to_collection('losses', classify_loss)
    return tf.add_n(tf.compat.v1.get_collection('losses'), name='total_loss')

def input_transform_net(point_cloud, is_training, bn_decay=None, K=3):
    """ Input (feature) Transform Net, input is BxNx6 gray points and normals
        Return:
            Transformation matrix of size 6xK """
    batch_size = point_cloud.get_shape()[0]
    num_point = point_cloud.get_shape()[1]

    input_image = tf.expand_dims(point_cloud, -1)
    net = tf_util.conv2d(input_image, 64, [1,K], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_XYZ') as sc:
        # Initialize with identity transformation
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                          initializer=tf.compat.v1.zeros_initializer(),
                                          dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                         initializer=tf.compat.v1.zeros_initializer(),
                                         dtype=tf.float32)
        biases = tf.add(biases, tf.constant(np.eye(K).flatten(), dtype=tf.float32))
        transform = tf.matmul(net, weights)
        transform = tf.add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform

def feature_transform_net(inputs, is_training, bn_decay=None, K=64):
    """ Feature Transform Net, input is BxNx1xK
        Return:
            Transformation matrix of size KxK """
    batch_size = inputs.get_shape()[0]
    num_point = inputs.get_shape()[1]

    net = tf_util.conv2d(inputs, 64, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv1', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv2', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='tconv3', bn_decay=bn_decay)
    net = tf_util.max_pool2d(net, [num_point,1], padding='VALID', scope='tmaxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='tfc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='tfc2', bn_decay=bn_decay)

    with tf.compat.v1.variable_scope('transform_feat') as sc:
        # Initialize with identity transformation
        weights = tf.compat.v1.get_variable('weights', [256, K*K],
                                          initializer=tf.compat.v1.zeros_initializer(),
                                          dtype=tf.float32)
        biases = tf.compat.v1.get_variable('biases', [K*K],
                                         initializer=tf.compat.v1.zeros_initializer(),
                                         dtype=tf.float32)
        biases = tf.add(biases, tf.constant(np.eye(K).flatten(), dtype=tf.float32))
        transform = tf.matmul(net, weights)
        transform = tf.add(transform, biases)

    transform = tf.reshape(transform, [batch_size, K, K])
    return transform 