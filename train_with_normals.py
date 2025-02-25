import argparse
import numpy as np
import tensorflow as tf
import importlib
import os
import sys
import h5py
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_with_normals', help='Model name [default: pointnet_with_normals]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--max_epoch', type=int, default=201, help='Epoch to run [default: 201]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
FLAGS = parser.parse_args()

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, 'models', FLAGS.model+'.py')

# Create log directory
timestr = datetime.now().strftime("%Y%m%d-%H%M%S")
LOG_DIR = os.path.join(FLAGS.log_dir, FLAGS.model + '_' + timestr)
if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp train_with_normals.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.compat.v1.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.compat.v1.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def augment_batch_data(batch_data, batch_normals):
    rotated_data = provider.rotate_point_cloud(batch_data)
    rotated_normals = provider.rotate_point_cloud(batch_normals)
    jittered_data = provider.jitter_point_cloud(rotated_data)
    
    # Scale points
    scale = np.random.uniform(0.8, 1.2, (BATCH_SIZE, 1, 1))
    scaled_data = jittered_data * scale
    
    # No need to scale normals as they represent directions
    return scaled_data, rotated_normals

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, normals_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            
            # Note the global_step=batch parameter to minimize. 
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.compat.v1.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            pred, end_points = MODEL.get_model(pointclouds_pl, normals_pl, is_training_pl, bn_decay=bn_decay)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            tf.compat.v1.summary.scalar('loss', loss)

            correct = tf.equal(tf.argmax(pred, 1), tf.cast(labels_pl, tf.int64))
            accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
            tf.compat.v1.summary.scalar('accuracy', accuracy)

            print("--- Get training operator")
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.compat.v1.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.compat.v1.train.Saver()
        
        # Create a session
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.compat.v1.Session(config=config)

        # Add summary writers
        merged = tf.compat.v1.summary.merge_all()
        train_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        test_writer = tf.compat.v1.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)

        # Init variables
        init = tf.compat.v1.global_variables_initializer()
        sess.run(init)

        ops = {'pointclouds_pl': pointclouds_pl,
               'normals_pl': normals_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'end_points': end_points}

        best_acc = -1
        
        # Load data files
        TRAIN_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Simulated/HDF5/train_files.txt'))
        TEST_FILES = provider.getDataFiles(os.path.join(BASE_DIR, 'Pre_Processed_data/3_classes/Simulated/HDF5/test_files.txt'))
        
        # Load all training data
        train_points = []
        train_normals = []
        train_labels = []
        for fn in range(len(TRAIN_FILES)):
            print('Loading %s' % TRAIN_FILES[fn])
            with h5py.File(TRAIN_FILES[fn], 'r') as f:
                points = f['data'][:]
                normals = f['normal'][:]
                labels = f['label'][:]
                train_points.append(points)
                train_normals.append(normals)
                train_labels.append(labels)
        train_points = np.concatenate(train_points, axis=0)
        train_normals = np.concatenate(train_normals, axis=0)
        train_labels = np.concatenate(train_labels, axis=0)
        
        # Convert per-point labels to per-cloud labels if needed
        if len(train_labels.shape) > 1 and train_labels.shape[1] == NUM_POINT:
            cloud_labels = []
            for i in range(train_labels.shape[0]):
                unique, counts = np.unique(train_labels[i], return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                cloud_labels.append(most_common_label)
            train_labels = np.array(cloud_labels)
        
        # Load all test data
        test_points = []
        test_normals = []
        test_labels = []
        for fn in range(len(TEST_FILES)):
            print('Loading %s' % TEST_FILES[fn])
            with h5py.File(TEST_FILES[fn], 'r') as f:
                points = f['data'][:]
                normals = f['normal'][:]
                labels = f['label'][:]
                test_points.append(points)
                test_normals.append(normals)
                test_labels.append(labels)
        test_points = np.concatenate(test_points, axis=0)
        test_normals = np.concatenate(test_normals, axis=0)
        test_labels = np.concatenate(test_labels, axis=0)
        
        # Convert per-point labels to per-cloud labels if needed
        if len(test_labels.shape) > 1 and test_labels.shape[1] == NUM_POINT:
            cloud_labels = []
            for i in range(test_labels.shape[0]):
                unique, counts = np.unique(test_labels[i], return_counts=True)
                most_common_label = unique[np.argmax(counts)]
                cloud_labels.append(most_common_label)
            test_labels = np.array(cloud_labels)
            
        print('train points shape:', train_points.shape)
        print('train normals shape:', train_normals.shape)
        print('train labels shape:', train_labels.shape)
        print('test points shape:', test_points.shape)
        print('test normals shape:', test_normals.shape)
        print('test labels shape:', test_labels.shape)

        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            
            train_one_epoch(sess, ops, train_points, train_normals, train_labels, train_writer)
            eval_acc = eval_one_epoch(sess, ops, test_points, test_normals, test_labels, test_writer)
            
            # Save the best model
            if eval_acc > best_acc:
                best_acc = eval_acc
                save_path = saver.save(sess, os.path.join(LOG_DIR, "best_model.ckpt"))
                log_string("Best model saved in file: %s" % save_path)
            
            # Save the model periodically
            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"), global_step=epoch)
                log_string("Model saved in file: %s" % save_path)

def train_one_epoch(sess, ops, train_points, train_normals, train_labels, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    # Shuffle train samples
    num_batches = len(train_points) // BATCH_SIZE
    
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    
    perm = np.random.permutation(len(train_points))
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        
        batch_data = train_points[perm[start_idx:end_idx]]
        batch_normals = train_normals[perm[start_idx:end_idx]]
        batch_label = train_labels[perm[start_idx:end_idx]]
        
        # Augment batched point clouds by rotation, jittering, and scaling
        aug_data, aug_normals = augment_batch_data(batch_data, batch_normals)
        
        feed_dict = {ops['pointclouds_pl']: aug_data,
                     ops['normals_pl']: aug_normals,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['train_op'], ops['loss'], ops['pred']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:BATCH_SIZE] == batch_label[0:BATCH_SIZE])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        
        if (batch_idx+1)%50 == 0:
            log_string(' ---- batch: %03d ----' % (batch_idx+1))
            log_string('mean loss: %f' % (loss_sum / 50))
            log_string('accuracy: %f' % (total_correct / float(total_seen)))
            total_correct = 0
            total_seen = 0
            loss_sum = 0

def eval_one_epoch(sess, ops, test_points, test_normals, test_labels, test_writer):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(3)]
    total_correct_class = [0 for _ in range(3)]
    
    num_batches = len(test_points) // BATCH_SIZE
    
    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx + 1) * BATCH_SIZE
        
        batch_data = test_points[start_idx:end_idx]
        batch_normals = test_normals[start_idx:end_idx]
        batch_label = test_labels[start_idx:end_idx]
        
        feed_dict = {ops['pointclouds_pl']: batch_data,
                     ops['normals_pl']: batch_normals,
                     ops['labels_pl']: batch_label,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'],
            ops['loss'], ops['pred']], feed_dict=feed_dict)
        test_writer.add_summary(summary, step)
        
        pred_val = np.argmax(pred_val, 1)
        correct = np.sum(pred_val[0:BATCH_SIZE] == batch_label[0:BATCH_SIZE])
        total_correct += correct
        total_seen += BATCH_SIZE
        loss_sum += loss_val
        
        for i in range(BATCH_SIZE):
            l = batch_label[i]
            total_seen_class[l] += 1
            total_correct_class[l] += (pred_val[i] == l)
    
    log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
    log_string('eval accuracy: %f'% (total_correct / float(total_seen)))
    log_string('eval avg class acc: %f' % (np.mean(np.array(total_correct_class)/np.array(total_seen_class, dtype=np.float64))))
    EPOCH_CNT += 1
    return total_correct/float(total_seen)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train() 