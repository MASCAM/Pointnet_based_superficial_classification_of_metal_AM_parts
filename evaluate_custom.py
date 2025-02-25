import os
import sys
import numpy as np
import h5py
import tensorflow as tf
import importlib
import argparse
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# Add paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
import provider

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_with_normals', help='Model name [default: pointnet_with_normals]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during evaluation [default: 16]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--model_path', default='log/pointnet_with_normals_20250126-193627/best_model.ckpt', 
                    help='Path to the trained model checkpoint')
parser.add_argument('--test_file', default='Pre_Processed_data/3_classes/Simulated/HDF5/train_files.txt',
                    help='Path to the test files list')
FLAGS = parser.parse_args()

# Constants
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
#CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4']  # Convert to strings for classification report
CLASS_NAMES = ['Class 0', 'Class 1', 'Class 2' ]#, 'Class 3', 'Class 4']  # Convert to strings for classification report

def plot_confusion_matrix(y_true, y_pred, output_dir):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=CLASS_NAMES,
                yticklabels=CLASS_NAMES)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
    plt.close()

def load_h5_data(h5_files):
    """Load and combine data from multiple H5 files"""
    all_data = []
    all_normals = []
    all_labels = []
    
    print("\nDebug: Loading H5 files...")
    for h5_file in h5_files:
        print(f"Loading file: {h5_file}")
        with h5py.File(h5_file, 'r') as f:
            data = f['data'][:]
            normals = f['normal'][:]
            labels = f['label'][:]
            
            print(f"File shapes before processing - Data: {data.shape}, Normals: {normals.shape}, Labels: {labels.shape}")
            print(f"Label values in file: {np.unique(labels)}")
            
            # Reshape data if needed
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=0)
                normals = np.expand_dims(normals, axis=0)
                labels = np.expand_dims(labels, axis=0)
            
            all_data.append(data)
            all_normals.append(normals)
            all_labels.append(labels)
    
    # Combine all data
    combined_data = np.concatenate(all_data, axis=0)
    combined_normals = np.concatenate(all_normals, axis=0)
    combined_labels = np.concatenate(all_labels, axis=0)
    
    print(f"\nDebug: Combined shapes - Data: {combined_data.shape}, Normals: {combined_normals.shape}, Labels: {combined_labels.shape}")
    print(f"Debug: Combined label values: {np.unique(combined_labels)}")
    
    # Convert per-point labels to per-cloud labels if needed
    if len(combined_labels.shape) > 1 and combined_labels.shape[1] == NUM_POINT:
        print("Debug: Converting per-point labels to per-cloud labels")
        cloud_labels = []
        for i in range(combined_labels.shape[0]):
            unique, counts = np.unique(combined_labels[i], return_counts=True)
            most_common_label = unique[np.argmax(counts)]
            cloud_labels.append(most_common_label)
        combined_labels = np.array(cloud_labels)
        print(f"Debug: After conversion - Labels shape: {combined_labels.shape}")
        print(f"Debug: Final label values: {np.unique(combined_labels)}")
    
    return combined_data, combined_normals, combined_labels

def augment_batch_data(batch_data, batch_normals):
    """Apply consistent augmentation to both points and normals"""
    # Rotate points and normals
    rotated_data = provider.rotate_point_cloud(batch_data)
    rotated_normals = provider.rotate_point_cloud(batch_normals)
    
    # Jitter only points
    jittered_data = provider.jitter_point_cloud(rotated_data)
    
    # Scale points
    scale = np.random.uniform(0.8, 1.2, (batch_data.shape[0], 1, 1))
    scaled_data = jittered_data * scale
    
    return scaled_data, rotated_normals

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            print("\nDebug: Setting up model...")
            pointclouds_pl, normals_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.compat.v1.placeholder(tf.bool, shape=())
            
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, normals_pl, is_training_pl)
            loss = MODEL.get_loss(pred, labels_pl, end_points)
            pred_softmax = tf.nn.softmax(pred)
            
            # Print model information
            print("Debug: Model placeholders shapes:")
            print(f"- pointclouds_pl: {pointclouds_pl.shape}")
            print(f"- normals_pl: {normals_pl.shape}")
            print(f"- labels_pl: {labels_pl.shape}")
            
            saver = tf.compat.v1.train.Saver()
        
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        
        with tf.compat.v1.Session(config=config) as sess:
            # Restore variables from disk
            saver.restore(sess, FLAGS.model_path)
            print(f"Model restored from {FLAGS.model_path}")
            
            # Load test file list
            print(f"Loading test files list from {FLAGS.test_file}")
            with open(FLAGS.test_file, 'r') as f:
                test_files = [line.strip() for line in f]
            print(f"Found {len(test_files)} test files")
            
            # Load and combine all test data
            print("Loading test data...")
            data, normals, labels = load_h5_data(test_files)
            
            print(f"Data shape: {data.shape}")
            print(f"Normals shape: {normals.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Unique labels: {np.unique(labels)}")
            print(f"Label distribution: {np.bincount(labels)}")
            
            # Add debug prints for normalization
            print("\nDebug: Data statistics before normalization:")
            print(f"Points - Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
            print(f"Points - Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
            print(f"Normals - Mean: {np.mean(normals):.4f}, Std: {np.std(normals):.4f}")
            print(f"Normals - Min: {np.min(normals):.4f}, Max: {np.max(normals):.4f}")
            
            # Normalize the point clouds (exactly as in training)
            data = data - np.mean(data, axis=1, keepdims=True)
            data = data / (np.maximum(np.sqrt(np.sum(np.square(data), axis=2, keepdims=True)), 1e-8))

            # Normalize the normal vectors (exactly as in training)
            normals = normals / (np.maximum(np.sqrt(np.sum(np.square(normals), axis=2, keepdims=True)), 1e-8))

            print("\nDebug: Data statistics after normalization:")
            print(f"Points - Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")
            print(f"Points - Min: {np.min(data):.4f}, Max: {np.max(data):.4f}")
            print(f"Normals - Mean: {np.mean(normals):.4f}, Std: {np.std(normals):.4f}")
            print(f"Normals - Min: {np.min(normals):.4f}, Max: {np.max(normals):.4f}")
            
            # Initialize statistics
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(len(CLASS_NAMES))]
            total_correct_class = [0 for _ in range(len(CLASS_NAMES))]
            #total_seen_class = [0, 0]
            #total_correct_class = [0, 0]
            
            all_preds = []
            all_labels = []
            
            # Evaluate all data
            num_batches = data.shape[0] // BATCH_SIZE
            
            print("\nStarting evaluation...")
            for batch_idx in range(num_batches):
                start_idx = batch_idx * BATCH_SIZE
                end_idx = min((batch_idx + 1) * BATCH_SIZE, data.shape[0])
                
                batch_data = data[start_idx:end_idx, :, :]
                batch_normals = normals[start_idx:end_idx, :, :]
                batch_labels = labels[start_idx:end_idx]
                
                # Debug first batch
                if batch_idx == 0:
                    print(f"\nDebug: First batch shapes:")
                    print(f"Batch data: {batch_data.shape}")
                    print(f"Batch normals: {batch_normals.shape}")
                    print(f"Batch labels: {batch_labels.shape}")
                    print(f"Batch labels values: {batch_labels}")
                
                feed_dict = {
                    pointclouds_pl: batch_data,
                    normals_pl: batch_normals,
                    labels_pl: batch_labels,
                    is_training_pl: False
                }
                
                # Get predictions and probabilities
                pred_val = sess.run(pred, feed_dict={pointclouds_pl: batch_data,
                                                    normals_pl: batch_normals,
                                                    is_training_pl: False})
                pred_softmax = sess.run(tf.nn.softmax(pred_val))
                pred_val_max = np.argmax(pred_softmax, axis=1)
                
                # Debug first batch predictions
                if batch_idx == 0:
                    print("\nDebug: First batch prediction details:")
                    print("Prediction distribution:", np.bincount(pred_val_max, minlength=len(CLASS_NAMES)))
                    print("True labels:", batch_labels)
                    print("Predicted labels:", pred_val_max)
                    print("Prediction probabilities:")
                    print(pred_softmax)
                    
                    # Print confidence statistics
                    confidence_scores = np.max(pred_softmax, axis=1)
                    print("\nConfidence Statistics:")
                    print(f"Mean confidence: {np.mean(confidence_scores):.4f}")
                    print(f"Min confidence: {np.min(confidence_scores):.4f}")
                    print(f"Max confidence: {np.max(confidence_scores):.4f}")
                    print(f"Std confidence: {np.std(confidence_scores):.4f}")
                    
                    # Print accuracy for high confidence predictions
                    high_conf_mask = confidence_scores > 0.9
                    if np.any(high_conf_mask):
                        high_conf_acc = np.mean(pred_val_max[high_conf_mask] == batch_labels[high_conf_mask])
                        print(f"\nAccuracy for predictions with confidence > 0.9: {high_conf_acc:.4f}")
                        print(f"Number of high confidence predictions: {np.sum(high_conf_mask)}")
                
                all_preds.extend(pred_val_max)
                all_labels.extend(batch_labels)
                
                # Update statistics
                for i in range(pred_val_max.shape[0]):
                    total_seen += 1
                    true_label = batch_labels[i]
                    pred_label = pred_val_max[i]
                    total_seen_class[true_label] += 1
                    total_correct_class[true_label] += (pred_label == true_label)
                    total_correct += (pred_label == true_label)
                
                if batch_idx % 20 == 0:
                    print(f"Processed batch {batch_idx}/{num_batches}")
                    current_accuracy = total_correct / float(total_seen)
                    print(f"Current accuracy: {current_accuracy:.4f}")
                
                # Print feature statistics for first batch
                if batch_idx == 0:
                    # Get features from the second-to-last layer (fc2)
                    fc2_features = sess.run('fc2/BiasAdd:0', feed_dict={pointclouds_pl: batch_data,
                                                                           normals_pl: batch_normals,
                                                                           is_training_pl: False})
                    print("\nDebug: Feature statistics for first batch:")
                    print(f"FC2 feature shape: {fc2_features.shape}")
                    print(f"FC2 feature mean: {np.mean(fc2_features):.4f}")
                    print(f"FC2 feature std: {np.std(fc2_features):.4f}")
                    print(f"FC2 feature min: {np.min(fc2_features):.4f}")
                    print(f"FC2 feature max: {np.max(fc2_features):.4f}")
            
            # Calculate metrics
            overall_accuracy = total_correct / float(total_seen) if total_seen != 0 else 0 
            class_accuracies = [float(total_correct_class[i])/float(total_seen_class[i])
                              if total_seen_class[i] != 0 else 0 for i in range(len(CLASS_NAMES))]
            
            # Output directory
            output_dir = os.path.dirname(FLAGS.model_path)
            
            # Save detailed results
            with open(os.path.join(output_dir, 'evaluation_results.txt'), 'w') as f:
                f.write("Evaluation Results:\n")
                f.write(f"Overall Accuracy: {overall_accuracy:.4f}\n\n")
                f.write("Per-class Accuracies:\n")
                for i in range(len(CLASS_NAMES)):
                    f.write(f"{CLASS_NAMES[i]}: {class_accuracies[i]:.4f} ({total_correct_class[i]}/{total_seen_class[i] if total_seen_class[i] != 0 else 1})\n")
                
                f.write("\nClassification Report:\n")
                f.write(classification_report(all_labels, all_preds, target_names=CLASS_NAMES[1:]))
                
                # Add confusion matrix to text file
                f.write("\nConfusion Matrix:\n")
                cm = confusion_matrix(all_labels, all_preds)
                f.write(str(cm))
            
            # Plot confusion matrix
            plot_confusion_matrix(all_labels, all_preds, output_dir)
            
            # Save predictions
            predictions_file = os.path.join(output_dir, 'predictions.txt')
            np.savetxt(predictions_file, all_preds, fmt='%d')
            print(f"\nResults saved in {output_dir}")
            print(f"- evaluation_results.txt")
            print(f"- confusion_matrix.png")
            print(f"- predictions.txt")

if __name__ == '__main__':
    # Import model
    try:
        MODEL = importlib.import_module(FLAGS.model)
        print(f"Successfully imported model: {FLAGS.model}")
    except ImportError as e:
        print(f"Error importing model {FLAGS.model}: {e}")
        sys.exit(1)
    evaluate() 