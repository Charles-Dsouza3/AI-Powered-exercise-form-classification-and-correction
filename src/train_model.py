# import os
# import time
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns
# from pathlib import Path
# import json
# import logging
# from typing import Tuple, Dict, List
#
# import tensorflow as tf
# from tensorflow.keras.models import Model, Sequential
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.layers import *
# from tensorflow.keras.callbacks import *
# from tensorflow.keras.regularizers import l2
# from tensorflow.keras import backend as K
#
# from sklearn.model_selection import train_test_split, StratifiedKFold
# from sklearn.metrics import (
#     classification_report, accuracy_score, confusion_matrix,
#     precision_recall_fscore_support, precision_score,
#     recall_score, f1_score, roc_auc_score, roc_curve
# )
# from sklearn.utils.class_weight import compute_class_weight
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
# # --- Configuration ---
# ACTIONS = np.array(['curl', 'press'])  # 2 classes
# SEQUENCE_LENGTH = 30
# NUM_LANDMARKS = 33
# NUM_DIMS = 4
# INPUT_SIZE = NUM_LANDMARKS * NUM_DIMS
# NUM_BIO_FEATURES = 8  # From improved data collection
#
# DATA_DIR = Path.cwd() / "data_2class"
# MODEL_DIR = Path.cwd() / "models"
# LOG_DIR = Path.cwd() / "logs"
# RESULTS_DIR = Path.cwd() / "results"
#
# # Training hyperparameters
# BATCH_SIZE = 16
# EPOCHS = 100
# LEARNING_RATE = 0.001
# USE_CROSS_VALIDATION = False  # Set to True for k-fold CV
# K_FOLDS = 5
#
#
# def ensure_dirs(*dirs: Path):
#     """Create directories if they don't exist."""
#     for d in dirs:
#         d.mkdir(parents=True, exist_ok=True)
#
#
# # --- Attention Mechanism ---
# class AttentionLayer(Layer):
#     """Self-attention mechanism for temporal sequences."""
#
#     def __init__(self, **kwargs):
#         super(AttentionLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.W = self.add_weight(
#             name='attention_weight',
#             shape=(input_shape[-1], input_shape[-1]),
#             initializer='glorot_uniform',
#             trainable=True
#         )
#         self.b = self.add_weight(
#             name='attention_bias',
#             shape=(input_shape[-1],),
#             initializer='zeros',
#             trainable=True
#         )
#         super(AttentionLayer, self).build(input_shape)
#
#     def call(self, x):
#         # Compute attention scores
#         e = K.tanh(K.dot(x, self.W) + self.b)
#         a = K.softmax(e, axis=1)
#         output = x * a
#         return K.sum(output, axis=1)
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], input_shape[-1])
#
#     def get_config(self):
#         return super(AttentionLayer, self).get_config()
#
#
# # --- Data Loading with Metadata ---
# def load_data_with_metadata() -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
#     """Enhanced data loading with metadata filtering."""
#     sequences, labels, metadata_list = [], [], []
#     label_map = {label: idx for idx, label in enumerate(ACTIONS)}
#
#     logger.info("Loading data with quality filtering...")
#
#     for action in ACTIONS:
#         action_path = DATA_DIR / action
#         if not action_path.exists():
#             logger.warning(f"No data found for action {action}")
#             continue
#
#         action_sequences = 0
#         rejected_sequences = 0
#
#         for seq_dir in action_path.glob("*"):
#             if seq_dir.is_dir():
#                 # Load metadata if available
#                 metadata_path = seq_dir / "metadata.json"
#                 metadata = {}
#                 if metadata_path.exists():
#                     with open(metadata_path, 'r') as f:
#                         metadata = json.load(f)
#
#                     # Filter by quality score (if available)
#                     if 'quality_score' in metadata:
#                         if metadata['quality_score'] < 0.8:  # 80% threshold
#                             rejected_sequences += 1
#                             continue
#
#                 # Load sequence
#                 npy_files = sorted(seq_dir.glob("*.npy"))
#
#                 # Exclude metadata.json when counting
#                 npy_files = [f for f in npy_files if f.suffix == '.npy']
#
#                 if len(npy_files) == SEQUENCE_LENGTH:
#                     window = []
#                     valid = True
#
#                     for f in range(SEQUENCE_LENGTH):
#                         frame_path = seq_dir / f"{f}.npy"
#                         if frame_path.exists():
#                             data = np.load(frame_path)
#
#                             # Handle data with biomechanical features
#                             expected_size = INPUT_SIZE + NUM_BIO_FEATURES
#                             if len(data) == expected_size:
#                                 # Data includes bio features
#                                 window.append(data)
#                             elif len(data) == INPUT_SIZE:
#                                 # Old data without bio features, pad with zeros
#                                 padded_data = np.concatenate([data, np.zeros(NUM_BIO_FEATURES)])
#                                 window.append(padded_data)
#                             else:
#                                 valid = False
#                                 break
#                         else:
#                             valid = False
#                             break
#
#                     if valid and len(window) == SEQUENCE_LENGTH:
#                         sequences.append(window)
#                         labels.append(label_map[action])
#                         metadata_list.append(metadata)
#                         action_sequences += 1
#
#         logger.info(
#             f"Loaded {action_sequences} sequences for {action} "
#             f"(rejected {rejected_sequences} low-quality)"
#         )
#
#     if not sequences:
#         logger.error("âŒ No valid sequences found!")
#         return None, None, None
#
#     X = np.array(sequences)
#     y = to_categorical(labels).astype(int)
#
#     logger.info(f"âœ… Total sequences: {len(sequences)}")
#     logger.info(f"Data shape: {X.shape}")
#     logger.info(f"Label distribution: {np.bincount(np.argmax(y, axis=1))}")
#
#     return X, y, metadata_list
#
#
# # --- Advanced Model Architectures ---
# def build_lstm_with_attention() -> Model:
#     """
#     Build LSTM model with self-attention mechanism for exercise detection.
#     Enhanced for bicep curls and shoulder press classification.
#     """
#     inp = Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE + NUM_BIO_FEATURES), name='input')
#
#     # Initial feature normalization
#     x = LayerNormalization()(inp)
#
#     # Temporal feature extraction with Conv1D
#     x = Conv1D(64, 5, activation='relu', padding='same',
#                kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#
#     x = Conv1D(128, 3, activation='relu', padding='same',
#                kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(2)(x)
#     x = Dropout(0.25)(x)
#
#     # Bidirectional LSTM layers with regularization
#     x = Bidirectional(
#         LSTM(256, return_sequences=True, dropout=0.3,
#              recurrent_dropout=0.2, kernel_regularizer=l2(0.001))
#     )(x)
#     x = BatchNormalization()(x)
#
#     x = Bidirectional(
#         LSTM(128, return_sequences=True, dropout=0.3,
#              recurrent_dropout=0.2, kernel_regularizer=l2(0.001))
#     )(x)
#     x = BatchNormalization()(x)
#
#     # Self-attention mechanism
#     attention_output = AttentionLayer(name='attention')(x)
#
#     # Dense classification layers
#     x = Dense(512, activation='relu', kernel_regularizer=l2(0.001))(attention_output)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#
#     x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.4)(x)
#
#     x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
#     x = Dropout(0.3)(x)
#
#     # Output layer - 2 classes
#     out = Dense(len(ACTIONS), activation='softmax', name='output')(x)
#
#     model = Model(inputs=inp, outputs=out, name='LSTM_Attention_Exercise_2Class')
#
#     return model
#
#
# def build_deep_lstm_model() -> Model:
#     """Alternative deeper LSTM model without attention."""
#     inp = Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE + NUM_BIO_FEATURES))
#
#     x = LayerNormalization()(inp)
#
#     # Conv1D feature extraction
#     x = Conv1D(64, 3, activation='relu', padding='same',
#                kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.2)(x)
#
#     x = Conv1D(128, 3, activation='relu', padding='same',
#                kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = MaxPooling1D(2)(x)
#     x = Dropout(0.3)(x)
#
#     # Stacked Bidirectional LSTM
#     x = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3,
#                            recurrent_dropout=0.2, kernel_regularizer=l2(0.001)))(x)
#     x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3,
#                            recurrent_dropout=0.2, kernel_regularizer=l2(0.001)))(x)
#     x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3,
#                            recurrent_dropout=0.2, kernel_regularizer=l2(0.001)))(x)
#
#     # Dense layers
#     x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
#     x = BatchNormalization()(x)
#     x = Dropout(0.5)(x)
#
#     x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
#     x = Dropout(0.4)(x)
#
#     out = Dense(len(ACTIONS), activation='softmax')(x)
#
#     return Model(inputs=inp, outputs=out)
#
#
# # --- Visualization Functions ---
# def plot_training_history(history, save_path=None):
#     """Plot comprehensive training history."""
#     fig, axes = plt.subplots(2, 2, figsize=(15, 10))
#
#     # Accuracy plot
#     axes[0, 0].plot(history.history['categorical_accuracy'], label='Train Accuracy')
#     axes[0, 0].plot(history.history['val_categorical_accuracy'], label='Val Accuracy')
#     axes[0, 0].set_title('Model Accuracy')
#     axes[0, 0].set_xlabel('Epoch')
#     axes[0, 0].set_ylabel('Accuracy')
#     axes[0, 0].legend()
#     axes[0, 0].grid(True)
#
#     # Loss plot
#     axes[0, 1].plot(history.history['loss'], label='Train Loss')
#     axes[0, 1].plot(history.history['val_loss'], label='Val Loss')
#     axes[0, 1].set_title('Model Loss')
#     axes[0, 1].set_xlabel('Epoch')
#     axes[0, 1].set_ylabel('Loss')
#     axes[0, 1].legend()
#     axes[0, 1].grid(True)
#
#     # Error rate plot
#     train_error = [1 - acc for acc in history.history['categorical_accuracy']]
#     val_error = [1 - acc for acc in history.history['val_categorical_accuracy']]
#     axes[1, 0].plot(train_error, 'r-', label='Train Error')
#     axes[1, 0].plot(val_error, 'b-', label='Val Error')
#     axes[1, 0].set_title('Error Rate')
#     axes[1, 0].set_xlabel('Epoch')
#     axes[1, 0].set_ylabel('Error Rate')
#     axes[1, 0].legend()
#     axes[1, 0].grid(True)
#
#     # Learning rate (if available)
#     if 'lr' in history.history:
#         axes[1, 1].plot(history.history['lr'], label='Learning Rate')
#         axes[1, 1].set_title('Learning Rate Schedule')
#         axes[1, 1].set_xlabel('Epoch')
#         axes[1, 1].set_ylabel('Learning Rate')
#         axes[1, 1].set_yscale('log')
#         axes[1, 1].legend()
#         axes[1, 1].grid(True)
#     else:
#         axes[1, 1].axis('off')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Training history saved to {save_path}")
#
#     plt.show()
#
#
# def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
#     """Plot enhanced confusion matrix with percentages."""
#     cm = confusion_matrix(y_true, y_pred)
#     cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
#
#     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
#
#     # Raw counts
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=class_names, yticklabels=class_names, ax=ax1)
#     ax1.set_title('Confusion Matrix (Counts)')
#     ax1.set_xlabel('Predicted Label')
#     ax1.set_ylabel('True Label')
#
#     # Percentages
#     sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Greens',
#                 xticklabels=class_names, yticklabels=class_names, ax=ax2)
#     ax2.set_title('Confusion Matrix (%)')
#     ax2.set_xlabel('Predicted Label')
#     ax2.set_ylabel('True Label')
#
#     plt.tight_layout()
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         logger.info(f"Confusion matrix saved to {save_path}")
#
#     plt.show()
#
#     return cm
#
#
# def plot_roc_curves(y_true, y_pred_probs, class_names, save_path=None):
#     """Plot ROC curves for each class."""
#     n_classes = len(class_names)
#     y_true_bin = np.eye(n_classes)[y_true]
#
#     plt.figure(figsize=(10, 8))
#
#     for i, class_name in enumerate(class_names):
#         fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
#         auc = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
#         plt.plot(fpr, tpr, label=f'{class_name} (AUC = {auc:.3f})')
#
#     plt.plot([0, 1], [0, 1], 'k--', label='Random')
#     plt.xlabel('False Positive Rate')
#     plt.ylabel('True Positive Rate')
#     plt.title('ROC Curves - Binary Classification (Curl vs Press)')
#     plt.legend(loc='lower right')
#     plt.grid(True)
#
#     if save_path:
#         plt.savefig(save_path, dpi=300, bbox_inches='tight')
#         logger.info(f"ROC curves saved to {save_path}")
#
#     plt.show()
#
#
# # --- Metrics Calculation ---
# def calculate_detailed_metrics(y_true, y_pred, y_pred_probs, class_names):
#     """Calculate comprehensive performance metrics."""
#     overall_accuracy = accuracy_score(y_true, y_pred)
#     overall_error = 1 - overall_accuracy
#
#     # Per-class metrics
#     precision, recall, f1, support = precision_recall_fscore_support(
#         y_true, y_pred, average=None, labels=range(len(class_names))
#     )
#
#     # Macro and weighted averages
#     macro_precision = precision_score(y_true, y_pred, average='macro')
#     macro_recall = recall_score(y_true, y_pred, average='macro')
#     macro_f1 = f1_score(y_true, y_pred, average='macro')
#
#     weighted_precision = precision_score(y_true, y_pred, average='weighted')
#     weighted_recall = recall_score(y_true, y_pred, average='weighted')
#     weighted_f1 = f1_score(y_true, y_pred, average='weighted')
#
#     # Per-class AUC
#     y_true_bin = np.eye(len(class_names))[y_true]
#     class_auc = []
#     for i in range(len(class_names)):
#         try:
#             auc = roc_auc_score(y_true_bin[:, i], y_pred_probs[:, i])
#             class_auc.append(auc)
#         except:
#             class_auc.append(0.0)
#
#     results_df = pd.DataFrame({
#         'Class': class_names,
#         'Precision': precision,
#         'Recall': recall,
#         'F1-Score': f1,
#         'AUC': class_auc,
#         'Support': support
#     })
#
#     confidence_scores = np.max(y_pred_probs, axis=1)
#     avg_confidence = np.mean(confidence_scores)
#
#     return {
#         'overall_accuracy': overall_accuracy,
#         'overall_error': overall_error,
#         'per_class_metrics': results_df,
#         'macro_avg': {
#             'precision': macro_precision,
#             'recall': macro_recall,
#             'f1_score': macro_f1
#         },
#         'weighted_avg': {
#             'precision': weighted_precision,
#             'recall': weighted_recall,
#             'f1_score': weighted_f1
#         },
#         'avg_confidence': avg_confidence,
#         'confidence_scores': confidence_scores,
#         'confusion_matrix': confusion_matrix(y_true, y_pred)
#     }
#
#
# def print_detailed_results(metrics, class_names):
#     """Print comprehensive evaluation results."""
#     print("\n" + "=" * 80)
#     print("ðŸŽ¯ DETAILED PERFORMANCE METRICS - 2-CLASS EXERCISE DETECTION")
#     print("=" * 80)
#
#     print(f"\nðŸ“ˆ OVERALL ACCURACY: {metrics['overall_accuracy']:.4f} ({metrics['overall_accuracy'] * 100:.2f}%)")
#     print(f"âŒ OVERALL ERROR RATE: {metrics['overall_error']:.4f} ({metrics['overall_error'] * 100:.2f}%)")
#
#     print(f"\nðŸ“Š PER-CLASS PERFORMANCE:")
#     print("-" * 80)
#     print(f"{'Class':<20} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'AUC':<12} {'Support':<10}")
#     print("-" * 80)
#
#     for _, row in metrics['per_class_metrics'].iterrows():
#         print(f"{row['Class']:<20} {row['Precision']:<12.4f} {row['Recall']:<12.4f} "
#               f"{row['F1-Score']:<12.4f} {row['AUC']:<12.4f} {int(row['Support']):<10}")
#
#     print("-" * 80)
#     print(f"{'Macro Avg':<20} {metrics['macro_avg']['precision']:<12.4f} "
#           f"{metrics['macro_avg']['recall']:<12.4f} {metrics['macro_avg']['f1_score']:<12.4f}")
#     print(f"{'Weighted Avg':<20} {metrics['weighted_avg']['precision']:<12.4f} "
#           f"{metrics['weighted_avg']['recall']:<12.4f} {metrics['weighted_avg']['f1_score']:<12.4f}")
#
#     print(f"\nðŸŽ¯ CONFIDENCE ANALYSIS:")
#     print(f"Average Prediction Confidence: {metrics['avg_confidence']:.4f} ({metrics['avg_confidence'] * 100:.2f}%)")
#
#
# def save_metrics_to_file(metrics, save_dir: Path, timestamp: str):
#     """Save comprehensive metrics to file."""
#     metrics_path = save_dir / f"metrics_{timestamp}.txt"
#
#     with open(metrics_path, 'w') as f:
#         f.write("EXERCISE CLASSIFICATION - PERFORMANCE METRICS (2 Classes)\n")
#         f.write("Exercise Types: Bicep Curls & Shoulder Press\n")
#         f.write("=" * 80 + "\n\n")
#
#         f.write(f"Overall Accuracy: {metrics['overall_accuracy']:.4f}\n")
#         f.write(f"Overall Error Rate: {metrics['overall_error']:.4f}\n\n")
#
#         f.write("Per-Class Metrics:\n")
#         f.write(metrics['per_class_metrics'].to_string(index=False))
#
#         f.write(f"\n\nMacro Average:\n")
#         f.write(f"  Precision: {metrics['macro_avg']['precision']:.4f}\n")
#         f.write(f"  Recall: {metrics['macro_avg']['recall']:.4f}\n")
#         f.write(f"  F1-Score: {metrics['macro_avg']['f1_score']:.4f}\n")
#
#         f.write(f"\nWeighted Average:\n")
#         f.write(f"  Precision: {metrics['weighted_avg']['precision']:.4f}\n")
#         f.write(f"  Recall: {metrics['weighted_avg']['recall']:.4f}\n")
#         f.write(f"  F1-Score: {metrics['weighted_avg']['f1_score']:.4f}\n")
#
#         f.write(f"\nAverage Confidence: {metrics['avg_confidence']:.4f}\n")
#
#         f.write(f"\nConfusion Matrix:\n")
#         f.write(str(metrics['confusion_matrix']))
#
#     logger.info(f"ðŸ“„ Metrics saved to {metrics_path}")
#
#
# # --- Main Training Function ---
# def train_model(use_attention=True, use_cross_validation=False):
#     """
#     Main training function with options for attention and cross-validation.
#
#     Args:
#         use_attention: Whether to use attention mechanism
#         use_cross_validation: Whether to perform k-fold cross-validation
#     """
#     ensure_dirs(MODEL_DIR, LOG_DIR, RESULTS_DIR)
#
#     # Load data
#     X, y, metadata = load_data_with_metadata()
#
#     if X is None:
#         logger.error("âŒ No data found. Run data collection first.")
#         return None, None
#
#     # Check minimum samples
#     y_labels = np.argmax(y, axis=1)
#     class_counts = np.bincount(y_labels)
#
#     logger.info(f"Class distribution: {dict(zip(ACTIONS, class_counts))}")
#
#     min_samples_per_class = 20
#     if any(count < min_samples_per_class for count in class_counts):
#         logger.warning(
#             f"âš ï¸ Some classes have fewer than {min_samples_per_class} samples. "
#             "Consider collecting more data to prevent overfitting."
#         )
#
#     timestamp = str(int(time.time()))
#
#     if use_cross_validation:
#         logger.info(f"\n{'=' * 60}")
#         logger.info(f"Performing {K_FOLDS}-Fold Cross-Validation")
#         logger.info(f"{'=' * 60}")
#
#         return perform_cross_validation(X, y, use_attention, timestamp)
#     else:
#         return train_single_model(X, y, use_attention, timestamp)
#
#
# def train_single_model(X, y, use_attention, timestamp):
#     """Train a single model with train/val/test split."""
#     y_labels = np.argmax(y, axis=1)
#
#     # Stratified split
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42, stratify=y_labels
#     )
#
#     X_train, X_val, y_train, y_val = train_test_split(
#         X_train, y_train, test_size=0.15, random_state=42,
#         stratify=np.argmax(y_train, axis=1)
#     )
#
#     logger.info(f"\nðŸ“Š Data Split:")
#     logger.info(f"  Training samples: {len(X_train)}")
#     logger.info(f"  Validation samples: {len(X_val)}")
#     logger.info(f"  Test samples: {len(X_test)}")
#
#     # Compute class weights
#     y_train_labels = np.argmax(y_train, axis=1)
#     class_weights = compute_class_weight(
#         'balanced',
#         classes=np.unique(y_train_labels),
#         y=y_train_labels
#     )
#     class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
#
#     logger.info(f"  Class weights: {class_weight_dict}")
#
#     # Build model
#     if use_attention:
#         logger.info("\nðŸ”¨ Building LSTM model with Attention mechanism...")
#         model = build_lstm_with_attention()
#     else:
#         logger.info("\nðŸ”¨ Building Deep LSTM model...")
#         model = build_deep_lstm_model()
#
#     # Compile model
#     optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#     model.compile(
#         optimizer=optimizer,
#         loss='categorical_crossentropy',
#         metrics=['categorical_accuracy']
#     )
#
#     logger.info(f"\nðŸ“‹ Model Architecture:")
#     model.summary(print_fn=logger.info)
#
#     # Callbacks
#     name = f"Exercise2Class_{'Attention' if use_attention else 'Deep'}_{timestamp}"
#
#     callbacks = [
#         TensorBoard(log_dir=LOG_DIR / name, histogram_freq=1),
#         EarlyStopping(
#             patience=25,
#             restore_best_weights=True,
#             monitor='val_loss',
#             verbose=1
#         ),
#         ReduceLROnPlateau(
#             factor=0.5,
#             patience=10,
#             min_lr=1e-7,
#             monitor='val_loss',
#             verbose=1
#         ),
#         ModelCheckpoint(
#             MODEL_DIR / f"best_model_2class_{timestamp}.h5",
#             save_best_only=True,
#             monitor='val_loss',
#             verbose=1
#         )
#     ]
#
#     # Train
#     logger.info(f"\nðŸš€ Starting training...")
#     logger.info(f"  Epochs: {EPOCHS}")
#     logger.info(f"  Batch size: {BATCH_SIZE}")
#     logger.info(f"  Learning rate: {LEARNING_RATE}")
#
#     history = model.fit(
#         X_train, y_train,
#         epochs=EPOCHS,
#         batch_size=BATCH_SIZE,
#         validation_data=(X_val, y_val),
#         callbacks=callbacks,
#         class_weight=class_weight_dict,
#         verbose=1
#     )
#
#     # Plot training history
#     history_plot_path = RESULTS_DIR / f"training_history_{timestamp}.png"
#     plot_training_history(history, save_path=history_plot_path)
#
#     # Evaluate on test set
#     logger.info(f"\nðŸ“Š Evaluating on test set...")
#     y_pred_probs = model.predict(X_test, verbose=0)
#     y_pred = np.argmax(y_pred_probs, axis=1)
#     y_true = np.argmax(y_test, axis=1)
#
#     # Calculate metrics
#     metrics = calculate_detailed_metrics(y_true, y_pred, y_pred_probs, ACTIONS)
#
#     # Print results
#     print_detailed_results(metrics, ACTIONS)
#
#     # Classification report
#     print("\n" + "=" * 80)
#     print("ðŸ“‹ SKLEARN CLASSIFICATION REPORT:")
#     print("=" * 80)
#     print(classification_report(y_true, y_pred, target_names=ACTIONS, digits=4))
#
#     # Plot confusion matrix
#     cm_path = RESULTS_DIR / f"confusion_matrix_{timestamp}.png"
#     plot_confusion_matrix(y_true, y_pred, ACTIONS, save_path=cm_path)
#
#     # Plot ROC curves
#     roc_path = RESULTS_DIR / f"roc_curves_{timestamp}.png"
#     plot_roc_curves(y_true, y_pred_probs, ACTIONS, save_path=roc_path)
#
#     # Save metrics
#     save_metrics_to_file(metrics, RESULTS_DIR, timestamp)
#
#     # Save final model
#     model_path = MODEL_DIR / f"exercise_2class_final_{timestamp}.h5"
#     model.save(model_path)
#     logger.info(f"\nâœ… Model saved to {model_path}")
#
#     return model, metrics
#
#
# def perform_cross_validation(X, y, use_attention, timestamp):
#     """Perform stratified k-fold cross-validation."""
#     y_labels = np.argmax(y, axis=1)
#     skf = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=42)
#
#     fold_metrics = []
#     fold_histories = []
#
#     for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_labels), 1):
#         logger.info(f"\n{'=' * 60}")
#         logger.info(f"Training Fold {fold}/{K_FOLDS}")
#         logger.info(f"{'=' * 60}")
#
#         X_train, X_val = X[train_idx], X[val_idx]
#         y_train, y_val = y[train_idx], y[val_idx]
#
#         # Compute class weights for this fold
#         y_train_labels = np.argmax(y_train, axis=1)
#         class_weights = compute_class_weight(
#             'balanced',
#             classes=np.unique(y_train_labels),
#             y=y_train_labels
#         )
#         class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
#
#         # Build and compile model
#         if use_attention:
#             model = build_lstm_with_attention()
#         else:
#             model = build_deep_lstm_model()
#
#         optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
#         model.compile(
#             optimizer=optimizer,
#             loss='categorical_crossentropy',
#             metrics=['categorical_accuracy']
#         )
#
#         # Callbacks
#         callbacks = [
#             EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss', verbose=0),
#             ReduceLROnPlateau(factor=0.5, patience=8, min_lr=1e-7, monitor='val_loss', verbose=0)
#         ]
#
#         # Train
#         history = model.fit(
#             X_train, y_train,
#             epochs=EPOCHS,
#             batch_size=BATCH_SIZE,
#             validation_data=(X_val, y_val),
#             callbacks=callbacks,
#             class_weight=class_weight_dict,
#             verbose=0
#         )
#
#         fold_histories.append(history)
#
#         # Evaluate
#         y_pred_probs = model.predict(X_val, verbose=0)
#         y_pred = np.argmax(y_pred_probs, axis=1)
#         y_true = np.argmax(y_val, axis=1)
#
#         metrics = calculate_detailed_metrics(y_true, y_pred, y_pred_probs, ACTIONS)
#         fold_metrics.append(metrics)
#
#         logger.info(f"Fold {fold} - Accuracy: {metrics['overall_accuracy']:.4f}")
#
#     # Aggregate results
#     logger.info(f"\n{'=' * 60}")
#     logger.info(f"Cross-Validation Results Summary")
#     logger.info(f"{'=' * 60}")
#
#     accuracies = [m['overall_accuracy'] for m in fold_metrics]
#     f1_scores = [m['macro_avg']['f1_score'] for m in fold_metrics]
#
#     logger.info(f"Mean Accuracy: {np.mean(accuracies):.4f} Â± {np.std(accuracies):.4f}")
#     logger.info(f"Mean F1-Score: {np.mean(f1_scores):.4f} Â± {np.std(f1_scores):.4f}")
#
#     # Save CV results
#     cv_results_path = RESULTS_DIR / f"cv_results_{timestamp}.txt"
#     with open(cv_results_path, 'w') as f:
#         f.write(f"{K_FOLDS}-Fold Cross-Validation Results\n")
#         f.write("=" * 60 + "\n\n")
#         for i, metrics in enumerate(fold_metrics, 1):
#             f.write(f"Fold {i}:\n")
#             f.write(f"  Accuracy: {metrics['overall_accuracy']:.4f}\n")
#             f.write(f"  Macro F1: {metrics['macro_avg']['f1_score']:.4f}\n\n")
#         f.write(f"\nOverall:\n")
#         f.write(f"  Mean Accuracy: {np.mean(accuracies):.4f} Â± {np.std(accuracies):.4f}\n")
#         f.write(f"  Mean F1-Score: {np.mean(f1_scores):.4f} Â± {np.std(f1_scores):.4f}\n")
#
#     logger.info(f"CV results saved to {cv_results_path}")
#
#     return fold_metrics, fold_histories
#
#
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(description="Train LSTM model for 2-class exercise detection")
#     parser.add_argument(
#         '--no-attention',
#         action='store_true',
#         help='Use deep LSTM without attention mechanism'
#     )
#     parser.add_argument(
#         '--cross-validation',
#         action='store_true',
#         help='Perform k-fold cross-validation'
#     )
#
#     args = parser.parse_args()
#
#     use_attention = not args.no_attention
#     use_cv = args.cross_validation
#
#     logger.info(f"\n{'=' * 60}")
#     logger.info("EXERCISE DETECTION - LSTM Training (2 Classes)")
#     logger.info("Exercises: Bicep Curls & Shoulder Press")
#     logger.info(f"{'=' * 60}")
#
#     trained_model, performance_metrics = train_model(
#         use_attention=use_attention,
#         use_cross_validation=use_cv)


import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight

ACTIONS = np.array(['curl_correct', 'curl_incorrect', 'press_correct', 'press_incorrect'])
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
NUM_DIMS = 4
INPUT_SIZE = NUM_LANDMARKS * NUM_DIMS
DATA_DIR = Path.cwd() / "data2"
MODEL_DIR = Path.cwd() / "models"
LOG_DIR = Path.cwd() / "logs"
PLOTS_DIR = Path.cwd() / "plots"  # New directory for saving plots


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def plot_training_history(history, save_path=None):
    """
    Plot training and validation accuracy and loss curves
    Suitable for research papers
    """
    # Set style for publication-quality plots
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Extract metrics
    epochs = range(1, len(history.history['loss']) + 1)
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    train_acc = history.history['categorical_accuracy']
    val_acc = history.history['val_categorical_accuracy']

    # Plot 1: Training and Validation Loss
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)

    # Mark the best epoch
    best_epoch = np.argmin(val_loss) + 1
    best_val_loss = np.min(val_loss)
    axes[0].scatter(best_epoch, best_val_loss, s=150, c='gold',
                    marker='*', edgecolors='black', linewidths=1.5,
                    label=f'Best Epoch ({best_epoch})', zorder=5)

    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Loss', fontsize=12, fontweight='bold')
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim([1, len(epochs)])

    # Plot 2: Training and Validation Accuracy
    axes[1].plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)

    # Mark the best epoch for accuracy
    best_epoch_acc = np.argmax(val_acc) + 1
    best_val_acc = np.max(val_acc)
    axes[1].scatter(best_epoch_acc, best_val_acc, s=150, c='gold',
                    marker='*', edgecolors='black', linewidths=1.5,
                    label=f'Best Epoch ({best_epoch_acc})', zorder=5)

    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(loc='lower right', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xlim([1, len(epochs)])
    axes[1].set_ylim([0, 1])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training history plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """
    Plot confusion matrix heatmap
    Suitable for research papers
    """
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    # Create figure
    plt.figure(figsize=(10, 8))

    # Create annotations with both count and percentage
    annotations = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annotations[i, j] = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'

    # Plot heatmap
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, linewidths=0.5, linecolor='gray')

    plt.xlabel('Predicted Label', fontsize=12, fontweight='bold')
    plt.ylabel('True Label', fontsize=12, fontweight='bold')
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Confusion matrix saved to {save_path}")

    plt.show()
    plt.close()


def plot_per_class_metrics(y_true, y_pred, class_names, save_path=None):
    """
    Plot precision, recall, and F1-score for each class
    Suitable for research papers
    """
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred)

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))

    bars1 = ax.bar(x - width, precision, width, label='Precision', color='#2E86AB', alpha=0.8)
    bars2 = ax.bar(x, recall, width, label='Recall', color='#A23B72', alpha=0.8)
    bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#F18F01', alpha=0.8)

    # Add value labels on bars
    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.3f}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    add_value_labels(bars1)
    add_value_labels(bars2)
    add_value_labels(bars3)

    ax.set_xlabel('Exercise Classes', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Per-Class Performance Metrics', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend(fontsize=10, loc='lower right')
    ax.set_ylim([0, 1.1])
    ax.grid(True, alpha=0.3, linestyle='--', axis='y')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Per-class metrics plot saved to {save_path}")

    plt.show()
    plt.close()


def plot_metrics_summary(history, y_true, y_pred, class_names, save_path=None):
    """
    Create a comprehensive summary plot combining multiple metrics
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Plot 1: Loss curve
    ax1 = fig.add_subplot(gs[0, :2])
    epochs = range(1, len(history.history['loss']) + 1)
    ax1.plot(epochs, history.history['loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history.history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontweight='bold')
    ax1.set_ylabel('Loss', fontweight='bold')
    ax1.set_title('Loss Curves', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy curve
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.plot(epochs, history.history['categorical_accuracy'], 'b-',
             label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history.history['val_categorical_accuracy'], 'r-',
             label='Validation Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontweight='bold')
    ax2.set_ylabel('Accuracy', fontweight='bold')
    ax2.set_title('Accuracy Curves', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, 1])

    # Plot 3: Final metrics table
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    final_train_acc = history.history['categorical_accuracy'][-1]
    final_val_acc = history.history['val_categorical_accuracy'][-1]
    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    test_acc = accuracy_score(y_true, y_pred)

    metrics_text = f"""
    Final Metrics:

    Training Acc: {final_train_acc:.4f}
    Validation Acc: {final_val_acc:.4f}
    Test Acc: {test_acc:.4f}

    Training Loss: {final_train_loss:.4f}
    Validation Loss: {final_val_loss:.4f}

    Total Epochs: {len(epochs)}
    """
    ax3.text(0.1, 0.5, metrics_text, fontsize=11, verticalalignment='center',
             fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Plot 4: Per-class metrics
    ax4 = fig.add_subplot(gs[1, 2])
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    metrics = ['Precision', 'Recall', 'F1-Score']
    values = [precision, recall, f1]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    bars = ax4.barh(metrics, values, color=colors, alpha=0.7)
    ax4.set_xlim([0, 1])
    ax4.set_xlabel('Score', fontweight='bold')
    ax4.set_title('Weighted Average Metrics', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='x')
    for i, (bar, value) in enumerate(zip(bars, values)):
        ax4.text(value + 0.02, i, f'{value:.4f}', va='center', fontweight='bold')

    # Plot 5: Confusion Matrix
    ax5 = fig.add_subplot(gs[2, :])
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax5,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    ax5.set_xlabel('Predicted Label', fontweight='bold')
    ax5.set_ylabel('True Label', fontweight='bold')
    ax5.set_title('Confusion Matrix', fontweight='bold')
    plt.setp(ax5.get_xticklabels(), rotation=45, ha='right')

    plt.suptitle('Model Performance Summary', fontsize=16, fontweight='bold', y=0.995)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Metrics summary saved to {save_path}")

    plt.show()
    plt.close()


def plot_learning_rate_schedule(history, save_path=None):
    """
    Plot learning rate changes during training (if using ReduceLROnPlateau)
    """
    if 'lr' in history.history:
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(history.history['lr']) + 1)
        plt.plot(epochs, history.history['lr'], 'b-', linewidth=2)
        plt.xlabel('Epoch', fontsize=12, fontweight='bold')
        plt.ylabel('Learning Rate', fontsize=12, fontweight='bold')
        plt.title('Learning Rate Schedule', fontsize=14, fontweight='bold')
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Learning rate schedule saved to {save_path}")

        plt.show()
        plt.close()


def load_data():
    """Enhanced data loading with validation"""
    sequences, labels = [], []
    label_map = {label: idx for idx, label in enumerate(ACTIONS)}

    for action in ACTIONS:
        action_path = DATA_DIR / action
        if not action_path.exists():
            print(f"Warning: No data found for action {action}")
            continue

        action_sequences = 0
        for seq_dir in action_path.glob("*"):
            if seq_dir.is_dir():
                npy_files = list(seq_dir.glob("*.npy"))
                if len(npy_files) == SEQUENCE_LENGTH:
                    window = []
                    valid = True

                    for f in range(SEQUENCE_LENGTH):
                        frame_path = seq_dir / f"{f}.npy"
                        if frame_path.exists():
                            data = np.load(frame_path)
                            if np.sum(data) > 0:  # Valid pose data
                                window.append(data)
                            else:
                                valid = False
                                break
                        else:
                            valid = False
                            break

                    if valid and len(window) == SEQUENCE_LENGTH:
                        sequences.append(window)
                        labels.append(label_map[action])
                        action_sequences += 1

        print(f"Loaded {action_sequences} sequences for {action}")

    if not sequences:
        print("❌ No valid sequences found!")
        return None, None

    X = np.array(sequences)
    y = np.array(labels)

    print(f"✅ Total sequences: {len(sequences)}")
    print(f"Data shape: {X.shape}")
    print(f"Label distribution: {np.bincount(y)}")

    unique_classes = np.unique(y)
    print(f"Classes present in data: {unique_classes}")
    print(f"Expected classes: {np.arange(len(ACTIONS))}")

    if len(unique_classes) != len(ACTIONS):
        print(f"⚠️ WARNING: Only {len(unique_classes)} out of {len(ACTIONS)} classes found!")
        print(f"Missing classes: {set(range(len(ACTIONS))) - set(unique_classes)}")
        for i, action in enumerate(ACTIONS):
            if i not in unique_classes:
                print(f"  - {action} (class {i}) has NO samples!")

    return X, y


def build_robust_model(num_classes):
    """Build a more robust model with regularization"""
    inp = Input(shape=(SEQUENCE_LENGTH, INPUT_SIZE))

    x = Conv1D(32, 3, activation='relu', padding='same',
               kernel_regularizer=l2(0.01))(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Conv1D(64, 3, activation='relu', padding='same',
               kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = MaxPooling1D(2)(x)
    x = Dropout(0.3)(x)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3,
                           recurrent_dropout=0.3, kernel_regularizer=l2(0.01)))(x)
    x = Bidirectional(LSTM(64, return_sequences=False, dropout=0.3,
                           recurrent_dropout=0.3, kernel_regularizer=l2(0.01)))(x)

    x = Dense(256, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)

    x = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(x)
    x = Dropout(0.4)(x)

    out = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inp, outputs=out)


def train():
    ensure_dirs(MODEL_DIR, LOG_DIR, PLOTS_DIR)

    X, y = load_data()
    if X is None:
        print("❌ No data found. Run data collection first.")
        return

    min_samples_per_class = 10
    class_counts = np.bincount(y)

    print(f"\n📊 Class distribution:")
    for i, count in enumerate(class_counts):
        print(f"  {ACTIONS[i]}: {count} samples")

    if any(count < min_samples_per_class for count in class_counts):
        print(f"\n⚠️ Warning: Some classes have fewer than {min_samples_per_class} samples")
        print("This may lead to overfitting. Collect more data.")

    if len(class_counts) != len(ACTIONS):
        print(f"\n❌ ERROR: Expected {len(ACTIONS)} classes but found {len(class_counts)} classes")
        print("Please collect data for all action classes before training.")
        return

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42,
            stratify=y_train
        )
    except ValueError as e:
        print(f"\n❌ Stratification failed: {e}")
        print("Falling back to non-stratified split...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

    num_classes = len(ACTIONS)
    y_train_cat = to_categorical(y_train, num_classes=num_classes)
    y_val_cat = to_categorical(y_val, num_classes=num_classes)
    y_test_cat = to_categorical(y_test, num_classes=num_classes)

    print(f"\nTraining samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Output shape: {y_train_cat.shape}")

    class_weights = compute_class_weight('balanced',
                                         classes=np.unique(y_train),
                                         y=y_train)
    class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

    model = build_robust_model(num_classes)
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['categorical_accuracy']
    )

    print(model.summary())

    name = f"ExerciseForm-{int(time.time())}"
    tb = TensorBoard(log_dir=LOG_DIR / name)
    es = EarlyStopping(patience=15, restore_best_weights=True, monitor='val_loss')
    lr = ReduceLROnPlateau(factor=0.7, patience=8, min_lr=1e-7, monitor='val_loss')
    ckpt = ModelCheckpoint(MODEL_DIR / "best_model.h5", save_best_only=True,
                           monitor='val_loss', verbose=1)

    print("\n🚀 Starting training...")
    history = model.fit(
        X_train, y_train_cat,
        epochs=40,
        batch_size=8,
        validation_data=(X_val, y_val_cat),
        callbacks=[tb, es, lr, ckpt],
        class_weight=class_weight_dict,
        verbose=1
    )

    print("\n📊 Generating visualizations...")

    # Generate predictions for plotting
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = y_test

    # Create all plots
    timestamp = int(time.time())

    # 1. Training history plot
    plot_training_history(history,
                          save_path=PLOTS_DIR / f'training_history_{timestamp}.png')

    # 2. Confusion matrix
    plot_confusion_matrix(y_true, y_pred, ACTIONS,
                          save_path=PLOTS_DIR / f'confusion_matrix_{timestamp}.png')

    # 3. Per-class metrics
    plot_per_class_metrics(y_true, y_pred, ACTIONS,
                           save_path=PLOTS_DIR / f'per_class_metrics_{timestamp}.png')

    # 4. Comprehensive summary
    plot_metrics_summary(history, y_true, y_pred, ACTIONS,
                         save_path=PLOTS_DIR / f'metrics_summary_{timestamp}.png')

    # 5. Learning rate schedule (if available)
    plot_learning_rate_schedule(history,
                                save_path=PLOTS_DIR / f'lr_schedule_{timestamp}.png')

    # Print detailed classification report
    print("\n📊 Test Set Results:")
    print(classification_report(y_true, y_pred, target_names=ACTIONS, digits=4))

    # Print additional metrics
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"\n📈 Weighted Average Metrics:")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Accuracy: {accuracy_score(y_true, y_pred):.4f}")

    # Save final model
    model.save(MODEL_DIR / "exercise_form_model.h5")
    print(f"\n✅ Model saved to {MODEL_DIR / 'exercise_form_model.h5'}")
    print(f"✅ All plots saved to {PLOTS_DIR}")


if __name__ == "__main__":
    train()
