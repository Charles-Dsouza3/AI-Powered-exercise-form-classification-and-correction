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
DATA_DIR = Path.cwd() / "data"
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
