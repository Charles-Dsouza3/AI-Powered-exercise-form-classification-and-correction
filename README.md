# AI-Powered Exercise Form Classification and Correction

An intelligent real-time exercise form analysis system that uses deep learning and computer vision to classify exercises and provide instant feedback on form quality. This system leverages **Attention-Augmented Bidirectional LSTM networks** and **MediaPipe Pose estimation** to detect, classify, and correct exercise form during workouts.

---

## 🎯 Overview

This project implements an end-to-end machine learning pipeline for exercise form analysis, featuring:

- **Real-time pose detection** using MediaPipe
- **Attention-based BiLSTM model** for temporal sequence classification
- **Live form correction feedback** with biomechanical analysis
- **Professional GUI application** built with PySide6
- **Comprehensive data collection and training pipeline**

The system currently supports **4 exercise classes**:
- ✅ Bicep Curl (Correct Form)
- ❌ Bicep Curl (Incorrect Form)
- ✅ Shoulder Press (Correct Form)
- ❌ Shoulder Press (Incorrect Form)

---

## ✨ Key Features

### 🤖 Advanced Deep Learning Architecture
- **Bidirectional LSTM layers** with attention mechanism for temporal pattern recognition
- **Conv1D layers** for feature extraction from pose sequences
- **Batch normalization and dropout** for regularization
- **Self-attention mechanism** to focus on critical frames in exercise sequences
- Trained on sequences of 30 frames capturing full exercise movements

### 🎥 Real-Time Analysis
- Live webcam feed processing at 30 FPS
- Instant exercise classification with confidence thresholds
- Real-time rep counting with form quality tracking
- Visual feedback with pose landmarks overlay
- Exercise-specific biomechanical validation

### 📊 Intelligent Form Correction
- **Joint angle analysis** (elbow, shoulder, hip, knee)
- **Alignment checks** (spine reference, elbow stability)
- **Movement symmetry detection**
- **Range of motion validation**
- Actionable feedback messages for form improvement

### 🖥️ Professional User Interface
- Modern, intuitive GUI with three main screens:
  - **Welcome Page**: Session overview and instructions
  - **Workout Page**: Live video feed with real-time statistics
  - **Summary Page**: Post-workout performance analysis
- Visual progress indicators and accuracy metrics
- Session statistics with rep counting and form accuracy

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   INPUT: Webcam/Video Feed                  │
└─────────────────────────────────┬───────────────────────────┘
                                  │
                    ┌─────────────▼─────────────┐
                    │  MediaPipe Pose Detection │
                    │  (33 landmarks × 4 dims)  │
                    └─────────────┬─────────────┘
                                  │
              ┌───────────────────┴───────────────────┐
              │                                       │
    ┌─────────▼──────────┐              ┌───────────▼────────────┐
    │ Keypoint Extraction│              │ Biomechanical Features │
    │   (132 features)   │              │   (8 joint angles)     │
    └─────────┬──────────┘              └───────────┬────────────┘
              │                                     │
              └──────────────┬──────────────────────┘
                             │
                ┌────────────▼────────────┐
                │  Sequence Buffer (30)   │
                │   Shape: (30, 140)      │
                └────────────┬────────────┘
                             │
                ┌────────────▼─────────────┐
                │  BiLSTM + Attention Net  │
                │    (Classification)      │
                └────────────┬─────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼──────────┐      ┌─────────▼──────────┐
    │  Exercise Class    │      │   Form Analysis    │
    │   (4 classes)      │      │  (Feedback Rules)  │
    └─────────┬──────────┘      └─────────┬──────────┘
              │                           │
              └──────────┬────────────────┘
                         │
              ┌──────────▼───────────┐
              │   Real-time Display  │
              │  + Rep Counter +     │
              │   Form Feedback      │
              └──────────────────────┘
```

---

## 📁 Project Structure

```
AI-Powered-Exercise-Form-Classification/
├── data2/                      # Training data directory
│   ├── curl_correct/          # Correct bicep curl sequences
│   ├── curl_incorrect/        # Incorrect bicep curl sequences
│   ├── press_correct/         # Correct shoulder press sequences
│   └── press_incorrect/       # Incorrect shoulder press sequences
│
├── src/                      # Training data directory
│   ├── app.py                      # Main GUI applications
│   ├── data_collection.py          # Data collection utilities
│   ├── train_model.py              # Model training pipeline
│
├── videos/
│
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation


```

---

## 🛠️ Technical Stack

### Core Technologies
- **Python 3.8+**
- **TensorFlow/Keras 2.x** - Deep learning framework
- **MediaPipe** - Real-time pose estimation
- **OpenCV** - Computer vision and video processing
- **NumPy** - Numerical computing
- **PySide6** - Modern GUI framework

### Model Architecture Details
```python
Input: (30, 140)  # 30 frames, 132 pose features + 8 biomechanical features

LayerNormalization
Conv1D(64, 5) + ReLU + BatchNorm + Dropout(0.2)
Conv1D(128, 3) + ReLU + BatchNorm + MaxPooling1D(2) + Dropout(0.25)

Bidirectional LSTM(256, return_sequences=True) + BatchNorm
Bidirectional LSTM(128, return_sequences=True) + BatchNorm

Self-Attention Layer (custom)

Dense(512) + ReLU + BatchNorm + Dropout(0.5)
Dense(256) + ReLU + BatchNorm + Dropout(0.4)
Dense(128) + ReLU + Dropout(0.3)

Output: Dense(4, softmax)  # 4 exercise classes
```

**Model Parameters**: ~2.5M trainable parameters

---

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8 or higher
- Webcam (for live inference)
- CUDA-compatible GPU (optional, for faster training)

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/AI-Powered-Exercise-Form-Classification.git
cd AI-Powered-Exercise-Form-Classification
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

**Requirements:**
```
tensorflow>=2.10.0
opencv-python>=4.7.0
mediapipe>=0.10.0
PySide6>=6.4.0
numpy>=1.23.0
scikit-learn>=1.2.0
matplotlib>=3.6.0
seaborn>=0.12.0
```

### Step 3: Directory Setup
```bash
# Create necessary directories
mkdir -p data2/{curl_correct,curl_incorrect,press_correct,press_incorrect}
mkdir -p models logs results videos
```

---

## 📊 Usage

### 1️⃣ Data Collection

#### Option A: Webcam Data Collection
```bash
python data_collection.py
```

**Instructions:**
- Select exercise type (0-3)
- Press **'S'** to start recording
- Perform the exercise (3-5 seconds)
- Press **'E'** to stop and save
- Press **'Q'** to quit

**Tips for Quality Data:**
- Ensure full body is visible in frame
- Maintain consistent lighting
- Record from frontal view
- Perform smooth, complete repetitions

#### Option B: Video File Processing
Place video files in the `videos/` directory with naming convention:
- `curl_correct_01.mp4`
- `curl_incorrect_01.mp4`
- `press_correct_01.mp4`
- `press_incorrect_01.mp4`

Run the collection script to process videos automatically.

### 2️⃣ Model Training

```bash
python train_model.py
```

**Training Configuration:**
- **Epochs**: 100 (with early stopping)
- **Batch Size**: 16
- **Learning Rate**: 0.001 (with ReduceLROnPlateau)
- **Train/Val/Test Split**: 68/12/20
- **Sequence Length**: 30 frames

**Training Features:**
- Stratified data splitting
- Class weight balancing
- Early stopping (patience: 25 epochs)
- Learning rate reduction (patience: 10 epochs)
- TensorBoard logging
- Model checkpointing

**Outputs:**
- Trained model: `models/exercise_form_model.h5`
- Training plots: `results/training_history.png`
- Confusion matrix: `results/confusion_matrix.png`
- Performance metrics: `results/metrics.txt`

### 3️⃣ Live Application

```bash
python app.py
```

**Application Workflow:**
1. **Welcome Screen**: Click "START WORKOUT"
2. **Workout Screen**: 
   - Live video feed with pose overlay
   - Real-time exercise detection
   - Rep counting and form feedback
   - Accuracy tracking per exercise
3. **Summary Screen**: 
   - Total reps per exercise
   - Form accuracy percentage
   - Common form issues identified

**Keyboard Shortcuts:**
- **ESC**: Exit application
- **End Session Button**: View workout summary

---

## 🧠 Model Performance

### Training Metrics (Example Results)
```
Overall Accuracy: 94.3%
Macro F1-Score: 0.932

Per-Class Performance:
┌──────────────────┬───────────┬────────┬──────────┐
│ Class            │ Precision │ Recall │ F1-Score │
├──────────────────┼───────────┼────────┼──────────┤
│ curl_correct     │   0.96    │  0.94  │   0.95   │
│ curl_incorrect   │   0.93    │  0.95  │   0.94   │
│ press_correct    │   0.95    │  0.93  │   0.94   │
│ press_incorrect  │   0.91    │  0.94  │   0.92   │
└──────────────────┴───────────┴────────┴──────────┘
```

### Model Strengths
- High accuracy on both correct and incorrect form detection
- Robust temporal pattern recognition
- Real-time inference capability (<50ms per frame)
- Generalization to different body types and camera angles

---

## 🔧 Key Components Explained

### 1. Pose Estimation (`app.py`)
```python
# MediaPipe configuration
mp_pose.Pose(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    model_complexity=1,
    smooth_landmarks=True
)
```

**Extracted Features (per frame):**
- 33 landmarks × (x, y, z, visibility) = 132 features
- 8 biomechanical joint angles
- **Total**: 140 features per frame

### 2. Form Analysis Functions

#### Bicep Curl Analysis
```python
def analyze_bicep_curl_form(landmarks):
    """
    Checks:
    - Elbow stability (alignment with spine)
    - Elbow drift from sides
    - Full range of motion
    - Symmetry between arms
    """
```

#### Shoulder Press Analysis
```python
def analyze_shoulder_press_form(landmarks):
    """
    Checks:
    - Elbow position relative to shoulders
    - Shoulder alignment
    - Vertical pressing path
    """
```

### 3. Rep Counting Logic
- **Bicep Curl**: Detects elbow angle transitions (<30° to >150°)
- **Shoulder Press**: Detects elbow angle transitions (>140° to <100°)
- Stage-based counting prevents double-counting
- Form validation on every completed rep

### 4. Attention Mechanism
```python
class AttentionLayer(Layer):
    """
    Self-attention to focus on important frames in sequences
    Learns which frames are most discriminative for classification
    """
```

---

## 📈 Data Augmentation

To improve model robustness, the following augmentations are applied:

1. **Horizontal Flipping**: Mirror exercises (for bilateral exercises)
2. **Rotation**: ±25° pose rotation
3. **Scaling**: 85%-115% size variation
4. **Translation**: ±8% positional shift
5. **Gaussian Noise**: 0.5%-1.5% landmark noise

Each original sequence generates **5 augmented variants**, expanding the dataset 6x.

---

## 🎓 Biomechanical Features

The system extracts 8 joint angles per frame:

| Joint | Landmarks Used | Purpose |
|-------|---------------|---------|
| Right Elbow | Shoulder-Elbow-Wrist | Curl/press detection |
| Left Elbow | Shoulder-Elbow-Wrist | Curl/press detection |
| Right Shoulder | Hip-Shoulder-Elbow | Upper body posture |
| Left Shoulder | Hip-Shoulder-Elbow | Upper body posture |
| Right Hip | Shoulder-Hip-Knee | Core stability |
| Left Hip | Shoulder-Hip-Knee | Core stability |
| Right Knee | Hip-Knee-Ankle | Lower body check |
| Left Knee | Hip-Knee-Ankle | Lower body check |

These features enhance temporal modeling and improve classification accuracy.

---

## 🐛 Troubleshooting

### Common Issues

**Issue 1: Model file not found**
```
Error: ❌ Model not found at models/exercise_form_model.h5
Solution: Train the model first using train_model.py
```

**Issue 2: Webcam not detected**
```
Error: Could not open webcam
Solutions:
- Check webcam permissions
- Try different camera index: cv2.VideoCapture(1)
- Verify camera is not in use by another application
```

**Issue 3: Low detection confidence**
```
Issue: Pose landmarks not detected consistently
Solutions:
- Ensure adequate lighting
- Move to uncluttered background
- Position full body in frame
- Adjust min_detection_confidence in code
```

**Issue 4: Poor classification accuracy**
```
Issue: Incorrect exercise predictions
Solutions:
- Collect more training data (aim for 100+ sequences per class)
- Ensure clear form differences between correct/incorrect classes
- Retrain model with balanced dataset
```

---

## 🔮 Future Enhancements

- [ ] Add more exercise types (squats, lunges, deadlifts)
- [ ] Multi-person pose tracking
- [ ] Mobile app development (Android/iOS)
- [ ] Cloud-based model deployment
- [ ] Workout plan integration
- [ ] Progress tracking over time
- [ ] Voice feedback system
- [ ] Integration with fitness wearables

---

## 📝 Citation

If you use this project in your research or applications, please cite:

```bibtex
@software{exercise_form_classifier,
  author = {Charles Dsouza, Malhar Rane},
  title = {AI-Powered Exercise Form Classification and Correction System},
  year = {2025},
  url = {https://github.com/yourusername/AI-Powered-Exercise-Form-Classification}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 👤 Author

**Charles Dsouza**
- GitHub: [@Charles-Dsouza3]([https://github.com/yourusername](https://github.com/Charles-Dsouza3))
- Email: dsouzacharles26@gmail.com
- LinkedIn: [Charles Dsouza](https://www.linkedin.com/in/charles-dsouza26/)

**Malhar Rane**
- GitHub: [@MalharRane]([https://github.com/yourusername](https://github.com/MalharRane))
- Email: malharrane@gmail.com
- LinkedIn: [Malhar Rane](https://www.linkedin.com/in/malharane/)

---

## 🙏 Acknowledgments

- **MediaPipe** team for the excellent pose estimation solution
- **TensorFlow/Keras** community for deep learning tools
- **PySide6** for the modern GUI framework
- Exercise science community for biomechanical insights

---

## 📊 Project Statistics

- **Lines of Code**: ~3,500
- **Training Time**: ~2-4 hours (depends on dataset size and hardware)
- **Inference Speed**: ~20-30 FPS (GPU) / ~10-15 FPS (CPU)
- **Model Size**: ~35 MB
- **Dataset Size**: Configurable (recommended: 100+ sequences per class)

---

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Troubleshooting](#-troubleshooting) section
2. Search existing [GitHub Issues](https://github.com/yourusername/AI-Powered-Exercise-Form-Classification/issues)
3. Open a new issue with detailed description and error logs

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ and 💪 for the fitness tech community

</div>
