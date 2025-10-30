import sys
import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
import time

# --- PySide6 Imports ---
from PySide6.QtCore import Qt, QThread, Signal, Slot, QSize, QTimer
from PySide6.QtGui import QImage, QPixmap, QFont, QPalette, QColor
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QStackedWidget,
    QSizePolicy,
    QGridLayout,
    QFrame,
    QProgressBar,
)

# --- Original ML/CV Constants & Helpers ---
ACTIONS = np.array(['curl_correct', 'curl_incorrect', 'press_correct', 'press_incorrect'])
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
NUM_DIMS = 4
INPUT_SIZE = NUM_LANDMARKS * NUM_DIMS
MODEL_PATH = Path.cwd() / "models" / "exercise_form_model.h5"

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


# --- Helper Functions ---
def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR), results


def draw_landmarks_simplified(image, results):
    if not results.pose_landmarks:
        return
    mp_drawing.draw_landmarks(
        image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2, circle_radius=2)
    )


def extract_keypoints(results):
    pose = np.zeros(INPUT_SIZE)
    if results and results.pose_landmarks:
        pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility]
                         for lmk in results.pose_landmarks.landmark]).flatten()
    return pose


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180 else angle


def get_coord(landmarks, side, joint):
    idx = getattr(mp_pose.PoseLandmark, f"{side.upper()}_{joint.upper()}").value
    return [landmarks[idx].x, landmarks[idx].y]


def get_spine_reference(landmarks):
    try:
        left_shoulder = get_coord(landmarks, 'left', 'shoulder')
        right_shoulder = get_coord(landmarks, 'right', 'shoulder')
        left_hip = get_coord(landmarks, 'left', 'hip')
        right_hip = get_coord(landmarks, 'right', 'hip')
        shoulder_mid_x = (left_shoulder[0] + right_shoulder[0]) / 2
        hip_mid_x = (left_hip[0] + right_hip[0]) / 2
        spine_x = (shoulder_mid_x + hip_mid_x) / 2
        spine_y = (left_shoulder[1] + right_shoulder[1]) / 2
        return [spine_x, spine_y]
    except Exception:
        return None


def check_elbow_alignment(landmarks, tolerance=0.12):
    try:
        spine_ref = get_spine_reference(landmarks)
        if spine_ref is None: return False, "Cannot detect spine"
        left_elbow = get_coord(landmarks, 'left', 'elbow')
        right_elbow = get_coord(landmarks, 'right', 'elbow')
        left_drift = abs(left_elbow[0] - spine_ref[0])
        right_drift = abs(right_elbow[0] - spine_ref[0])
        left_aligned = left_drift <= tolerance
        right_aligned = right_drift <= tolerance

        if not left_aligned and not right_aligned:
            return False, "Keep both elbows stable"
        elif not left_aligned:
            return False, "Keep your left elbow stable"
        elif not right_aligned:
            return False, "Keep your right elbow stable"
        else:
            return True, "Good alignment"
    except Exception as e:
        return False, f"Error: {str(e)}"


def analyze_bicep_curl_form(landmarks):
    try:
        left_shoulder = get_coord(landmarks, 'left', 'shoulder')
        left_elbow = get_coord(landmarks, 'left', 'elbow')
        left_wrist = get_coord(landmarks, 'left', 'wrist')
        right_shoulder = get_coord(landmarks, 'right', 'shoulder')
        right_elbow = get_coord(landmarks, 'right', 'elbow')
        right_wrist = get_coord(landmarks, 'right', 'wrist')

        feedback = []
        form_score = 100

        is_aligned, alignment_msg = check_elbow_alignment(landmarks, tolerance=0.12)
        if not is_aligned:
            feedback.append(alignment_msg)
            form_score -= 30

        left_elbow_drift = abs(left_elbow[0] - left_shoulder[0])
        right_elbow_drift = abs(right_elbow[0] - right_shoulder[0])
        if left_elbow_drift > 0.15 or right_elbow_drift > 0.15:
            feedback.append("Keep elbows tucked at your sides")
            form_score -= 20

        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_angle > 170 or right_angle > 170:
            feedback.append("Avoid locking arms at the bottom")
            form_score -= 15

        if left_angle < 35 or right_angle < 35:
            feedback.append("Bring weights higher for a full curl")
            form_score -= 10

        angle_diff = abs(left_angle - right_angle)
        if angle_diff > 30:
            feedback.append("Move both arms at the same speed")
            form_score -= 15

        return feedback, form_score > 60
    except Exception as e:
        return ["Form check unavailable"], False


def analyze_shoulder_press_form(landmarks):
    try:
        left_shoulder = get_coord(landmarks, 'left', 'shoulder')
        left_elbow = get_coord(landmarks, 'left', 'elbow')
        left_wrist = get_coord(landmarks, 'left', 'wrist')
        right_shoulder = get_coord(landmarks, 'right', 'shoulder')
        right_elbow = get_coord(landmarks, 'right', 'elbow')
        right_wrist = get_coord(landmarks, 'right', 'wrist')

        feedback = []
        form_score = 100

        if left_elbow[1] > left_shoulder[1] + 0.1 or right_elbow[1] > right_shoulder[1] + 0.1:
            feedback.append("Lower elbows to shoulder height")
            form_score -= 25

        shoulder_alignment = abs(left_shoulder[0] - right_shoulder[0])
        if shoulder_alignment > 0.2:
            feedback.append("Keep your shoulders level")
            form_score -= 20

        left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)

        if left_wrist[1] > left_elbow[1] or right_wrist[1] > right_elbow[1]:
            if left_angle < 90 or right_angle < 90:
                feedback.append("Press weights straight up, not forward")
                form_score -= 25

        return feedback, form_score > 65
    except Exception as e:
        return ["Form check unavailable"], False


def count_reps(pred_class, landmarks, counters, stages):
    exercise = pred_class.split('_')[0] if pred_class else ""

    if exercise == 'curl':
        try:
            left_shoulder = get_coord(landmarks, 'left', 'shoulder')
            left_elbow = get_coord(landmarks, 'left', 'elbow')
            left_wrist = get_coord(landmarks, 'left', 'wrist')
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            if angle < 30:
                stages['curl'] = 'up'
            elif angle > 150 and stages['curl'] == 'up':
                stages['curl'] = 'down'
                counters['curl'] += 1

                feedback, is_good_form = analyze_bicep_curl_form(landmarks)
                if is_good_form:
                    counters['curl_good'] += 1
                else:
                    if 'curl_feedback' not in counters:
                        counters['curl_feedback'] = []
                    counters['curl_feedback'].extend(feedback)
        except Exception as e:
            pass

    elif exercise == 'press':
        try:
            left_shoulder = get_coord(landmarks, 'left', 'shoulder')
            left_elbow = get_coord(landmarks, 'left', 'elbow')
            left_wrist = get_coord(landmarks, 'left', 'wrist')
            angle = calculate_angle(left_shoulder, left_elbow, left_wrist)

            if angle > 140:
                stages['press'] = 'up'
            elif angle < 100 and stages['press'] == 'up':
                stages['press'] = 'down'
                counters['press'] += 1

                feedback, is_good_form = analyze_shoulder_press_form(landmarks)
                if is_good_form:
                    counters['press_good'] += 1
                else:
                    if 'press_feedback' not in counters:
                        counters['press_feedback'] = []
                    counters['press_feedback'].extend(feedback)
        except Exception as e:
            pass


# --- VideoThread Class ---
class VideoThread(QThread):
    change_pixmap_signal = Signal(QImage)
    update_stats_signal = Signal(dict)
    session_finished_signal = Signal(dict)

    def __init__(self):
        super().__init__()
        self._run_flag = True

    def run(self):
        if not MODEL_PATH.exists():
            print("❌ Model not found.")
            return

        print("🔄 Loading model...")
        model = load_model(MODEL_PATH, compile=False)
        print("✅ Model loaded successfully")

        sequence = deque(maxlen=SEQUENCE_LENGTH)
        self.counters = {'curl': 0, 'press': 0, 'curl_good': 0, 'press_good': 0}
        stages = {'curl': None, 'press': None}
        threshold = 0.45
        predictions_history = deque(maxlen=4)
        pred_class = ''
        feedback = []

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        with mp_pose.Pose(
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7,
                model_complexity=1,
                smooth_landmarks=True
        ) as pose:
            while self._run_flag and cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    continue

                frame = cv2.flip(frame, 1)
                image, results = mediapipe_detection(frame, pose)
                draw_landmarks_simplified(image, results)
                keypoints = extract_keypoints(results)
                sequence.append(keypoints)
                current_exercise = ""

                if len(sequence) == SEQUENCE_LENGTH:
                    res = model.predict(np.expand_dims(list(sequence), axis=0), verbose=0)[0]
                    pred_label = np.argmax(res)
                    confidence = np.max(res)

                    if confidence > threshold:
                        predicted_action = ACTIONS[pred_label]
                        predictions_history.append(predicted_action)

                        if len(predictions_history) >= 3:
                            pred_class = max(set(predictions_history), key=predictions_history.count)
                        else:
                            pred_class = predicted_action

                        current_exercise = pred_class.split('_')[0]

                        if results.pose_landmarks:
                            if current_exercise == 'curl':
                                feedback, _ = analyze_bicep_curl_form(results.pose_landmarks.landmark)
                            elif current_exercise == 'press':
                                feedback, _ = analyze_shoulder_press_form(results.pose_landmarks.landmark)

                        if results.pose_landmarks and pred_class:
                            count_reps(pred_class, results.pose_landmarks.landmark, self.counters, stages)
                    else:
                        feedback = []
                        pred_class = ''

                qt_image = self.convert_cv_to_qt(image)
                self.change_pixmap_signal.emit(qt_image)

                stats = {
                    'current_exercise': current_exercise.title() if current_exercise else "Detecting...",
                    'curl_total': self.counters.get('curl', 0),
                    'curl_good': self.counters.get('curl_good', 0),
                    'press_total': self.counters.get('press', 0),
                    'press_good': self.counters.get('press_good', 0),
                    'feedback': " | ".join(feedback[:2]) if feedback else "Great form! Keep going!"
                }
                self.update_stats_signal.emit(stats)

        cap.release()
        print("\nVideo thread stopped.")
        self.session_finished_signal.emit(self.counters)

    def convert_cv_to_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        return convert_to_Qt_format.scaled(1280, 720, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def stop(self):
        self._run_flag = False
        self.wait()


# --- WelcomeWidget ---
class WelcomeWidget(QWidget):
    start_button_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("WelcomePage")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)

        icon_label = QLabel("💪")
        icon_label.setObjectName("IconLabel")
        icon_label.setAlignment(Qt.AlignCenter)

        title = QLabel("AI FORM TRAINER")
        title.setObjectName("PageTitle")
        title.setAlignment(Qt.AlignCenter)

        subtitle = QLabel("Your intelligent workout assistant")
        subtitle.setObjectName("PageSubtitle")
        subtitle.setAlignment(Qt.AlignCenter)

        description = QLabel(
            "Get real-time feedback on your exercise form. Track your progress with AI-powered analysis")
        description.setObjectName("PageDescription")
        description.setAlignment(Qt.AlignCenter)

        self.start_button = QPushButton("START WORKOUT")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_button_clicked.emit)
        self.start_button.setCursor(Qt.PointingHandCursor)

        features_layout = QHBoxLayout()
        features_layout.setSpacing(40)

        for icon, text in [("🎯", "Real-time \nTracking"), ("📊", "Detailed \nAnalysis"), ("🏆", "Form \nCorrection")]:
            feature_box = QVBoxLayout()
            feature_icon = QLabel(icon)
            feature_icon.setObjectName("FeatureIcon")
            feature_icon.setAlignment(Qt.AlignCenter)
            feature_text = QLabel(text)
            feature_text.setObjectName("FeatureText")
            feature_text.setAlignment(Qt.AlignCenter)
            feature_box.addWidget(feature_icon)
            feature_box.addWidget(feature_text)
            features_layout.addLayout(feature_box)

        layout.addStretch(2)
        layout.addWidget(icon_label)
        layout.addSpacing(20)
        layout.addWidget(title)
        layout.addSpacing(10)
        layout.addWidget(subtitle)
        layout.addSpacing(20)
        layout.addWidget(description)
        layout.addSpacing(50)
        layout.addLayout(features_layout)
        layout.addSpacing(50)
        layout.addWidget(self.start_button, 0, Qt.AlignCenter)
        layout.addStretch(2)


# --- WorkoutWidget ---
class WorkoutWidget(QWidget):
    end_session_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("WorkoutPage")
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(30)
        main_layout.setContentsMargins(20, 20, 20, 20)

        video_container = QWidget()
        video_container.setObjectName("VideoContainer")
        video_layout = QVBoxLayout(video_container)
        video_layout.setContentsMargins(0, 0, 0, 0)

        self.video_label = QLabel("🎥 Initializing camera...")
        self.video_label.setObjectName("VideoFeed")
        self.video_label.setFixedSize(1280, 720)
        self.video_label.setAlignment(Qt.AlignCenter)

        status_bar = QWidget()
        status_bar.setObjectName("StatusBar")
        status_layout = QHBoxLayout(status_bar)
        status_layout.setContentsMargins(15, 10, 15, 10)

        self.status_indicator = QLabel("●")
        self.status_indicator.setObjectName("StatusIndicator")
        status_text = QLabel("LIVE")
        status_text.setObjectName("StatusText")

        status_layout.addWidget(self.status_indicator)
        status_layout.addWidget(status_text)
        status_layout.addStretch()

        video_layout.addWidget(self.video_label)
        video_layout.addWidget(status_bar)

        dashboard_container = QWidget()
        dashboard_container.setObjectName("DashboardContainer")
        dashboard_layout = QVBoxLayout(dashboard_container)
        dashboard_layout.setContentsMargins(30, 30, 30, 30)
        dashboard_layout.setSpacing(25)

        exercise_header = QHBoxLayout()
        exercise_label_text = QLabel("CURRENT EXERCISE")
        exercise_label_text.setObjectName("SectionLabel")
        exercise_header.addWidget(exercise_label_text)
        exercise_header.addStretch()

        self.current_exercise_label = QLabel("Detecting...")
        self.current_exercise_label.setObjectName("CurrentExercise")

        stats_container = QWidget()
        stats_container.setObjectName("StatsContainer")
        stats_grid = QGridLayout(stats_container)
        stats_grid.setSpacing(20)
        stats_grid.setContentsMargins(0, 0, 0, 0)

        curl_card = self._create_stat_card("💪 BICEP CURLS", "curl")
        stats_grid.addWidget(curl_card, 0, 0)

        press_card = self._create_stat_card("🏋️ SHOULDER PRESS", "press")
        stats_grid.addWidget(press_card, 0, 1)

        feedback_container = QWidget()
        feedback_container.setObjectName("FeedbackContainer")
        feedback_layout = QVBoxLayout(feedback_container)
        feedback_layout.setContentsMargins(20, 20, 20, 20)
        feedback_layout.setSpacing(15)

        feedback_header = QLabel("💡 COACH'S TIPS")
        feedback_header.setObjectName("FeedbackHeader")

        self.feedback_label = QLabel("Starting your workout session...")
        self.feedback_label.setObjectName("FeedbackText")
        self.feedback_label.setWordWrap(True)

        feedback_layout.addWidget(feedback_header)
        feedback_layout.addWidget(self.feedback_label)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)

        self.end_session_button = QPushButton("⏹ END SESSION")
        self.end_session_button.setObjectName("EndButton")
        self.end_session_button.clicked.connect(self.end_session_clicked.emit)
        self.end_session_button.setCursor(Qt.PointingHandCursor)

        buttons_layout.addWidget(self.end_session_button)

        dashboard_layout.addLayout(exercise_header)
        dashboard_layout.addWidget(self.current_exercise_label)
        dashboard_layout.addSpacing(20)
        dashboard_layout.addWidget(stats_container)
        dashboard_layout.addStretch()
        dashboard_layout.addWidget(feedback_container)
        dashboard_layout.addSpacing(20)
        dashboard_layout.addLayout(buttons_layout)

        main_layout.addWidget(video_container, 3)
        main_layout.addWidget(dashboard_container, 1)

        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._pulse_status)
        self.pulse_timer.start(1000)
        self.pulse_state = False

    def _pulse_status(self):
        self.pulse_state = not self.pulse_state
        color = "#FF3B3B" if self.pulse_state else "#FF6B6B"
        self.status_indicator.setStyleSheet(f"color: {color}; font-size: 16px;")

    def _create_stat_card(self, title, exercise_type):
        card = QWidget()
        card.setObjectName("StatCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(20, 20, 20, 20)
        card_layout.setSpacing(15)

        title_label = QLabel(title)
        title_label.setObjectName("CardTitle")

        total_label = QLabel("TOTAL REPS")
        total_label.setObjectName("CardSubtitle")

        total_value = QLabel("0")
        total_value.setObjectName("CardValue")
        setattr(self, f"{exercise_type}_total_label", total_value)

        progress_bar = QProgressBar()
        progress_bar.setObjectName("RepProgress")
        progress_bar.setTextVisible(False)
        progress_bar.setMaximum(100)
        progress_bar.setValue(0)
        setattr(self, f"{exercise_type}_progress", progress_bar)

        good_label = QLabel("GOOD FORM")
        good_label.setObjectName("CardSubtitle")

        good_value = QLabel("0")
        good_value.setObjectName("CardValueSmall")
        setattr(self, f"{exercise_type}_good_label", good_value)

        card_layout.addWidget(title_label)
        card_layout.addSpacing(10)
        card_layout.addWidget(total_label)
        card_layout.addWidget(total_value)
        card_layout.addWidget(progress_bar)
        card_layout.addSpacing(10)
        card_layout.addWidget(good_label)
        card_layout.addWidget(good_value)
        card_layout.addStretch()

        return card

    @Slot(QImage)
    def update_image(self, qt_image):
        self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    @Slot(dict)
    def update_stats(self, stats_dict):
        self.current_exercise_label.setText(stats_dict['current_exercise'])

        curl_total = stats_dict['curl_total']
        curl_good = stats_dict['curl_good']
        self.curl_total_label.setText(str(curl_total))
        self.curl_good_label.setText(str(curl_good))
        curl_accuracy = (curl_good / curl_total * 100) if curl_total > 0 else 0
        self.curl_progress.setValue(int(curl_accuracy))

        press_total = stats_dict['press_total']
        press_good = stats_dict['press_good']
        self.press_total_label.setText(str(press_total))
        self.press_good_label.setText(str(press_good))
        press_accuracy = (press_good / press_total * 100) if press_total > 0 else 0
        self.press_progress.setValue(int(press_accuracy))

        self.feedback_label.setText(stats_dict['feedback'])

    def reset_session(self):
        self.current_exercise_label.setText("Detecting...")
        self.curl_total_label.setText("0")
        self.curl_good_label.setText("0")
        self.curl_progress.setValue(0)
        self.press_total_label.setText("0")
        self.press_good_label.setText("0")
        self.press_progress.setValue(0)
        self.feedback_label.setText("Starting your workout session...")
        self.video_label.setText("🎥 Initializing camera...")
        self.video_label.setPixmap(QPixmap())


# --- SIMPLIFIED SummaryWidget with QPalette for visibility ---
class SummaryWidget(QWidget):
    """Session Summary - Simplified Two-Column Layout with Guaranteed Text Visibility"""
    home_button_clicked = Signal()

    def __init__(self):
        super().__init__()
        self.setObjectName("SummaryPage")
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignCenter)
        layout.setContentsMargins(50, 50, 50, 50)
        layout.setSpacing(30)

        # Header
        header_layout = QVBoxLayout()
        header_layout.setAlignment(Qt.AlignCenter)
        header_layout.setSpacing(10)

        icon = QLabel("🎉")
        icon_font = QFont("Segoe UI Emoji", 70)
        icon.setFont(icon_font)
        icon.setAlignment(Qt.AlignCenter)

        title = QLabel("Session Summary")
        title_font = QFont("Segoe UI", 48, QFont.Black)
        title.setFont(title_font)
        title.setStyleSheet("color: #FFFFFF;")
        title.setAlignment(Qt.AlignCenter)

        header_layout.addWidget(icon)
        header_layout.addWidget(title)

        # Two-column content layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(50)

        # Column 1: Bicep Curls
        curl_box = self._create_exercise_box("Bicep Curls", "curl")
        content_layout.addLayout(curl_box)

        # Column 2: Shoulder Press
        press_box = self._create_exercise_box("Shoulder Press", "press")
        content_layout.addLayout(press_box)

        # Back to Home Button
        button_layout = QHBoxLayout()
        self.home_button = QPushButton("Back to Home")
        self.home_button.setObjectName("PrimaryButton")
        self.home_button.clicked.connect(self.home_button_clicked.emit)
        self.home_button.setCursor(Qt.PointingHandCursor)

        button_layout.addStretch()
        button_layout.addWidget(self.home_button)
        button_layout.addStretch()

        # Assemble page
        layout.addStretch(1)
        layout.addLayout(header_layout)
        layout.addSpacing(40)
        layout.addLayout(content_layout, 3)
        layout.addStretch(1)
        layout.addLayout(button_layout)
        layout.addSpacing(20)

    def _create_exercise_box(self, title, exercise_type):
        """Creates a column for one exercise type with QPalette-based text"""
        box_layout = QVBoxLayout()
        box_layout.setSpacing(20)

        # Title
        title_label = QLabel(title)
        title_font = QFont("Segoe UI", 28, QFont.Bold)
        title_label.setFont(title_font)
        title_palette = title_label.palette()
        title_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        title_label.setPalette(title_palette)
        title_label.setAlignment(Qt.AlignCenter)

        # Total Reps
        total_label = QLabel("Total Reps: 0")
        total_font = QFont("Segoe UI", 20, QFont.Bold)
        total_label.setFont(total_font)
        total_palette = total_label.palette()
        total_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        total_label.setPalette(total_palette)
        setattr(self, f"{exercise_type}_total_label", total_label)

        # Good Form Reps
        good_label = QLabel("Good Form Reps: 0")
        good_font = QFont("Segoe UI", 20, QFont.Bold)
        good_label.setFont(good_font)
        good_palette = good_label.palette()
        good_palette.setColor(QPalette.WindowText, QColor(255, 255, 255))
        good_label.setPalette(good_palette)
        setattr(self, f"{exercise_type}_good_label", good_label)

        # Accuracy
        accuracy_label = QLabel("Accuracy: 0.0%")
        accuracy_font = QFont("Segoe UI", 24, QFont.Bold)
        accuracy_label.setFont(accuracy_font)
        accuracy_label.setStyleSheet("margin-top: 10px;")
        setattr(self, f"{exercise_type}_accuracy_label", accuracy_label)

        # Issues
        issues_label = QLabel("No issues.")
        issues_font = QFont("Segoe UI", 16)
        issues_label.setFont(issues_font)
        issues_palette = issues_label.palette()
        issues_palette.setColor(QPalette.WindowText, QColor(255, 187, 187))
        issues_label.setPalette(issues_palette)
        issues_label.setWordWrap(True)
        issues_label.setAlignment(Qt.AlignTop)
        setattr(self, f"{exercise_type}_issues_label", issues_label)

        # Assemble box
        box_layout.addWidget(title_label, 0, Qt.AlignCenter)
        box_layout.addSpacing(15)
        box_layout.addWidget(total_label)
        box_layout.addWidget(good_label)
        box_layout.addWidget(accuracy_label)
        box_layout.addSpacing(10)
        box_layout.addWidget(issues_label, 1)

        return box_layout

    def format_issues(self, issues_list):
        """Format issues list for display - FIXED VERSION"""
        if not issues_list:
            return "No issues detected."

        issue_counts = {}
        for issue in issues_list:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1

        issues_lines = ["Common Issues:"]
        for issue, count in sorted(issue_counts.items(), key=lambda x: x[1], reverse=True):
            issues_lines.append(f"- {issue}: {count} times")
        return "\n".join(issues_lines)

    @Slot(dict)
    def update_data(self, counters):
        """Update summary with session data"""
        print(f"\n{'=' * 60}")
        print("SUMMARY PAGE - UPDATING DATA")
        print(f"{'=' * 60}")
        print(f"Received counters: {counters}")

        # Curl Data
        curl_total = counters.get('curl', 0)
        curl_good = counters.get('curl_good', 0)
        curl_accuracy = (curl_good / curl_total * 100) if curl_total > 0 else 100

        self.curl_total_label.setText(f"Total Reps: {curl_total}")
        self.curl_good_label.setText(f"Good Form Reps: {curl_good}")
        self.curl_accuracy_label.setText(f"Accuracy: {curl_accuracy:.1f}%")
        self.curl_issues_label.setText(self.format_issues(counters.get('curl_feedback')))

        # Set accuracy color
        curl_color = '#00F260' if curl_accuracy > 75 else '#FFBABA'
        self.curl_accuracy_label.setStyleSheet(
            f"color: {curl_color}; font-size: 24px; font-weight: 700; margin-top: 5px;"
        )

        print(f"\nCURL: Total={curl_total}, Good={curl_good}, Accuracy={curl_accuracy:.1f}%")

        # Press Data
        press_total = counters.get('press', 0)
        press_good = counters.get('press_good', 0)
        press_accuracy = (press_good / press_total * 100) if press_total > 0 else 100

        self.press_total_label.setText(f"Total Reps: {press_total}")
        self.press_good_label.setText(f"Good Form Reps: {press_good}")
        self.press_accuracy_label.setText(f"Accuracy: {press_accuracy:.1f}%")
        self.press_issues_label.setText(self.format_issues(counters.get('press_feedback')))

        # Set accuracy color
        press_color = '#00F260' if press_accuracy > 75 else '#FFBABA'
        self.press_accuracy_label.setStyleSheet(
            f"color: {press_color}; font-size: 24px; font-weight: 700; margin-top: 5px;"
        )

        print(f"PRESS: Total={press_total}, Good={press_good}, Accuracy={press_accuracy:.1f}%")
        print(f"{'=' * 60}\n")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AI Form Trainer - Professional Workout Assistant")
        self.video_thread = None

        self.stacked_widget = QStackedWidget()
        self.setCentralWidget(self.stacked_widget)

        self.welcome_page = WelcomeWidget()
        self.workout_page = WorkoutWidget()
        self.summary_page = SummaryWidget()

        self.stacked_widget.addWidget(self.welcome_page)
        self.stacked_widget.addWidget(self.workout_page)
        self.stacked_widget.addWidget(self.summary_page)

        self.welcome_page.start_button_clicked.connect(self.start_workout)
        self.workout_page.end_session_clicked.connect(self.end_workout)
        self.summary_page.home_button_clicked.connect(self.go_to_home)

        self.showMaximized()
        self.go_to_home()

    def start_workout(self):
        self.workout_page.reset_session()
        self.stacked_widget.setCurrentWidget(self.workout_page)
        self.video_thread = VideoThread()
        self.video_thread.change_pixmap_signal.connect(self.workout_page.update_image)
        self.video_thread.update_stats_signal.connect(self.workout_page.update_stats)
        self.video_thread.session_finished_signal.connect(self.show_summary)
        self.video_thread.start()

    def end_workout(self):
        if self.video_thread:
            self.video_thread.stop()

    def show_summary(self, counters_dict):
        print(f"\n🎯 MainWindow: Displaying summary page")
        print(f"   Data to display: {counters_dict}")
        self.summary_page.update_data(counters_dict)
        self.stacked_widget.setCurrentWidget(self.summary_page)

    def go_to_home(self):
        self.stacked_widget.setCurrentWidget(self.welcome_page)

    def closeEvent(self, event):
        self.end_workout()
        event.accept()


# --- Stylesheet ---
STYLESHEET = """
QWidget {
    background-color: #0A0E1A;
    color: #FFFFFF;
    font-family: 'Segoe UI', 'SF Pro Display', 'Arial', sans-serif;
}

QWidget#WelcomePage {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
                stop:0 #0A0E1A, stop:0.5 #12182B, stop:1 #1A1F35);
}

QLabel#IconLabel {
    font-size: 120px;
    margin: 20px;
}

QLabel#PageTitle {
    font-size: 64px;
    font-weight: 900;
    color: #FFFFFF;
    letter-spacing: 2px;
}

QLabel#PageSubtitle {
    font-size: 22px;
    color: #C0C8E0;
    font-weight: 500;
    letter-spacing: 1px;
}

QLabel#PageDescription {
    font-size: 16px;
    color: #9BA5C8;
}

QLabel#FeatureIcon {
    font-size: 48px;
}

QLabel#FeatureText {
    font-size: 14px;
    color: #9BA5C8;
    font-weight: 600;
}

QPushButton {
    border: none;
    border-radius: 12px;
    font-size: 16px;
    font-weight: 700;
    padding: 18px 40px;
    min-width: 220px;
    letter-spacing: 1px;
}

QPushButton#PrimaryButton {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00F260, stop:1 #0575E6);
    color: #000000;
}

QPushButton#PrimaryButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00FF6F, stop:1 #0A84FF);
}

QPushButton#EndButton {
    background-color: #FF4757;
    color: #FFFFFF;
}

QPushButton#EndButton:hover {
    background-color: #FF3838;
}

QWidget#WorkoutPage {
    background-color: #0A0E1A;
}

QWidget#VideoContainer {
    background-color: #12182B;
    border-radius: 20px;
}

QLabel#VideoFeed {
    border: 3px solid #1E2742;
    border-radius: 20px;
    background-color: #000000;
    font-size: 20px;
    color: #666;
}

QWidget#StatusBar {
    background-color: #1A2035;
    border-radius: 10px;
    margin-top: 10px;
}

QLabel#StatusIndicator {
    color: #FF3B3B;
    font-size: 16px;
    font-weight: bold;
}

QLabel#StatusText {
    color: #FF6B6B;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 1px;
}

QWidget#DashboardContainer {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #12182B, stop:1 #1A2035);
    border-radius: 20px;
    border: 1px solid #2A3045;
}

QLabel#SectionLabel {
    font-size: 12px;
    color: #9BA5C8;
    font-weight: 700;
    letter-spacing: 2px;
}

QLabel#CurrentExercise {
    font-size: 32px;
    font-weight: 800;
    color: #00F260;
    padding: 15px;
    background-color: rgba(0, 242, 96, 0.1);
    border-radius: 15px;
    border-left: 4px solid #00F260;
}

QWidget#StatsContainer {
    background-color: transparent;
}

QWidget#StatCard {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1A2035, stop:1 #12182B);
    border-radius: 18px;
    border: 2px solid #2A3550;
}

QLabel#CardTitle {
    font-size: 14px;
    font-weight: 700;
    color: #9BA5C8;
    letter-spacing: 1px;
}

QLabel#CardSubtitle {
    font-size: 11px;
    color: #7B85A8;
    font-weight: 600;
    letter-spacing: 1px;
}

QLabel#CardValue {
    font-size: 56px;
    font-weight: 900;
    color: #00F260;
}

QLabel#CardValueSmall {
    font-size: 28px;
    font-weight: 700;
    color: #FFFFFF;
}

QProgressBar#RepProgress {
    border: none;
    border-radius: 8px;
    background-color: #1A2035;
    height: 12px;
}

QProgressBar#RepProgress::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #00F260, stop:1 #0575E6);
    border-radius: 8px;
}

QWidget#FeedbackContainer {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #1A2540, stop:1 #1A2035);
    border-radius: 15px;
    border: 2px solid #2A3550;
}

QLabel#FeedbackHeader {
    font-size: 14px;
    font-weight: 700;
    color: #00F260;
    letter-spacing: 1px;
}

QLabel#FeedbackText {
    font-size: 16px;
    color: #FFFFFF;
    line-height: 1.6;
    min-height: 60px;
}

QWidget#SummaryPage {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #0A0E1A, stop:0.5 #12182B, stop:1 #1A1F35);
}
"""

if __name__ == "__main__":
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}")
        print("Please ensure 'models/exercise_form_model.h5' exists.")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setStyleSheet(STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())