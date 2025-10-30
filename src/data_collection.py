# import os
# import cv2
# import numpy as np
# from pathlib import Path
# import mediapipe as mp
# import random
# import json
# import logging
# from typing import List, Tuple, Optional
# import argparse
#
# # Setup logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s'
# )
# logger = logging.getLogger(__name__)
#
# # --- Constants and Setup ---
# ACTIONS = np.array(['curl', 'press'])  # 2 classes
# SEQUENCE_LENGTH = 30
# NUM_LANDMARKS = 33
# NUM_DIMS = 4  # x, y, z, visibility
# INPUT_SIZE = NUM_LANDMARKS * NUM_DIMS
#
# # --- Configuration ---
# DATA_DIR = Path.cwd() / "data_2class"
# VIDEO_PATH = Path.cwd() / "videos"
# NUM_AUGMENTATIONS_PER_SEQUENCE = 5
# VISIBILITY_THRESHOLD = 0.7
# MIN_VISIBLE_LANDMARKS_RATIO = 0.8
# TEMPORAL_JUMP_THRESHOLD = 0.15
#
# # Critical landmarks for exercise form
# CRITICAL_LANDMARKS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26]
#
# # Exercises that should NOT be flipped
# UNILATERAL_EXERCISES = []
#
# # --- MediaPipe Setup ---
# mp_pose = mp.solutions.pose
# mp_drawing = mp.solutions.drawing_utils
#
# # Landmark flip mapping
# LANDMARK_FLIP_MAP = [
#     (1, 2), (3, 6), (4, 5), (7, 8), (9, 10),  # Face
#     (11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22),  # Body
#     (23, 24), (25, 26), (27, 28), (29, 30), (31, 32)  # Limbs
# ]
#
# def ensure_dirs(*dirs: Path):
#     """Create directories if they don't exist."""
#     for d in dirs:
#         d.mkdir(parents=True, exist_ok=True)
#
# def calculate_angle(point1: np.ndarray, point2: np.ndarray, point3: np.ndarray) -> float:
#     """Calculate angle between three points (in degrees)."""
#     vector1 = point1 - point2
#     vector2 = point3 - point2
#     cosine_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2) + 1e-6)
#     cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
#     angle = np.arccos(cosine_angle)
#     return np.degrees(angle)
#
# def extract_biomechanical_features(keypoints: np.ndarray) -> np.ndarray:
#     """Extract biomechanical features: joint angles."""
#     kps = keypoints.reshape(NUM_LANDMARKS, NUM_DIMS)
#     features = []
#
#     try:
#         # Right elbow angle
#         if all(kps[[12, 14, 16], 3] > VISIBILITY_THRESHOLD):
#             right_elbow_angle = calculate_angle(kps[12, :3], kps[14, :3], kps[16, :3])
#             features.append(right_elbow_angle)
#         else:
#             features.append(0.0)
#
#         # Left elbow angle
#         if all(kps[[11, 13, 15], 3] > VISIBILITY_THRESHOLD):
#             left_elbow_angle = calculate_angle(kps[11, :3], kps[13, :3], kps[15, :3])
#             features.append(left_elbow_angle)
#         else:
#             features.append(0.0)
#
#         # Right shoulder angle
#         if all(kps[[24, 12, 14], 3] > VISIBILITY_THRESHOLD):
#             right_shoulder_angle = calculate_angle(kps[24, :3], kps[12, :3], kps[14, :3])
#             features.append(right_shoulder_angle)
#         else:
#             features.append(0.0)
#
#         # Left shoulder angle
#         if all(kps[[23, 11, 13], 3] > VISIBILITY_THRESHOLD):
#             left_shoulder_angle = calculate_angle(kps[23, :3], kps[11, :3], kps[13, :3])
#             features.append(left_shoulder_angle)
#         else:
#             features.append(0.0)
#
#         # Right hip angle
#         if all(kps[[12, 24, 26], 3] > VISIBILITY_THRESHOLD):
#             right_hip_angle = calculate_angle(kps[12, :3], kps[24, :3], kps[26, :3])
#             features.append(right_hip_angle)
#         else:
#             features.append(0.0)
#
#         # Left hip angle
#         if all(kps[[11, 23, 25], 3] > VISIBILITY_THRESHOLD):
#             left_hip_angle = calculate_angle(kps[11, :3], kps[23, :3], kps[25, :3])
#             features.append(left_hip_angle)
#         else:
#             features.append(0.0)
#
#         # Right knee angle
#         if all(kps[[24, 26, 28], 3] > VISIBILITY_THRESHOLD):
#             right_knee_angle = calculate_angle(kps[24, :3], kps[26, :3], kps[28, :3])
#             features.append(right_knee_angle)
#         else:
#             features.append(0.0)
#
#         # Left knee angle
#         if all(kps[[23, 25, 27], 3] > VISIBILITY_THRESHOLD):
#             left_knee_angle = calculate_angle(kps[23, :3], kps[25, :3], kps[27, :3])
#             features.append(left_knee_angle)
#         else:
#             features.append(0.0)
#
#     except Exception as e:
#         logger.warning(f"Error calculating angles: {e}")
#         features = [0.0] * 8
#
#     return np.array(features, dtype=np.float32)
#
# def mediapipe_detection(image, model):
#     """Process image with MediaPipe pose detection."""
#     try:
#         if image is None or image.size == 0:
#             return None, None
#         image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#         image_rgb.flags.writeable = False
#         results = model.process(image_rgb)
#         image_rgb.flags.writeable = True
#         image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
#         return image_bgr, results
#     except Exception as e:
#         logger.error(f"Error in mediapipe_detection: {e}")
#         return image, None
#
# def draw_landmarks(image, results):
#     """Draw pose landmarks on image."""
#     if results and results.pose_landmarks:
#         mp_drawing.draw_landmarks(
#             image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#         )
#
# def extract_keypoints(results) -> np.ndarray:
#     """Extract pose keypoints from MediaPipe results."""
#     pose = np.zeros(INPUT_SIZE)
#     if results and results.pose_landmarks:
#         pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility]
#                          for lmk in results.pose_landmarks.landmark]).flatten()
#     return pose
#
# def validate_sequence_quality(sequence: List[np.ndarray]) -> Tuple[bool, float]:
#     """Validate sequence quality based on landmark visibility and temporal consistency."""
#     if len(sequence) != SEQUENCE_LENGTH:
#         return False, 0.0
#
#     quality_scores = []
#     for i, keypoints in enumerate(sequence):
#         kps = keypoints.reshape(NUM_LANDMARKS, NUM_DIMS)
#
#         # Check visibility of critical landmarks
#         critical_kps = kps[CRITICAL_LANDMARKS]
#         visible_count = np.sum(critical_kps[:, 3] > VISIBILITY_THRESHOLD)
#         visibility_ratio = visible_count / len(CRITICAL_LANDMARKS)
#
#         if visibility_ratio < MIN_VISIBLE_LANDMARKS_RATIO:
#             return False, 0.0
#
#         quality_scores.append(visibility_ratio)
#
#         # Check temporal consistency
#         if i > 0:
#             prev_kps = sequence[i - 1].reshape(NUM_LANDMARKS, NUM_DIMS)
#             movement = np.linalg.norm(kps[:, :2] - prev_kps[:, :2], axis=1)
#             max_movement = np.max(movement[CRITICAL_LANDMARKS])
#
#             if max_movement > TEMPORAL_JUMP_THRESHOLD:
#                 logger.debug(f"Temporal jump detected at frame {i}: {max_movement:.3f}")
#                 return False, 0.0
#
#     avg_quality = np.mean(quality_scores)
#     return True, avg_quality
#
# # --- Augmentation Functions ---
# def flip_keypoints(keypoints: np.ndarray) -> np.ndarray:
#     """Horizontally flip keypoints."""
#     flipped_kps = keypoints.copy().reshape(NUM_LANDMARKS, NUM_DIMS)
#     flipped_kps[:, 0] = 1.0 - flipped_kps[:, 0]
#     for left_idx, right_idx in LANDMARK_FLIP_MAP:
#         flipped_kps[[left_idx, right_idx]] = flipped_kps[[right_idx, left_idx]]
#     return flipped_kps.flatten()
#
# def rotate_keypoints(keypoints: np.ndarray, angle_deg: float) -> np.ndarray:
#     """Rotate keypoints around pose center."""
#     kps = keypoints.copy().reshape(NUM_LANDMARKS, NUM_DIMS)
#     visible_kps = kps[kps[:, 3] > VISIBILITY_THRESHOLD]
#
#     if len(visible_kps) == 0:
#         return keypoints
#
#     center_x = np.mean(visible_kps[:, 0])
#     center_y = np.mean(visible_kps[:, 1])
#     angle_rad = np.radians(angle_deg)
#     cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
#     rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
#
#     kps[:, :2] = (kps[:, :2] - [center_x, center_y]) @ rotation_matrix.T + [center_x, center_y]
#     return kps.flatten()
#
# def scale_keypoints(keypoints: np.ndarray, scale_factor: float) -> np.ndarray:
#     """Scale keypoints from pose center."""
#     kps = keypoints.copy().reshape(NUM_LANDMARKS, NUM_DIMS)
#     visible_kps = kps[kps[:, 3] > VISIBILITY_THRESHOLD]
#
#     if len(visible_kps) == 0:
#         return keypoints
#
#     center_x = np.mean(visible_kps[:, 0])
#     center_y = np.mean(visible_kps[:, 1])
#     kps[:, :2] = (kps[:, :2] - [center_x, center_y]) * scale_factor + [center_x, center_y]
#     return kps.flatten()
#
# def translate_keypoints(keypoints: np.ndarray, dx: float, dy: float) -> np.ndarray:
#     """Translate keypoints by delta."""
#     kps = keypoints.copy().reshape(NUM_LANDMARKS, NUM_DIMS)
#     kps[:, 0] += dx
#     kps[:, 1] += dy
#     return kps.flatten()
#
# def add_noise_to_keypoints(keypoints: np.ndarray, noise_level: float = 0.01) -> np.ndarray:
#     """Add Gaussian noise to keypoints."""
#     noise = np.random.normal(0, noise_level, keypoints.shape)
#     reshaped_noise = noise.reshape(NUM_LANDMARKS, NUM_DIMS)
#     reshaped_noise[:, 3] = 0  # No noise on visibility
#     return keypoints + reshaped_noise.flatten()
#
# def apply_augmentations(sequence: List[np.ndarray], action: str) -> List[np.ndarray]:
#     """Apply random augmentations to sequence."""
#     augmented_sequence = []
#
#     do_flip = random.random() < 0.5 and action not in UNILATERAL_EXERCISES
#     angle = random.uniform(-25, 25)
#     scale = random.uniform(0.85, 1.15)
#     dx = random.uniform(-0.08, 0.08)
#     dy = random.uniform(-0.08, 0.08)
#     noise_level = random.uniform(0.005, 0.015)
#
#     for keypoints in sequence:
#         aug_kps = keypoints.copy()
#         if do_flip:
#             aug_kps = flip_keypoints(aug_kps)
#         aug_kps = rotate_keypoints(aug_kps, angle)
#         aug_kps = scale_keypoints(aug_kps, scale)
#         aug_kps = translate_keypoints(aug_kps, dx, dy)
#         aug_kps = add_noise_to_keypoints(aug_kps, noise_level)
#
#         # Clip values
#         aug_kps_reshaped = aug_kps.reshape(NUM_LANDMARKS, NUM_DIMS)
#         aug_kps_reshaped[:, :2] = np.clip(aug_kps_reshaped[:, :2], 0, 1)
#         augmented_sequence.append(aug_kps_reshaped.flatten())
#
#     return augmented_sequence
#
# def save_sequence_with_metadata(sequence: List[np.ndarray], seq_dir: Path, metadata: dict):
#     """Save sequence with metadata JSON."""
#     ensure_dirs(seq_dir)
#
#     # Save sequence frames
#     for j, keypoints in enumerate(sequence):
#         bio_features = extract_biomechanical_features(keypoints)
#         combined_features = np.concatenate([keypoints, bio_features])
#         np.save(seq_dir / f"{j}.npy", combined_features)
#
#     # Save metadata
#     with open(seq_dir / "metadata.json", 'w') as f:
#         json.dump(metadata, f, indent=2)
#
# # --- Webcam Data Collection ---
# def collect_data_from_webcam():
#     """
#     NEW: Collect exercise data directly from webcam.
#     Press keys to select exercise type, then perform reps.
#     """
#     ensure_dirs(DATA_DIR)
#
#     print("\n" + "="*60)
#     print("🎥 WEBCAM DATA COLLECTION MODE")
#     print("="*60)
#     print("\nInstructions:")
#     print("  Press '1' = Start recording BICEP CURLS")
#     print("  Press '2' = Start recording SHOULDER PRESS")
#     print("  Press 's' = Stop current recording")
#     print("  Press 'q' = Quit")
#     print("\nTip: Position yourself so full body is visible")
#     print("="*60 + "\n")
#
#     cap = cv2.VideoCapture(0)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
#
#     recording = False
#     current_action = None
#     sequence_data = []
#     session_counter = {"curl": 0, "press": 0}
#
#     with mp_pose.Pose(
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.7,
#         model_complexity=1
#     ) as pose:
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#
#             frame = cv2.flip(frame, 1)
#             image, results = mediapipe_detection(frame, pose)
#
#             if image is None:
#                 continue
#
#             draw_landmarks(image, results)
#             keypoints = extract_keypoints(results)
#
#             # UI Display
#             if recording and current_action:
#                 cv2.rectangle(image, (0, 0), (image.shape[1], 60), (0, 0, 200), -1)
#                 action_name = "BICEP CURL" if current_action == "curl" else "SHOULDER PRESS"
#                 cv2.putText(image, f"🔴 RECORDING: {action_name}",
#                            (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
#                 cv2.putText(image, f"Frames: {len(sequence_data)}",
#                            (image.shape[1]-250, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#
#                 # Record keypoints
#                 if np.sum(keypoints) > 0:
#                     sequence_data.append(keypoints)
#             else:
#                 cv2.rectangle(image, (0, 0), (image.shape[1], 100), (50, 50, 50), -1)
#                 cv2.putText(image, "Ready to Record",
#                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#                 cv2.putText(image, "1: Bicep Curl | 2: Shoulder Press | q: Quit",
#                            (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
#
#             # Session stats
#             cv2.putText(image, f"Curl sequences: {session_counter['curl']}",
#                        (20, image.shape[0]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#             cv2.putText(image, f"Press sequences: {session_counter['press']}",
#                        (20, image.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
#             cv2.imshow('Webcam Data Collection', image)
#
#             key = cv2.waitKey(10) & 0xFF
#
#             if key == ord('q'):
#                 break
#             elif key == ord('1') and not recording:
#                 # Start recording bicep curls
#                 recording = True
#                 current_action = "curl"
#                 sequence_data = []
#                 logger.info("🔴 Started recording BICEP CURLS")
#             elif key == ord('2') and not recording:
#                 # Start recording shoulder press
#                 recording = True
#                 current_action = "press"
#                 sequence_data = []
#                 logger.info("🔴 Started recording SHOULDER PRESS")
#             elif key == ord('s') and recording:
#                 # Stop recording and save
#                 recording = False
#                 logger.info(f"⏹️  Stopped recording. Frames collected: {len(sequence_data)}")
#
#                 # Save sequences
#                 if len(sequence_data) >= SEQUENCE_LENGTH:
#                     sequences_saved = 0
#                     sequences_rejected = 0
#                     min_gap = SEQUENCE_LENGTH // 3
#
#                     for i in range(0, len(sequence_data) - SEQUENCE_LENGTH + 1, min_gap):
#                         sequence = sequence_data[i:i + SEQUENCE_LENGTH]
#
#                         if len(sequence) == SEQUENCE_LENGTH:
#                             is_valid, quality_score = validate_sequence_quality(sequence)
#
#                             if not is_valid:
#                                 sequences_rejected += 1
#                                 continue
#
#                             # Metadata
#                             metadata = {
#                                 "source": "webcam",
#                                 "action": current_action,
#                                 "start_frame": i,
#                                 "end_frame": i + SEQUENCE_LENGTH,
#                                 "quality_score": float(quality_score),
#                                 "is_augmented": False,
#                                 "sequence_length": SEQUENCE_LENGTH,
#                                 "session_id": session_counter[current_action]
#                             }
#
#                             # Save original
#                             seq_dir = DATA_DIR / current_action / f"webcam_session{session_counter[current_action]}_orig_{i}"
#                             save_sequence_with_metadata(sequence, seq_dir, metadata)
#                             sequences_saved += 1
#
#                             # Save augmented versions
#                             for aug_num in range(NUM_AUGMENTATIONS_PER_SEQUENCE):
#                                 augmented_sequence = apply_augmentations(sequence, current_action)
#                                 aug_metadata = metadata.copy()
#                                 aug_metadata["is_augmented"] = True
#                                 aug_metadata["augmentation_id"] = aug_num
#                                 aug_seq_dir = DATA_DIR / current_action / f"webcam_session{session_counter[current_action]}_aug_{i}_{aug_num}"
#                                 save_sequence_with_metadata(augmented_sequence, aug_seq_dir, aug_metadata)
#                                 sequences_saved += 1
#
#                     session_counter[current_action] += 1
#                     logger.info(f"✅ Saved {sequences_saved} sequences (rejected {sequences_rejected})")
#                 else:
#                     logger.warning(f"❌ Not enough frames collected: {len(sequence_data)}/{SEQUENCE_LENGTH}")
#
#                 sequence_data = []
#                 current_action = None
#
#     cap.release()
#     cv2.destroyAllWindows()
#     print_summary()
#
# # --- Video Data Collection (Original) ---
#
# def collect_data_from_video(show_display: bool = False):
#     """
#     Enhanced data collection from video files with validation.
#     """
#     ensure_dirs(DATA_DIR)
#
#     if not VIDEO_PATH.exists():
#         logger.error(f"Video directory not found: {VIDEO_PATH}")
#         return
#
#     video_files = list(VIDEO_PATH.glob("*.mp4")) + list(VIDEO_PATH.glob("*.avi"))
#
#     if not video_files:
#         logger.error(f"No video files found in {VIDEO_PATH}")
#         return
#
#     logger.info(f"Found {len(video_files)} video files. Starting processing...")
#     logger.info(f"Each original sequence will be augmented {NUM_AUGMENTATIONS_PER_SEQUENCE} times.")
#
#     with mp_pose.Pose(
#         min_detection_confidence=0.7,
#         min_tracking_confidence=0.7,
#         model_complexity=1
#     ) as pose:
#         for video_idx, video_file in enumerate(video_files):
#             logger.info(f"\n🎥 Processing: {video_file.name}")
#
#             action = next((act for act in ACTIONS if act in video_file.name.lower()), None)
#             if not action:
#                 logger.warning(f"Skipping {video_file.name} (no matching action found)")
#                 continue
#
#             try:
#                 cap = cv2.VideoCapture(str(video_file))
#                 if not cap.isOpened():
#                     logger.error(f"Could not open video: {video_file.name}")
#                     continue
#
#                 fps = cap.get(cv2.CAP_PROP_FPS)
#                 total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#                 sequence_data = []
#                 frame_count = 0
#
#                 while cap.isOpened():
#                     ret, frame = cap.read()
#                     if not ret:
#                         break
#
#                     frame_count += 1
#                     frame = cv2.resize(frame, (640, 480))
#                     image, results = mediapipe_detection(frame, pose)
#
#                     if image is None:
#                         continue
#
#                     if show_display:
#                         draw_landmarks(image, results)
#                         cv2.putText(
#                             image, f"Processing: {action} - Frame: {frame_count}/{total_frames}",
#                             (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2
#                         )
#                         cv2.imshow('Processing Video', image)
#
#                         if cv2.waitKey(1) & 0xFF == ord('q'):
#                             cap.release()
#                             cv2.destroyAllWindows()
#                             return
#
#                     keypoints = extract_keypoints(results)
#                     if np.sum(keypoints) > 0:
#                         sequence_data.append(keypoints)
#
#                 cap.release()
#
#                 # Save sequences with validation
#                 sequences_saved_total = 0
#                 sequences_rejected = 0
#                 min_gap = SEQUENCE_LENGTH // 3
#
#                 for i in range(0, len(sequence_data) - SEQUENCE_LENGTH + 1, min_gap):
#                     sequence = sequence_data[i:i + SEQUENCE_LENGTH]
#
#                     if len(sequence) == SEQUENCE_LENGTH:
#                         is_valid, quality_score = validate_sequence_quality(sequence)
#
#                         if not is_valid:
#                             sequences_rejected += 1
#                             continue
#
#                         metadata = {
#                             "video_file": video_file.name,
#                             "action": action,
#                             "start_frame": i,
#                             "end_frame": i + SEQUENCE_LENGTH,
#                             "fps": fps,
#                             "quality_score": float(quality_score),
#                             "is_augmented": False,
#                             "sequence_length": SEQUENCE_LENGTH
#                         }
#
#                         # Save original
#                         seq_dir = DATA_DIR / action / f"{video_file.stem}_orig_{i}"
#                         save_sequence_with_metadata(sequence, seq_dir, metadata)
#                         sequences_saved_total += 1
#
#                         # Save augmented versions
#                         for aug_num in range(NUM_AUGMENTATIONS_PER_SEQUENCE):
#                             augmented_sequence = apply_augmentations(sequence, action)
#                             aug_metadata = metadata.copy()
#                             aug_metadata["is_augmented"] = True
#                             aug_metadata["augmentation_id"] = aug_num
#                             aug_seq_dir = DATA_DIR / action / f"{video_file.stem}_aug_{i}_{aug_num}"
#                             save_sequence_with_metadata(augmented_sequence, aug_seq_dir, aug_metadata)
#                             sequences_saved_total += 1
#
#                 logger.info(
#                     f" ✅ Saved {sequences_saved_total} sequences from {video_file.name} "
#                     f"(rejected {sequences_rejected} low-quality sequences)"
#                 )
#
#             except Exception as e:
#                 logger.error(f"Error processing {video_file.name}: {e}")
#                 continue
#
#     if show_display:
#         cv2.destroyAllWindows()
#
#     print_summary()
#
# def print_summary():
#     """Print data collection summary."""
#     logger.info(f"\n📈 Data Collection Summary (in {DATA_DIR}):")
#     total_sequences = 0
#     for action in ACTIONS:
#         action_dir = DATA_DIR / action
#         if action_dir.exists():
#             seq_count = len(list(action_dir.iterdir()))
#             total_sequences += seq_count
#             logger.info(f"  {action}: {seq_count} sequences")
#     logger.info(f"\n  Total sequences collected: {total_sequences}")
#
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Exercise form data collection - 2 Classes")
#     parser.add_argument(
#         '--mode',
#         choices=['webcam', 'video', 'both'],
#         default='both',
#         help='Data collection mode: webcam, video, or both'
#     )
#     parser.add_argument(
#         '--show-display',
#         action='store_true',
#         help='Show processing window for video mode (slows down processing)'
#     )
#
#     args = parser.parse_args()
#
#     if args.mode in ['webcam', 'both']:
#         logger.info("\n🎥 Starting WEBCAM data collection...")
#         collect_data_from_webcam()
#
#     if args.mode in ['video', 'both']:
#         logger.info("\n📹 Starting VIDEO data collection...")
#         collect_data_from_video(show_display=args.show_display)


import os
import cv2
import numpy as np
from pathlib import Path
import mediapipe as mp
import random

ACTIONS = np.array(['curl_correct', 'curl_incorrect', 'press_correct', 'press_incorrect'])
SEQUENCE_LENGTH = 30
NUM_LANDMARKS = 33
NUM_DIMS = 4
INPUT_SIZE = NUM_LANDMARKS * NUM_DIMS
DATA_DIR = Path.cwd() / "data2"
VIDEO_PATH = Path.cwd() / "videos"
COLOURS = [(245, 117, 16), (117, 245, 16), (16, 117, 245), (245, 16, 117)]

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def ensure_dirs(*dirs: Path):
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)


def mediapipe_detection(image, model):
    """Enhanced MediaPipe detection with error handling"""
    try:
        if image is None or image.size == 0:
            return None, None

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_rgb.flags.writeable = False
        results = model.process(image_rgb)
        image_rgb.flags.writeable = True
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        return image_bgr, results
    except Exception as e:
        print(f"Error in mediapipe_detection: {e}")
        return image, None


def draw_landmarks(image, results):
    """Draw pose landmarks on image"""
    if results and results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
        )


def extract_keypoints(results):
    """Extract keypoints from MediaPipe results"""
    pose = np.zeros(INPUT_SIZE)
    if results and results.pose_landmarks:
        pose = np.array([[lmk.x, lmk.y, lmk.z, lmk.visibility]
                         for lmk in results.pose_landmarks.landmark]).flatten()
    return pose


def add_noise_augmentation(keypoints, noise_factor=0.01):
    """Add slight noise for data augmentation"""
    noise = np.random.normal(0, noise_factor, keypoints.shape)
    return keypoints + noise


def collect_data_from_webcam():
    """Collect data from live webcam with manual start/stop control"""
    ensure_dirs(DATA_DIR)

    print("\n🎥 WEBCAM DATA COLLECTION MODE")
    print("=" * 60)
    print("\nAvailable actions:")
    for idx, action in enumerate(ACTIONS):
        print(f"  {idx}: {action}")

    action_idx = int(input("\nSelect action number: "))
    if action_idx < 0 or action_idx >= len(ACTIONS):
        print("❌ Invalid action selection")
        return

    action = ACTIONS[action_idx]

    print("\n📝 Instructions:")
    print("  - Press 'S' to START recording a sequence")
    print("  - Press 'E' to END/STOP the current sequence")
    print("  - Press 'Q' to quit data collection")
    print("\n⚠️  Important: Each sequence needs at least 30 frames")
    print("              Record for 3-5 seconds per sequence")
    print("\nStarting webcam...")

    cap = cv2.VideoCapture(0)

    # Set webcam properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    if not cap.isOpened():
        print("❌ Could not open webcam")
        return

    with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
    ) as pose:

        sequence_data = []
        recording = False
        sequences_saved = 0
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame")
                break

            # Flip for mirror effect
            frame = cv2.flip(frame, 1)

            image, results = mediapipe_detection(frame, pose)

            if image is None:
                continue

            draw_landmarks(image, results)
            keypoints = extract_keypoints(results)

            # Recording logic - continuous capture
            if recording:
                if np.sum(keypoints) > 0:
                    sequence_data.append(keypoints)
                    frame_count += 1

                    # Visual feedback during recording
                    cv2.rectangle(image, (0, 0), (640, 480), (0, 0, 255), 15)
                    cv2.putText(image, f"🔴 RECORDING: {frame_count} frames",
                                (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(image, "Press 'E' to STOP",
                                (150, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                else:
                    # Warn if pose not detected during recording
                    cv2.putText(image, "⚠️ NO POSE DETECTED!",
                                (120, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)

            # Display instructions
            action_label = action.replace("_", " ").title()
            cv2.putText(image, f"Action: {action_label}", (15, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[action_idx % len(COLOURS)], 2)
            cv2.putText(image, f"Sequences Saved: {sequences_saved}", (15, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            if not recording:
                cv2.putText(image, "Press 'S' to START recording", (100, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Show detection status
            detection_color = (0, 255, 0) if np.sum(keypoints) > 0 else (0, 0, 255)
            cv2.circle(image, (600, 30), 10, detection_color, -1)
            cv2.putText(image, "Pose", (555, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            cv2.imshow('Webcam Data Collection', image)

            key = cv2.waitKey(10) & 0xFF

            # Start recording
            if key == ord('s') or key == ord('S'):
                if not recording:
                    recording = True
                    sequence_data = []
                    frame_count = 0
                    print(f"\n🔴 Recording started for sequence {sequences_saved + 1}...")

            # Stop recording and save
            elif key == ord('e') or key == ord('E'):
                if recording:
                    recording = False

                    if len(sequence_data) < SEQUENCE_LENGTH:
                        print(
                            f"\n⚠️  Sequence too short ({len(sequence_data)} frames). Need at least {SEQUENCE_LENGTH} frames.")
                        print("   Sequence discarded. Try recording for longer.")
                        sequence_data = []
                        frame_count = 0
                    else:
                        # Save the sequence
                        seq_dir = DATA_DIR / action / f"webcam_seq_{sequences_saved}"
                        ensure_dirs(seq_dir)

                        # Take exactly SEQUENCE_LENGTH frames evenly distributed
                        indices = np.linspace(0, len(sequence_data) - 1, SEQUENCE_LENGTH, dtype=int)
                        sampled_sequence = [sequence_data[i] for i in indices]

                        for j, kp in enumerate(sampled_sequence):
                            np.save(seq_dir / f"{j}.npy", kp)

                        sequences_saved += 1
                        print(
                            f"   ✅ Saved sequence {sequences_saved} with {len(sequence_data)} frames (sampled to {SEQUENCE_LENGTH})")

                        sequence_data = []
                        frame_count = 0

            # Quit
            elif key == ord('q') or key == ord('Q'):
                if recording:
                    print("\n⚠️  Recording in progress. Stopping and discarding current sequence...")
                break

    cap.release()
    cv2.destroyAllWindows()

    print(f"\n✅ Webcam collection complete: {sequences_saved} sequences saved for '{action}'")


def collect_data_from_video():
    """Enhanced data collection with better sampling"""
    ensure_dirs(DATA_DIR)

    if not VIDEO_PATH.exists():
        print(f"❌ Video directory not found: {VIDEO_PATH}")
        return

    video_files = list(VIDEO_PATH.glob("*.mp4")) + list(VIDEO_PATH.glob("*.avi"))

    if not video_files:
        print(f"❌ No video files found in {VIDEO_PATH}")
        return

    print(f"✅ Found {len(video_files)} video files")

    with mp_pose.Pose(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            model_complexity=1
    ) as pose:

        for video_idx, video_file in enumerate(video_files):
            print(f"\n🎥 Processing: {video_file.name}")

            # Determine action from filename
            action = None
            for act in ACTIONS:
                if act in video_file.name.lower():
                    action = act
                    break

            if not action:
                print(f"⚠️ Skipping {video_file.name} - no matching action in filename")
                continue

            cap = cv2.VideoCapture(str(video_file))

            if not cap.isOpened():
                print(f"❌ Could not open video: {video_file.name}")
                continue

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            sequence_data = []
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize for consistency
                frame = cv2.resize(frame, (640, 480))

                image, results = mediapipe_detection(frame, pose)

                if image is None:
                    continue

                draw_landmarks(image, results)
                keypoints = extract_keypoints(results)

                # Only save frames with detected pose
                if np.sum(keypoints) > 0:
                    sequence_data.append(keypoints)

                # Display progress
                label_text = action.replace("_", " ").title()
                cv2.putText(image, f"Processing: {label_text}", (15, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOURS[video_idx % len(COLOURS)], 2)
                cv2.putText(image, f"Frame: {frame_count + 1}/{total_frames}", (15, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow('Processing Video', image)
                frame_count += 1

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            cap.release()

            # Save sequences with better sampling
            sequences_saved = 0
            min_gap = SEQUENCE_LENGTH // 2

            for i in range(0, len(sequence_data) - SEQUENCE_LENGTH + 1, min_gap):
                sequence = sequence_data[i:i + SEQUENCE_LENGTH]

                if len(sequence) == SEQUENCE_LENGTH:
                    # Check sequence quality
                    valid_frames = sum(1 for seq in sequence if np.sum(seq) > 0)

                    if valid_frames >= SEQUENCE_LENGTH * 0.9:
                        seq_dir = DATA_DIR / action / f"{video_file.stem}_seq_{sequences_saved}"
                        ensure_dirs(seq_dir)

                        # Save original sequence
                        for j, keypoints in enumerate(sequence):
                            np.save(seq_dir / f"{j}.npy", keypoints)

                        # Add augmented version
                        if sequences_saved < 10:
                            aug_seq_dir = DATA_DIR / action / f"{video_file.stem}_aug_{sequences_saved}"
                            ensure_dirs(aug_seq_dir)

                            for j, keypoints in enumerate(sequence):
                                aug_keypoints = add_noise_augmentation(keypoints)
                                np.save(aug_seq_dir / f"{j}.npy", aug_keypoints)

                        sequences_saved += 1

            print(f"   ✅ Saved {sequences_saved} sequences from {video_file.name}")

    cv2.destroyAllWindows()

    # Print summary
    print(f"\n📈 Video Data Collection Summary:")
    for action in ACTIONS:
        action_dir = DATA_DIR / action
        if action_dir.exists():
            seq_count = len(list(action_dir.glob("*")))
            print(f"   {action}: {seq_count} sequences")


def display_menu():
    """Display main menu for data collection mode selection"""
    print("\n" + "=" * 60)
    print("🎯 EXERCISE FORM DATA COLLECTION SYSTEM")
    print("=" * 60)
    print("\nData Collection Modes:")
    print("  1. Collect from Videos")
    print("  2. Collect from Webcam")
    print("  3. Both (Videos first, then Webcam)")
    print("  4. Show Dataset Summary")
    print("  5. Exit")
    print("=" * 60)

    choice = input("\nSelect mode (1-5): ")
    return choice


def show_dataset_summary():
    """Display summary of collected data"""
    print("\n📊 DATASET SUMMARY")
    print("=" * 60)

    if not DATA_DIR.exists():
        print("❌ No data directory found")
        return

    total_sequences = 0
    for action in ACTIONS:
        action_dir = DATA_DIR / action
        if action_dir.exists():
            seq_dirs = list(action_dir.glob("*"))
            seq_count = len(seq_dirs)
            total_sequences += seq_count

            # Count webcam vs video sequences
            webcam_count = len([d for d in seq_dirs if 'webcam' in d.name])
            video_count = seq_count - webcam_count

            print(f"\n{action}:")
            print(f"  Total: {seq_count} sequences")
            print(f"  └─ Webcam: {webcam_count}")
            print(f"  └─ Video: {video_count}")
        else:
            print(f"\n{action}: 0 sequences")

    print(f"\n{'=' * 60}")
    print(f"TOTAL SEQUENCES: {total_sequences}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    while True:
        choice = display_menu()

        if choice == '1':
            collect_data_from_video()

        elif choice == '2':
            collect_data_from_webcam()

        elif choice == '3':
            print("\n📹 Starting with video collection...")
            collect_data_from_video()
            print("\n📷 Now starting webcam collection...")
            collect_data_from_webcam()

        elif choice == '4':
            show_dataset_summary()

        elif choice == '5':
            print("\n👋 Exiting... Goodbye!")
            break

        else:
            print("\n❌ Invalid choice. Please select 1-5.")

        input("\nPress Enter to continue...")

