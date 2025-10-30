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

