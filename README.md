# AI-Powered Exercise Form Classification and Correction System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

*An intelligent real-time workout assistant that analyzes exercise form using deep learning and computer vision*

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Demo](#demo)
- [System Architecture](#system-architecture)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Dataset](#dataset)
- [Training](#training)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project implements an **AI-powered exercise form classifier and corrector** that provides real-time feedback on workout form quality. Using **Bidirectional LSTM networks** with **attention mechanisms** and **MediaPipe pose estimation**, the system analyzes two exercises:

- 🏋️ **Bicep Curls** (Correct/Incorrect form)
- 💪 **Shoulder Press** (Correct/Incorrect form)

The application detects improper form, counts repetitions, provides coaching tips, and tracks workout statistics through an intuitive GUI.

---

## ✨ Features

### Real-Time Analysis
- **Live pose detection** using MediaPipe with 33 body landmarks
- **Frame-by-frame exercise classification** with confidence thresholds
- **Instant feedback** on form correctness during workout sessions

### Intelligent Form Correction
- **Biomechanical analysis** for specific form issues:
  - Elbow alignment and stability detection
  - Joint angle calculations (shoulder, elbow, wrist)
  - Spine reference tracking
  - Movement symmetry assessment
- **Actionable coaching tips** displayed in real-time

### Workout Tracking
- **Automatic rep counting** with up/down motion detection
- **Good form vs. total reps** tracking
- **Accuracy percentage** calculation per exercise
- **Common issues logging** for post-workout review

### User Interface
- **Modern desktop GUI** built with PySide6/Qt6
- **Three-screen workflow**:
  1. Welcome screen with feature overview
  2. Live workout session with video feed and dashboard
  3. Session summary with detailed statistics
- **Visual feedback** with live camera feed and pose landmarks
- **Responsive design** with real-time progress bars

### Data Collection & Training
- **Flexible data collection** from webcam or video files
- **Automated sequence extraction** (30 frames per sequence)
- **Data augmentation** with noise injection
- **Comprehensive training visualizations**:
  - Training/validation curves
  - Confusion matrices
  - Per-class metrics
  - Learning rate schedules

---

## 🎥 Demo

### Application Flow

