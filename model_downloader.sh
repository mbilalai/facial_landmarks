#!/bin/bash

# Create a directory for the models
mkdir -p mediapipe_models

# Download face_landmarker_v2_with_blendshapes.task
curl -L "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task" \
    -o "mediapipe_models/face_landmarker_v2_with_blendshapes.task"

# Download hand_landmarker.task
curl -L "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task" \
    -o "mediapipe_models/hand_landmarker.task"

echo "Download completed. Models saved in mediapipe_models directory."
