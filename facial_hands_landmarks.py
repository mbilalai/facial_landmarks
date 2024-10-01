
import os
import cv2
import numpy as np
import mediapipe as mp
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(image, face_detection_result, hand_detection_result):
    black_image = np.zeros(image.shape, dtype=np.uint8)

    # draw face landmarks
    if face_detection_result.face_landmarks:
        for face_landmarks in face_detection_result.face_landmarks:
            face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            face_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
            ])

            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style())
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style())
            solutions.drawing_utils.draw_landmarks(
                image=black_image,
                landmark_list=face_landmarks_proto,
                connections=mp.solutions.face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_iris_connections_style())

    # draw hand landmarks
    if hand_detection_result.hand_landmarks:
        for hand_landmarks in hand_detection_result.hand_landmarks:
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                black_image,
                hand_landmarks_proto,
                mp.solutions.hands.HAND_CONNECTIONS,
                mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                mp.solutions.drawing_styles.get_default_hand_connections_style())

    return black_image

# create facelandmarker and handlandmarker with gpu support
face_model_path = os.path.join('mediapipe_models', 'face_landmarker.task')
hand_model_path = os.path.join('mediapipe_models', 'hand_landmarker.task')

base_options = python.BaseOptions(model_asset_path=face_model_path, delegate=python.BaseOptions.Delegate.GPU)
face_options = vision.FaceLandmarkerOptions(base_options=base_options,
                                            output_face_blendshapes=True,
                                            output_facial_transformation_matrixes=True,
                                            num_faces=1)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

base_options = python.BaseOptions(model_asset_path=hand_model_path, delegate=python.BaseOptions.Delegate.GPU)
hand_options = vision.HandLandmarkerOptions(base_options=base_options,
                                            num_hands=2)
hand_detector = vision.HandLandmarker.create_from_options(hand_options)

# initialize webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # convert to rgb
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # detect face and hand landmarks
    face_detection_result = face_detector.detect(mp_image)
    hand_detection_result = hand_detector.detect(mp_image)

    # draw landmarks on black image
    annotated_image = draw_landmarks_on_image(rgb_frame, face_detection_result, hand_detection_result)
    # resize the image to make it bigger
    scale_factor = 1.5  # adjust this value to make the window larger or smaller
    width = int(annotated_image.shape[1] * scale_factor)
    height = int(annotated_image.shape[0] * scale_factor)
    resized_image = cv2.resize(annotated_image, (width, height), interpolation=cv2.INTER_LINEAR)

    # show only the face mesh and hand landmarks
    cv2.imshow('Face Mesh and Hands', resized_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


