import cv2
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from codrone_edu.drone import *

# Load MoveNet model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
movenet = model.signatures['serving_default']

# Initialize and pair the drone
drone = Drone()
drone.pair()
print("Paired!")

drone_flying = False  # Track drone state

def run_inference(frame):
    """Preprocess frame and run MoveNet inference."""
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = cv2.resize(img, (192, 192))  # Resize to model input size
    img = np.expand_dims(img, axis=0).astype(np.int32)  # Add batch dim
    outputs = movenet(tf.convert_to_tensor(img))  # Run inference
    keypoints = outputs['output_0'].numpy()[0, 0, :, :]  # Extract keypoints (17, 3)
    return keypoints

def is_right_arm_raised(keypoints):
    """Check if right wrist (10) is above right shoulder (6)."""
    right_shoulder = keypoints[6]
    right_wrist = keypoints[10]
    return right_wrist[2] > 0.5 and right_shoulder[2] > 0.5 and right_wrist[1] < right_shoulder[1]

def is_left_arm_raised(keypoints):
    """Check if left wrist (9) is above left shoulder (5) and not too close to the right wrist."""
    left_shoulder = keypoints[5]
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    
    # Adjust confidence threshold to see if it improves detection
    return (
        left_wrist[2] > 0.4 and left_shoulder[2] > 0.4 and left_wrist[1] < left_shoulder[1]
        and abs(left_wrist[0] - right_wrist[0]) > 0.05  # Loosen the wrist distance threshold
    )

def is_clapping(keypoints):
    """Check if wrists are close together, indicating a clap."""
    left_wrist = keypoints[9]
    right_wrist = keypoints[10]
    return left_wrist[2] > 0.5 and right_wrist[2] > 0.5 and abs(left_wrist[0] - right_wrist[0]) < 0.05

def draw_keypoints(frame, keypoints):
    """Draw keypoints on the frame."""
    height, width, _ = frame.shape
    for kp in keypoints:
        x, y, confidence = int(kp[0] * width), int(kp[1] * height), kp[2]
        if confidence > 0.5:
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return frame

# Start webcam capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    keypoints = run_inference(frame)

    if is_right_arm_raised(keypoints):
        print("Takeoff")
        drone.takeoff()
        drone_flying = True

    elif is_left_arm_raised(keypoints):
        print("Landing")
        drone.land()
        drone_flying = False

    elif is_clapping(keypoints) and drone_flying:
        print("Flip!")
        drone.flip()

    frame = draw_keypoints(frame, keypoints)  # Draw keypoints
    cv2.imshow("Webcam", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
