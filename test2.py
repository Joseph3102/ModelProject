import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np
from codrone_edu.drone import *

# Load the MoveNet model for pose detection
model = hub.load('https://tfhub.dev/google/movenet/singlepose/lightning/4')

# Get the inference function from the model
infer = model.signatures['serving_default']

# Function to preprocess the frame for MoveNet
def preprocess_frame(frame):
    # Convert the image to RGB and resize it to (192, 192)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 192))
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = tf.cast(img, dtype=tf.int32)
    return img

# Function to draw pose keypoints on the frame
def draw_pose_keypoints(frame, keypoints, confidence_threshold=0.3):
    # Each keypoint is an array [y, x, confidence]
    for kp in keypoints:
        y, x, confidence = kp[0], kp[1], kp[2]  # Unpack y, x, and confidence
        if confidence > confidence_threshold:
            cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 0), -1)
    return frame

# Function to recognize yoga pose based on keypoints
def recognize_yoga_pose(keypoints):
    # Example simple logic based on keypoints (you can refine this for more accuracy)
    # Detect if the pose corresponds to specific poses like Tadasana or Downward Dog
    if keypoints[0][1] > keypoints[1][1] and keypoints[2][1] < keypoints[3][1]:
        return "Tadasana"
    elif keypoints[1][1] < keypoints[2][1] and keypoints[3][1] < keypoints[4][1]:
        return "Downward Dog"
    else:
        return "Unknown Pose"

# Drone code
drone = Drone()
drone.pair()
print("Paired!")

# Initialize the webcam or camera for capturing frames
cap = cv2.VideoCapture(0)  # Use the first available camera
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Define yoga pose classes (adjust based on your model's output logic)
pose_map = {
    "Tadasana": 0,  # Adjust the action logic as needed
    "Downward Dog": 1
}

while True:
    # Capture a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    # Preprocess the frame for the model
    input_frame = preprocess_frame(frame)

    # Run pose detection using the model
    outputs = infer(input_frame)

    # Extract keypoints from the model's output (assuming keypoints are in 'output_0')
    keypoints = outputs['output_0'][0].numpy()

    # The keypoints are structured as [y, x, confidence] for each joint, and we need to reshape
    keypoints = keypoints.reshape((-1, 3))  # Reshape to (num_keypoints, 3)

    # Draw the keypoints on the frame
    frame_with_keypoints = draw_pose_keypoints(frame, keypoints)

    # Recognize the yoga pose based on the keypoints
    pose_name = recognize_yoga_pose(keypoints)

    print(f"Detected Pose: {pose_name}")

    # Perform drone actions based on detected pose
    if pose_name == "Tadasana":
        drone.takeoff()
        print("Taking off!")
    elif pose_name == "Downward Dog":
        drone.land()
        print("Landing!")

    # Display the frame with pose label
    cv2.putText(frame_with_keypoints, pose_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.imshow("Yoga Pose Detection", frame_with_keypoints)

    # Press 'q' to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()

# Close the drone connection
drone.close()
print("Program complete")
