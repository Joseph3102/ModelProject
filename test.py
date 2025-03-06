import tensorflow as tf
import tensorflow_io as tfio
import sounddevice as sd
from codrone_edu.drone import *

# Parameters for audio recording
duration = 5  # seconds
sample_rate = 44100  # Hz

# Record audio from the microphone
print("Recording...")
audio_data = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='float32')
sd.wait()  # Wait until the recording is finished
print("Recording complete")

# Convert the recorded audio to a TensorFlow tensor
audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
audio_tensor = tf.squeeze(audio_tensor, axis=[-1])

print("Audio tensor shape:", audio_tensor.shape)

# Load your trained model
model = tf.keras.models.load_model('path_to_your_model')

# Preprocess the audio tensor as required by your model
# This step will depend on how your model was trained
# For example, you might need to reshape the tensor, normalize it, etc.
# audio_tensor = preprocess_audio(audio_tensor)

# Predict the command
command = model.predict(tf.expand_dims(audio_tensor, axis=0))
command = tf.argmax(command, axis=1).numpy()[0]

# Map the predicted command to a string
command_map = {0: "takeoff", 1: "land"}  # Adjust this mapping based on your model's output
command_str = command_map.get(command, "unknown")

print("Recognized command:", command_str)

# Drone code
drone = Drone()
drone.pair()
print("Paired!")

if command_str == "takeoff":
    drone.takeoff()
    print("In the air!")
elif command_str == "land":
    drone.land()
    print("Landing")

drone.close()
print("Program complete")