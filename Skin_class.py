import cv2
import tensorflow as tf
import numpy as np
from keras.models import load_model
from mtcnn import MTCNN

model = load_model('model.h5')  # Replace with your model path

# List of class names
classes = ['Fair_Light', 'Medium_Tan', 'Dark_Deep']

descriptive_labels = {
    'Fair_Light': 'Fair / Light',
    'Medium_Tan': 'Medium / Tan',
    'Dark_Deep': 'Dark / Deep'
}

mtcnn = MTCNN()

def skin_tone(input_face):
    # image = np.array(bytearray(input_face), dtype=np.uint8)
    image = cv2.imread(input_face)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(image)
    if len(faces) > 0:
        largest_face = max(faces, key=lambda f: f['box'][2] * f['box'][3])
        x, y, w, h = largest_face['box']
        detected_face = image[y:y+h, x:x+w]
        
        # Resize the detected face to the desired input shape
        detected_face = cv2.resize(detected_face, (120, 90))
        
        # Preprocess the detected face for classification
        detected_face = tf.keras.applications.mobilenet_v2.preprocess_input(detected_face[np.newaxis, ...])
        
        # Predict the class of the face
        predictions = model.predict(detected_face)
        predicted_class_idx = np.argmax(predictions)
        predicted_class = classes[predicted_class_idx]
        descriptive_label = descriptive_labels[predicted_class]
        return descriptive_label