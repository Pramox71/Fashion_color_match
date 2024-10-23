import os
from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import torch.nn as nn
import numpy as np
from collections import Counter
import webcolors
import matplotlib.pyplot as plt
from keras.preprocessing import image  # Added for compatibility with the original skin_tone classification model
from Skin_class import skin_tone  # Import the skin tone classification function
import google.generativeai as genai

GOOGLE_API_KEY = "AIzaSyAakCO0wQdenLCBv7lZphEWoAjounEnFGw"
genai.configure(api_key=GOOGLE_API_KEY)

# Load model and processor

processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")

# Image URL
# url = "https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80"
url = r"E:\Data_C\Documents\KULIAH\Codingan\Kerja\Kecocokan Pakaian\Upload\est_detek.jpeg"
# image_pil = Image.open(requests.get(url, stream=True).raw)
image_pil = Image.open(url)

# Preprocess image
inputs = processor(images=image_pil, return_tensors="pt")

# Model inference
outputs = model(**inputs)
logits = outputs.logits.cpu()

# Upsample logits to the image size
upsampled_logits = nn.functional.interpolate(
    logits,
    size=image_pil.size[::-1],
    mode="bilinear",
    align_corners=False,
)

# Get segmentation map
pred_seg = upsampled_logits.argmax(dim=1)[0]

# Class indices (example, need to check with the actual class index for your model)
face_class_idx = 11  # Replace with actual class index for face
shirt_class_idx = 4  # Replace with actual class index for shirt
pants_class_idx = 6  # Replace with actual class index for pants

# Define folder paths
folders = {
    face_class_idx: "muka",
    shirt_class_idx: "shirt",
    pants_class_idx: "pants",
}

skin_color = ""
shirt_color = ""
pants_color = ""
# Create folders if they do not exist
for folder in folders.values():
    if not os.path.exists(folder):
        os.makedirs(folder)

# Convert PIL image to NumPy array
image_np = np.array(image_pil)

# Function to get the dominant color from the segmented image
def get_dominant_color(segmented_image):
    # Reshape the image to be a list of pixels
    pixels = segmented_image.reshape(-1, 3)
    # Remove black pixels (background)
    pixels = pixels[~np.all(pixels == 0, axis=1)]
    
    if len(pixels) == 0:
        return (0, 0, 0)  # Return black if no pixels are left
    
    # Count the frequency of each color
    color_counts = Counter(map(tuple, pixels))
    # Get the most common color
    dominant_color = color_counts.most_common(1)[0][0]
    return dominant_color

# Function to convert RGB to color name
def rgb_to_color_name(rgb):
    try:
        color_name = webcolors.rgb_to_name(rgb)
    except ValueError:
        # If no exact match, find the closest color
        closest_name = min(webcolors.CSS3_HEX_TO_NAMES.keys(), 
                            key=lambda name: np.linalg.norm(np.array(webcolors.hex_to_rgb(name)) - np.array(rgb)))
        color_name = webcolors.CSS3_HEX_TO_NAMES[closest_name]
    
    return color_name

# Function to find the bounding box of the face region in the segmentation mask
def get_face_bounding_box(mask):
    # Get the coordinates of pixels belonging to the face
    face_pixels = np.argwhere(mask == True)
    
    # If no face is detected, return None
    if len(face_pixels) == 0:
        return None
    
    # Get the bounding box of the face
    y_min, x_min = face_pixels.min(axis=0)
    y_max, x_max = face_pixels.max(axis=0)
    
    return x_min, x_max, y_min, y_max

# Function to crop face in 1:1 ratio based on bounding box from segmentation
def crop_face_with_bbox(image_pil, mask):
    # Get the bounding box of the face
    bounding_box = get_face_bounding_box(mask)
    
    if bounding_box is None:
        print("No face detected.")
        return None

    x_min, x_max, y_min, y_max = bounding_box
    
    # Calculate the width and height of the bounding box
    width = x_max - x_min
    height = y_max - y_min
    
    # Make the bounding box square (1:1 aspect ratio)
    side_length = max(width, height)
    
    # Adjust the box to be centered and square
    x_center = (x_min + x_max) // 2
    y_center = (y_min + y_max) // 2
    
    x_min = max(0, x_center - side_length // 2)
    x_max = min(image_pil.size[0], x_center + side_length // 2)
    
    y_min = max(0, y_center - side_length // 2)
    y_max = min(image_pil.size[1], y_center + side_length // 2)
    
    # Crop the image to the bounding box
    cropped_face = image_pil.crop((x_min, y_min, x_max, y_max))
    
    return cropped_face

# Function to save extracted class segment and calculate dominant color
def save_segmented_class(class_idx, class_name):
    global skin_color, shirt_color, pants_color
    if class_name == "face":
        # Get mask for face
        mask = (pred_seg == class_idx).numpy()
        
        # Crop face using bounding box
        cropped_face = crop_face_with_bbox(image_pil, mask)
        if cropped_face is not None:
            output_path = os.path.join(folders[class_idx], f"{class_name}_cropped.png")
            cropped_face.save(output_path)
            print(f"Saved cropped {class_name} to {output_path}")

            # Call the skin tone classification function on the cropped image
            predicted_skin_tone = skin_tone(output_path)  # Pass the cropped image path to the skin_tone function
            skin_color = predicted_skin_tone
            print(f'Predicted Skin Tone Class: {predicted_skin_tone}')
    else:
        # Perform segmentation for other classes (shirt and pants)
        mask = (pred_seg == class_idx).numpy()  # Create mask for the class
        segmented_image = np.zeros_like(image_np)  # Create empty array for segmented image

        # Apply the mask to the image, copying only the areas that belong to the class
        segmented_image[mask] = image_np[mask]

        # Convert the segmented image back to a PIL Image
        segmented_image_pil = Image.fromarray(segmented_image)

        # Save the segmented image
        output_path = os.path.join(folders[class_idx], f"{class_name}.png")
        segmented_image_pil.save(output_path)
        print(f"Saved {class_name} segment to {output_path}")

        # Calculate and display the dominant color
        dominant_color = get_dominant_color(segmented_image)
        color_name = rgb_to_color_name(dominant_color)
        if class_name == "shirt":
            shirt_color = color_name
        elif class_name == "pants":
            pants_color = color_name
        # Convert RGB to color name
        print(f"The dominant color for {class_name} is: {color_name}")

# Save each segment (face, shirt, pants) to corresponding folders
save_segmented_class(face_class_idx, "face")
save_segmented_class(shirt_class_idx, "shirt")
save_segmented_class(pants_class_idx, "pants")
# Create the prompt
myfile = genai.upload_file(url)
prompt = f"I have a {skin_color} skin tone, a {shirt_color} shirt, and {pants_color} pants. Can you tell me how well these colors match together? Please provide a percentage match and, if it's below 60%, recommend better colors."


model_ai = genai.GenerativeModel('gemini-1.5-flash')

response = model_ai.generate_content([myfile, "\n\n",prompt])
print(response.text)