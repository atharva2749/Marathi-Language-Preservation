import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocessing import preprocess_image  # Ensure this import works
import os
import random
import sys  # ✅ To force exit after one run

# ✅ Load the trained CNN model
model = load_model("cnn_character_recognition.h5")

# ✅ Load class labels
class_indices = {
    0: "अ", 1: "आ", 2: "इ", 3: "ई", 4: "उ", 5: "ऊ", 6: "ऋ", 7: "ए", 8: "ऐ", 9: "ओ",
    10: "औ", 11: "अं", 12: "अः", 13: "क", 14: "ख", 15: "ग", 16: "घ", 17: "ङ",
    18: "च", 19: "छ", 20: "ज", 21: "झ", 22: "ञ", 23: "ट", 24: "ठ", 25: "ड", 26: "ढ",
    27: "ण", 28: "त", 29: "थ", 30: "द", 31: "ध", 32: "न", 33: "प", 34: "फ", 35: "ब",
    36: "भ", 37: "म", 38: "य", 39: "र", 40: "ल", 41: "व", 42: "श", 43: "ष", 44: "स",
    45: "ह", 46: "ळ", 47: "क्ष", 48: "ज्ञ", 49: "०", 50: "१", 51: "२", 52: "३",
    53: "४", 54: "५", 55: "६", 56: "७", 57: "८", 58: "९"
}

# ✅ Pick one random image (Only once)
dataset_path = "data/preprocessed/"
category = random.choice(["consonants", "vowels", "numerals"])
category_path = os.path.join(dataset_path, category)

subfolders = [f for f in os.listdir(category_path) if os.path.isdir(os.path.join(category_path, f))]
selected_subfolder = random.choice(subfolders)
subfolder_path = os.path.join(category_path, selected_subfolder)

images = [f for f in os.listdir(subfolder_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
test_image_path = os.path.join(subfolder_path, random.choice(images))

# 🔍 Debugging: Print selected image path
print(f"🖼 Selected Image: {test_image_path}")

# ✅ Load the original image
original_img = cv2.imread(test_image_path)

# ✅ Preprocess the image for prediction
input_image = preprocess_image(test_image_path)

if input_image is not None:
    input_image = np.expand_dims(input_image, axis=0)  # Add batch dimension

    print("⚙️ Running Prediction...")

    # ✅ Make a prediction
    prediction = model.predict(input_image)

    print("✅ Prediction Completed!")

    # ✅ Get the class with the highest probability
    predicted_class = np.argmax(prediction)
    predicted_character = class_indices.get(predicted_class, "Unknown")

    # ✅ Print the predicted character
    print(f"🔍 Predicted Character: {predicted_character}")

    # ✅ Draw a green bounding box around the detected character
    height, width, _ = original_img.shape
    start_point = (5, 5)
    end_point = (width - 5, height - 5)
    color = (0, 255, 0)  # Green
    thickness = 3
    cv2.rectangle(original_img, start_point, end_point, color, thickness)

    # ✅ Put predicted character on the image
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(original_img, predicted_character, (10, 40), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # ✅ Show the image with prediction
    cv2.imshow("Predicted Character", original_img)
    cv2.waitKey(0)  # Wait for key press
    cv2.destroyAllWindows()  # Close the image window

    # ✅ Exit script after displaying the result
    sys.exit()

else:
    print("⚠️ Error: Unable to process image.")
    sys.exit()
