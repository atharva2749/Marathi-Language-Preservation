import cv2
import numpy as np
import os

# Load dataset paths
dataset_path = "data/extracted_data/"
output_path = "data/preprocessed/"

if not os.path.exists(output_path):
    os.makedirs(output_path)

# ✅ Preprocessing function
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
    if img is None:
        print(f"⚠️ Warning: Unable to read image {image_path}")
        return None
    img = cv2.resize(img, (32, 32))  # Resize to standard size
    img = cv2.GaussianBlur(img, (3,3), 0)  # Noise removal
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)  # Binarization
    return img

# ✅ This function runs only when explicitly called
def run_preprocessing():
    for char in os.listdir(dataset_path):
        char_path = os.path.join(dataset_path, char)

        if not os.path.isdir(char_path):  # Skip if not a directory
            continue

        for subfolder in os.listdir(char_path):  # Iterate over subfolders (e.g., "1", "2", etc.)
            subfolder_path = os.path.join(char_path, subfolder)

            if not os.path.isdir(subfolder_path):  # Skip if not a directory
                continue

            save_subfolder_path = os.path.join(output_path, char, subfolder)
            if not os.path.exists(save_subfolder_path):
                os.makedirs(save_subfolder_path)

            for image in os.listdir(subfolder_path):
                img_path = os.path.join(subfolder_path, image)
                processed_img = preprocess_image(img_path)

                if processed_img is not None:
                    save_path = os.path.join(save_subfolder_path, image)
                    cv2.imwrite(save_path, processed_img)

    print("✅ Preprocessing Complete. Processed images saved in:", output_path)

# ✅ Run preprocessing only if script is executed directly
if __name__ == "__main__":
    run_preprocessing()
