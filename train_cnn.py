import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Dataset Path
train_data_path = "data/preprocessed/"

# Data Augmentation
datagen = ImageDataGenerator(validation_split=0.2, rescale=1./255)

train_generator = datagen.flow_from_directory(train_data_path, target_size=(32, 32),
                                             color_mode="grayscale", batch_size=32,
                                             class_mode="categorical", subset="training")

val_generator = datagen.flow_from_directory(train_data_path, target_size=(32, 32),
                                           color_mode="grayscale", batch_size=32,
                                           class_mode="categorical", subset="validation")

print(f"Total training images: {train_generator.samples}")
print(f"Total validation images: {val_generator.samples}")


# CNN Model
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(train_generator.class_indices), activation='softmax')  # Output layer (one node per character)
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# Train Model
model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save Model
model.save("cnn_character_recognition.h5")
print("✅ Model Training Complete. Model saved as cnn_character_recognition.h5")
