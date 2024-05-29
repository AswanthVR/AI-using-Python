import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

model_path = 'brain_tumor_detector.h5'

model = load_model(model_path)

test_dir = 'dataset/test'

# Prepare test data
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=1,
    class_mode='binary',
    shuffle=False
)

# Predict on test images
predictions = model.predict(test_generator)

# Convert predicted probabilities to binary labels
predicted_labels = np.round(predictions).flatten()

# Get filenames and true labels
filenames = test_generator.filenames
true_labels = test_generator.classes

# Print predictions
for filename, predicted_label, true_label in zip(filenames, predicted_labels, true_labels):
    print(f"Filename: {filename}, Predicted: {'Tumor' if predicted_label > 0.5 else 'Normal'}, True: {'Tumor' if true_label == 1 else 'Normal'}")
