# predict_folder.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import pandas as pd
import os
from pathlib import Path
from glob import glob

# 1. Load the model and define classes
model = load_model('apple_disease_mobilenet_final.h5')
class_names = [
    'Apple___Apple_scab', 
    'Apple___Black_rot', 
    'Apple___Cedar_apple_rust', 
    'Apple___healthy', 
    'Apple___Powdery_mildew'
]

# 2. Define the folder containing your test images
test_folder = "D:/apple_disease_model/dataset_split/test"  # <-- CHANGE THIS PATH
image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')

# 3. Get all image paths (including subfolders)
image_paths = glob(os.path.join(test_folder, "**", "*.*"), recursive=True)
image_paths = [p for p in image_paths if p.lower().endswith(image_extensions)]

print(f"Found {len(image_paths)} images to predict.")

# 4. Collect results
results = []

for i, img_path in enumerate(image_paths):
    print(f"\n--- Processing Image {i+1}/{len(image_paths)}: {os.path.basename(img_path)} ---")
    
    try:
        # Load and preprocess the image
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make prediction
        predictions = model.predict(img_array, verbose=0)
        predicted_index = np.argmax(predictions[0])
        confidence = np.max(predictions[0])

        predicted_class = class_names[predicted_index]

        # Save results
        results.append({
            "filename": os.path.basename(img_path),
            "true_label": Path(img_path).parent.name,  # from subfolder
            "predicted_label": predicted_class,
            "confidence": round(confidence * 100, 2)
        })

        # Print results to console
        print(f"Predicted Disease: {predicted_class}")
        print(f"Confidence: {confidence * 100:.2f}%")
        
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

# 5. Save results to CSV
df = pd.DataFrame(results)
df.to_csv("prediction_results.csv", index=False)

print("\n--- All images processed! ---")
print("Results saved to prediction_results.csv")
