import os, shutil
from sklearn.model_selection import train_test_split

dataset_dir = "D:/apple_disease_model/dataset"   # your original folder
output_dir = "D:/apple_disease_model/dataset_split"

# ratios
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

os.makedirs(output_dir, exist_ok=True)

for class_name in os.listdir(dataset_dir):
    class_dir = os.path.join(dataset_dir, class_name)
    if not os.path.isdir(class_dir):
        continue
    
    images = os.listdir(class_dir)
    train_imgs, temp_imgs = train_test_split(images, test_size=(1-train_ratio), random_state=42)
    val_imgs, test_imgs = train_test_split(temp_imgs, test_size=test_ratio/(val_ratio+test_ratio), random_state=42)
    
    for split, split_imgs in zip(["train", "val", "test"], [train_imgs, val_imgs, test_imgs]):
        split_dir = os.path.join(output_dir, split, class_name)
        os.makedirs(split_dir, exist_ok=True)
        for img in split_imgs:
            shutil.copy(os.path.join(class_dir, img), os.path.join(split_dir, img))

print("✅ Dataset split into train/val/test successfully!")
