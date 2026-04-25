# Apple Disease Model
# 🍎 Apple Leaf Disease Detection using Deep Learning

## 📌 Project Overview

This project implements a **deep learning–based image classification system** to detect and classify **apple leaf diseases** using **transfer learning with MobileNetV2**. The aim is to assist farmers and researchers in early disease detection, improving crop yield and reducing losses through timely intervention.

Apple crops are prone to various leaf diseases that are difficult to detect manually at early stages. By leveraging **Convolutional Neural Networks (CNNs)**, this project automates the identification of apple leaf diseases from images.

You can download the models here: 
Best_model.h5: https://drive.google.com/file/d/1ssd3b9srG3G4QCJi8rFXwWMBUcOVw9mE/view?usp=sharing


apple_disease_mobilenet_final.h5 : https://drive.google.com/file/d/1cbzbYks59ZGXczzVIcgn8_dRkjaiJobX/view?usp=sharing

---

## 🎯 Objectives

* Automate apple leaf disease detection using AI
* Reduce dependency on manual inspection
* Improve accuracy using transfer learning
* Provide a scalable solution for smart agriculture

---

## 🧠 Model & Methodology

* **Base Model:** MobileNetV2 (pre-trained on ImageNet)
* **Approach:** Transfer Learning
* **Input Image Size:** 224 × 224
* **Optimizer:** Adam
* **Loss Function:** Categorical Crossentropy
* **Batch Size:** 32
* **Epochs:** 15

The MobileNetV2 architecture is chosen for its efficiency and performance, making it suitable for real-world agricultural applications.

---

## 🏷️ Disease Classes

The model classifies apple leaf images into the following categories:

* Apple Scab
* Black Rot
* Cedar Apple Rust
* Healthy Apple Leaf
* Powdery Mildew

---

## 📂 Project Structure

```
apple-disease-model-master/
│
├── apple.py                          # Model training and evaluation
├── predict.py                        # Disease prediction script
├── split.py                          # Dataset splitting utility
├── apple_disease_mobilenet_final.h5  # Final trained model
├── best_model.h5                     # Best saved model during training
├── confusion_matrix.png              # Model performance visualization
├── sample_predictions.png            # Sample prediction output
├── prediction_results.csv            # CSV file containing predictions
├── README.md                         # Project documentation
```

---

## 📊 Dataset Description

* Dataset consists of apple leaf images organized into class-wise folders
* Split into:

  * Training set
  * Validation set
  * Test set

Example structure:

```
dataset/
├── train/
├── val/
└── test/
```

---

## ▶️ How to Run the Project

### 1️⃣ Install Required Libraries

```bash
pip install tensorflow numpy matplotlib seaborn pandas scikit-learn
```

### 2️⃣ Train the Model

```bash
python apple.py
```

### 3️⃣ Predict Disease from Images

```bash
python predict.py
```

Prediction results will be saved as:

```
prediction_results.csv
```

---

## 📈 Results & Evaluation

* Achieves high classification accuracy using MobileNetV2
* Confusion matrix and classification report generated
* Performs well on unseen test images

---

## 🌱 Applications

* Precision agriculture
* Smart orchard monitoring
* Early disease detection systems
* AI-assisted decision making for farmers

---

## 🔮 Future Scope

* Deploy as a web or mobile application
* Integrate drone or multispectral imagery
* Expand dataset for improved robustness
* Real-time disease detection

---

## 👨‍💻 Author

**Sarmad Fayaz**
B.Tech (Artificial Intelligence)
AI in Agriculture Enthusiast

---

## 📜 License

This project is intended for academic and research purposes.
