# Iris-Recognition-on-UBIRIS-v2

![](image2.jpg)

## **Definition**

This project focuses on developing a robust iris recognition pipeline tailored for challenging conditions, using the **UBIRIS.v2 dataset**. The project addresses real-world challenges such as low-light environments, motion blur, and other non-ideal conditions by employing advanced image processing and deep learning techniques. The primary objective is to accurately detect, enhance, and classify iris images with high precision, even under suboptimal conditions.

### **Pipeline Overview**
1. **Synthetic Noise Addition**:  
   To simulate real-world challenges, noise is artificially introduced to the dataset. This includes:
   - **Motion Blur**: Created using kernel-based filtering to mimic camera movement.
   - **Low Light**: Simulated using gamma correction techniques.  
   These adjustments ensure the model is robust to variations in image quality.

2. **Data Augmentation**:  
   A series of augmentation techniques, such as flips, rotations, and brightness adjustments, are applied to the dataset. This step increases the dataset diversity and helps the model generalize better across unseen conditions.

3. **Image Enhancement**:  
   A deep learning-based enhancement model is used to restore image quality by reducing noise and emphasizing iris features. This involves:
   - **DenseNet-121 Encoder**: A pre-trained convolutional neural network that extracts critical features from images.
   - **Custom Decoder**: Uses transposed convolutions to reconstruct enhanced images.  
   The output is a set of denoised and enhanced iris images ready for classification.

4. **Iris Detection and Segmentation**:  
   The pipeline uses traditional computer vision techniques:
   - **Haar Cascade**: For detecting eye regions.
   - **Hough Transform**: For precise iris segmentation within detected eyes.  
   This step isolates the iris region, which is crucial for accurate classification.

5. **Classification**:  
   Using **ResNet-50**, a state-of-the-art CNN pretrained on ImageNet, features are extracted from the enhanced iris images. A custom classification model built on top of these features distinguishes between 214 unique classes in the dataset.

### **Dataset**  
The **UBIRIS.v2 dataset** contains **1,214 images** representing **214 unique classes**, collected under diverse and challenging conditions. This dataset is widely recognized for testing iris recognition models in real-world scenarios.

---

## **Key Results**

- **Overall Accuracy**: **63.69%**
- **Macro F1 Score**: **0.6191**
- **Kappa Statistic**: **0.63615** (Substantial Agreement)
- **False Positive Rate**: **0.07%**
- **Hamming Loss**: **36.31%**
- **Mutual Information**: **8.00414**

These metrics demonstrate the effectiveness of the pipeline in recognizing irises under challenging conditions, showcasing a balance between precision, recall, and robustness.

---

## **Files**

- **`IrisEnhancement_Ubiris.ipynb`**:
  - Implements synthetic noise simulation to create motion blur and low-light conditions.
  - Performs data augmentation to increase dataset size and diversity.
  - Applies a deep learning-based image enhancement model to denoise images.

- **`IrisRecognition_Ubiris.ipynb`**:
  - Detects eyes and irises using Haar Cascade and Hough Circles.
  - Segments and processes iris regions for classification.
  - Extracts features using ResNet-50 and classifies images into 214 classes.

---

## **How to Run**

### **Dependencies**
Install the following libraries:
```plaintext
numpy
opencv-python
matplotlib
seaborn
torch
torchvision
tensorflow
keras
scikit-learn
tqdm
