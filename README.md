# Deep Learning Wafer Defect Detection Overcomes Limited Label Data  
## Semi-Supervised Learning Techniques with FixMatch and Autoencoder

> **Note:** This project is developed as an IPython Notebook to enhance its visibility and ease of use.  
> - For a detailed explanation of the research paper, please refer to **WDC_Eng.pdf**.  
> - To review the code implementation, please check **WDC_ENG.ipynb**.

### Seongjin Park  
University of Wisconsin-Madison  
Computer Science & Data Science  
[seongjinpark99@gmail.com](mailto:seongjinpark99@gmail.com)  
[GitHub Repository](https://github.com/Seongjin74/Wafer-Defect-Classification)

---

## Table of Contents
- [Introduction](#introduction)
- [Related Work](#related-work)
- [Proposed Methodology](#proposed-methodology)
- [Experiments](#experiments)
  - [Dataset](#dataset)
  - [Deep Learning Model](#deep-learning-model)
  - [Supervised Learning](#supervised-learning)
  - [Semi-Supervised Learning](#semi-supervised-learning)
- [Experimental Evaluation](#experimental-evaluation)
- [Conclusion and Future Work](#conclusion-and-future-work)
- [References](#references)

---

## Introduction

With the rapid development of deep learning technology, tasks traditionally performed by humans—such as classification and prediction—are increasingly automated. This project tackles wafer defect detection using semi-supervised learning techniques that make efficient use of limited labeled data. Our approach integrates the FixMatch algorithm with an autoencoder-based augmentation module to reduce labeling costs while maintaining high detection accuracy in semiconductor manufacturing processes.

---

## Related Work

Recent advances in semi-supervised learning have made significant strides in leveraging unlabeled data to boost model performance. The **FixMatch algorithm** is at the core of our approach. FixMatch uniquely combines pseudo-labeling with consistency regularization by generating pseudo-labels from weakly augmented images and then applying them to strongly augmented versions. This method has proven effective in scenarios with limited labeled data, as it uses a confidence threshold (typically 0.95) to ensure only reliable pseudo-labels are used for training.

In the realm of wafer defect detection:
- **CNN-based approaches** have been widely explored to improve defect inspection rates compared to traditional methods.
- **Autoencoder techniques** are utilized to extract robust latent features, which significantly enhance data augmentation by reconstructing images with subtle defect patterns.
- **FixMatch in Wafer Defect Classification:** Our work specifically applies FixMatch to the wafer defect detection problem, combining its pseudo-labeling strategy with autoencoder-driven strong augmentation. This integration not only improves the robustness of the model under semi-supervised settings but also directly addresses the challenges of limited labeled data in semiconductor manufacturing.

These combined techniques provide a strong foundation for our system, setting it apart from earlier approaches that relied solely on either CNNs or traditional semi-supervised methods without the additional augmentation benefits provided by autoencoders.

---

## Proposed Methodology

Our approach consists of several key components:

- **Data Preprocessing:**  
  Wafer map data is converted into normalized 26×26 three-channel images using a custom `process_wafer_map` function. This function applies a predefined colormap to emphasize defect patterns.

- **Data Augmentation:**  
  Two strategies are combined:
  - *Weak Augmentation:* Resizing and minor rotations to preserve the core image characteristics.
  - *Strong Augmentation:* Aggressive geometric transformations using an autoencoder that:
    - Encodes the input image into a latent representation.
    - Adds a small amount of noise to the latent space.
    - Reconstructs the image via a decoder.
    - Applies additional random rotations (90-degree increments) to generate diverse images.

- **Semi-Supervised Learning with FixMatch:**  
  Pseudo-labels are generated from weakly augmented images when the model’s prediction confidence exceeds a predefined threshold. These labels are then enforced on the strongly augmented images, ensuring the model learns consistency even with a mix of labeled and unlabeled data.

- **Model Architecture:**  
  A simple yet efficient CNN, termed **SimpleCNN**, is used. It consists of:
  - Three convolutional layers
  - Two max pooling layers
  - Fully connected layers for final classification

---

## Experiments

### Dataset

We use the WM-811K wafer map dataset, which contains 811,457 wafer maps collected from semiconductor manufacturing processes. The dataset includes various defect types such as Center, Donut, Edge-Loc, Edge-Ring, Loc, Random, Scratch, Near-full, and none.  
- **Preprocessing:** Images are converted into normalized three-channel formats.  
- **Dataset Split:**  
  - *Supervised Learning:* Up to 100 training and 50 test samples per class.
  - *Semi-Supervised Learning:* An additional 10,000 unlabeled samples supplement 100 labeled samples per class.

### Deep Learning Model

**SimpleCNN** is employed with the following specifications:
- **Architecture:**  
  - Three convolutional layers for feature extraction.
  - Two max pooling layers to reduce spatial dimensions.
  - Fully connected layers for classification.
- **Training Details:**  
  - Optimizer: Adam (learning rate = 1e-3)
  - Loss Function: CrossEntropyLoss
  - Learning Rate Scheduler: Reduces the learning rate by half every 10 epochs
  - Total Epochs: 50

### Supervised Learning

In the supervised learning experiments:
- The model is trained using the limited labeled dataset.
- Achieved overall test accuracy is approximately **73.72%**.
- Detailed metrics (precision, recall, f1-score) per class and a confusion matrix provide insights into performance across various defect types.

### Semi-Supervised Learning

For semi-supervised learning:
- The FixMatch algorithm is applied using both labeled (900 samples) and unlabeled data (10,000 samples).
- Pseudo-labels are generated from weakly augmented images (with a threshold, e.g., 0.95) and enforced through strong augmentation.
- The autoencoder-based augmentation module reconstructs images with added noise and geometric transformations.
- The semi-supervised approach achieved an overall test accuracy of approximately **76.05%**, demonstrating improvements in classes with subtle defect patterns.

---

## Experimental Evaluation

- **Training Loss:**  
  In the supervised experiment, loss decreased from around 1.73 to 0.0920 over 50 epochs.
  
- **Performance Metrics:**  
  - **Supervised Learning:** Accuracy of 73.72% with varied performance across defect classes.
  - **Semi-Supervised Learning:** Improved accuracy (76.05%), with significant f1-score enhancements in classes such as Center, Edge-Ring, Scratch, and none.
  
- **Insights:**  
  The integration of unlabeled data via the FixMatch algorithm combined with autoencoder-based augmentation not only accelerates convergence but also enhances the generalization capability of the model.

---

## Conclusion and Future Work

This project demonstrates that combining semi-supervised learning with autoencoder-based strong augmentation can effectively address the challenge of limited labeled data in wafer defect detection.  
- **Key Outcomes:**
  - Supervised Learning Accuracy: 73.72%
  - Semi-Supervised Learning Accuracy: 76.05%
  - Notable improvements in defect classes with subtle patterns.
  
- **Future Directions:**
  - Dynamically adjusting the noise level in the latent space via hyperparameter tuning.
  - Expanding the autoencoder to deeper architectures for more precise feature extraction.
  - Refining the pseudo-label confidence calculation and exploring advanced network architectures (e.g., ResNet, DenseNet).

---

## References

1. Sohn, K., et al. (2020). *FixMatch: Simplifying Semi-Supervised Learning with Consistency and Confidence*.
2. Wang, Y., et al. (2018). *Wafer Defect Inspection Using Deep Convolutional Neural Networks*.
3. Kim, S., et al. (2019). *Semi-supervised Learning for Wafer Defect Detection*.
4. Choi, J., et al. (2021). *Autoencoder-Based Defect Detection in Semiconductor Manufacturing*.
5. Wu, M.-J., Jang, J.-S. R., & Chen, J.-L. (2015). *Wafer Map Failure Pattern Recognition and Similarity Ranking for Large-Scale Data Sets*. IEEE Transactions on Semiconductor Manufacturing.

---

This repository contains all the necessary code for data preprocessing, model training, and evaluation. Contributions and improvements are welcome!
