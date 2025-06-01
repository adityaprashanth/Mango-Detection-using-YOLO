# YOLO v8, v10, v11, and v12: A Performance Comparison for Mango Detection in Agricultural Applications

## Abstract

Accurate mango detection is essential for effective agricultural management and harvesting automation. Due to challenges posed by varying lighting conditions, occlusions, and plant variability, existing computer vision algorithms often struggle to achieve satisfactory detection accuracy in real-world scenarios. This paper employs multiple versions of YOLO (You Only Look Once), a state-of-the-art object detection algorithm family, to reliably detect mangoes in agricultural settings. A comprehensive evaluation and benchmark of four state-of-the-art YOLO object detection algorithms (YOLOv8, YOLOv10, YOLOv11, and YOLOv12) is established for mango detection. YOLOv11 demonstrates remarkable test accuracy of 91.94% mAP@0.5, slightly outperforming other versions. Performance comparisons between these YOLO iterations using various metrics further illustrate their detection capabilities for mangoes. The real-time object detection capabilities of these models render them ideal solutions for mobile and embedded devices, opening new possibilities for automated agricultural management and harvesting systems.

**Index Terms**—YOLOv12, Object Detection, Artificial Intelligence Modeling, Deep Learning, Agricultural Vision

## I. Introduction

Fruit detection, a computer vision technique used to identify and localize fruits within images or video frames, has become increasingly valuable for various aspects of mango cultivation. Some potential applications include precision agriculture, crop monitoring and yield estimation, automated harvesting systems, quality assessment, and disease detection. Mangoes are important commercial fruits in many tropical and subtropical regions, contributing significantly to agricultural economies worldwide.

Artificial intelligence (A.I.) in precision agriculture has brought an agricultural revolution. Automated decision support systems using A.I. can enable farmers to make smarter decisions that improve production and management. A.I. can greatly help improve crop yield, reduce losses, and improve farming efficiencies. One of the major challenges for fruit detection is complex backgrounds, as they may contain other plants, leaves, and branches. Along with this, occlusions, varying lighting conditions, scale variation, and weather conditions are potential challenges.

Various pieces of research have been conducted previously for the detection of fruits in agricultural settings. Traditional computer vision techniques have been used in the past, but deep learning approaches, particularly convolutional neural networks (CNNs), have shown significant improvements in accuracy. YOLO (You Only Look Once) object detection algorithm is a single-stage detector that performs all its predictions from a single fully connected convolution layer in a single iteration, rendering it highly suitable for real-time applications, particularly for resource-constrained embedded devices. Given its remarkable balance between accuracy and speed, YOLO has emerged as one of the most widely utilized object detection algorithms in agricultural applications.

The literature review reveals substantial research efforts on the detection and classification of various agricultural products. Precision and accuracy are crucial metrics in agricultural technology as they determine the effectiveness of crop detection systems in the field. YOLO has gained widespread adoption in industrial settings due to its combination of high accuracy, efficient computation, and versatility. Moreover, in the field of object detection and computer vision, assessing the performance and generalization capabilities of a model is crucial. In this paper, multiple versions of YOLO (v8, v10, v11, and v12) are employed to reliably detect mangoes, and a comprehensive evaluation and benchmark of these state-of-the-art object detection algorithms is established.

## II. Proposed Method

### A. Dataset Description

This study utilized a mango dataset from Roboflow, specifically the "Clasificación-de-mangos" dataset created by Luigui Andre Cerna Grados. The dataset contains images of mangoes in various conditions and environments, providing a diverse set of samples for training and testing. The dataset was collected over time, capturing images from different angles and distances, under various lighting conditions and backgrounds.

The dataset offers a diverse and challenging real-world dataset for training AI models, as it comprises various factors of variations including different lighting conditions, backgrounds, occlusions, and different growth stages and sizes of mangoes. The dataset was properly labeled, with annotations provided in formats compatible with the YOLO framework.

The dataset was divided into the following splits:
- The training set was utilized to train the YOLO algorithm for the detection of mangoes.
- The validation set was utilized by YOLO during training to assess its performance. The validation set provides an unbiased estimate of the model's performance on unseen data. It is used to tune the model's hyperparameters and ensure that it is not overfitting to the training data.
- The evaluation of the trained model was performed on a test set of randomly selected images. The test set was used to assess the model's performance on unseen images.

YOLO employs text files for image annotations, with each file corresponding to an image and containing bounding box information specified by the object's class label, center points, and dimensions (width and height). This approach is efficient and effective for image annotation in object detection tasks.

### B. YOLO

YOLO is a real-time object detection algorithm that uses a deep neural network to detect objects in an image. YOLO is widely utilized in the industry due to its fast speed, high accuracy, and extensive community support. It was first introduced in 2016 by Joseph Redmon and Ali Farhadi and is known for its fast object detection speed and high accuracy. Since then, multiple versions of YOLO have been released, with continuous improvements in architecture and performance.

### C. YOLOv8, v10, v11, and v12

This study compares four recent versions of the YOLO framework:

1. **YOLOv8**: Released as a significant improvement over previous versions, YOLOv8 features improved developer convenience and reduces bounding box predictions because of being an anchor-free model, resulting in faster non-max suppression (NMS). YOLOv8 incorporates mosaic augmentation, which attaches multiple images in each epoch to encourage the model to learn objects in various locations, partial occlusions, and diverse surrounding pixels.

2. **YOLOv10**: Building on the success of YOLOv8, YOLOv10 introduces architectural refinements for improved computational efficiency and detection accuracy. It maintains the anchor-free approach while further optimizing the network structure.

3. **YOLOv11**: This version incorporates additional improvements to the backbone network and detection head, aiming for better feature extraction and localization capabilities. YOLOv11 also enhances the training process through refined augmentation strategies and loss functions.

4. **YOLOv12**: The latest version examined in this study, YOLOv12 features an optimized architecture with improved computational efficiency and detection accuracy. It incorporates advancements in network design and training methodologies to enhance performance.

### D. Working of YOLO

The YOLO algorithm divides its work into four different tasks:
- YOLO divides the input image into a grid of S x S cells and localizes each object within its corresponding cell, accompanied by the respective probability score.
- A single regression module determines the attributes of the bounding boxes and presents them in the form of a vector as shown in (1).
  ```
  y = [pc, bx, by, bh, bw, c1]    (1)
  ```
- Intersection over Union (IOU) metric is used to eliminate irrelevant boxes. IOU is as shown in (2).
  ```
  IOUtruth/pred = A ∩ B / A ∪ B    (2)
  ```
- Non-Maximum Suppression (NMS) is employed to retain only the highest probability bounding boxes, avoiding redundant or noisy detections.

### E. Training Parameters

In this research, the Google Colaboratory platform with T4 GPU was utilized for executing various deep learning models. The models were trained on the mango dataset described earlier. Training parameters are mentioned in Table 1.

**TABLE I: Training Parameters**
| Parameters | Parameter Values |
|------------|------------------|
| Learning Rate | 0.01 |
| Learning Rate Momentum | 0.937 |
| Learning Rate Decline Function | Cosine |
| Optimizer | SGD |
| Weight Decay Factor | 0.0005 |
| No. of Epochs | 30 |
| Batch Size | 16 |
| No. of Workers | 2 |
| Input Image | 640 x 640 pixels |

### F. Evaluation Metrics

The models' performance in mango detection was evaluated based on several indicators, including mean average precision (mAP) using IOU thresholds of 0.5 and 0.5:0.95. Precision, recall, F1-score, and inference time were also used for evaluation. True positives (TP), false positives (FP), true negatives (TN), and false negatives (FN) were used to evaluate these indicators. 

Precision is the ratio of true positive detections to all positive detections made by a model and is often paired with recall, which measures the proportion of total objects detected by the model as shown in (3) and (4).
```
Pr = TP / (TP + FP)    (3)
Re = TP / (TP + FN)    (4)
```

The F1-Score is a metric that incorporates both precision and recall and is calculated as the harmonic mean of these two values, as shown in (5).
```
F1 = 2 · Pr · Re / (Pr + Re)    (5)
```

Mean average precision (mAP) provides a single numeric value to evaluate the model's performance. Average precision (AP) represents the correct prediction percentage of each class. The formula for mAP is shown in (6).
```
mAP = (1/n) ∑(k=1 to n) APk    (6)
```

### G. Model Complexity Metrics

Model complexity is critical to assessing the computational cost and real-world performance. The complexity of a model depends on various factors, including:
- The number of parameters in a model indicates its complexity and impacts memory usage, computation costs, and inference times in real-world settings.
- Inference time is critical for real-time applications and was measured in this study to provide practical insights into deployment considerations.

## III. Results and Discussions

The model performance was evaluated using mAP with IOU thresholds of 0.5 and 0.5:0.95, and the results were recorded accordingly. The study trained all four versions of YOLO for a consistent number of epochs to provide a fair comparison of the performance of the models on the same dataset.

Table 2 presents the performance of models on test data, highlighting YOLOv11 as the superior performer with a mean average precision (mAP@0.5) of 91.94% on the mango dataset, slightly outperforming other YOLO versions. YOLOv11 also achieved the highest F1-score of 0.8810, indicating a good balance between precision and recall.

**TABLE II: COMPARISON OF YOLOv8, YOLOv10, YOLOv11, AND YOLOv12 MODELS**
| Model | mAP@0.5 | mAP@0.5:0.95 | Precision | Recall | F1-Score | Inference Time (s) | Training Time (min) |
|-------|---------|--------------|-----------|--------|----------|-------------------|-----------------|
| YOLOv8 | 0.9087 | 0.6425 | 0.9052 | 0.8534 | 0.8785 | 13.3864 | 49.0 |
| YOLOv10 | 0.8971 | 0.6178 | 0.8900 | 0.8305 | 0.8592 | 7.4955 | 55.5 |
| YOLOv11 | 0.9194 | 0.6410 | 0.9341 | 0.8336 | 0.8810 | 7.0858 | 52.2 |
| YOLOv12 | 0.9162 | 0.6412 | 0.9165 | 0.8291 | 0.8706 | 7.4938 | 69.3 |

The results indicate that all four YOLO versions achieved a mAP@0.5 of over 89%, with YOLOv11 showing the best performance at 91.94%. YOLOv8, despite being an earlier version, demonstrated competitive performance with a mAP@0.5 of 90.87%. However, it exhibited significantly longer inference time (13.38 seconds) compared to other versions, making it less suitable for real-time applications.

YOLOv11 not only achieved the highest mAP@0.5 but also demonstrated the fastest inference time at 7.09 seconds, making it the most efficient model for practical deployment in agricultural settings. Despite being the newest version, YOLOv12 showed slightly lower performance than YOLOv11 in terms of mAP@0.5 and F1-score, but it maintained comparable metrics with the earlier versions. However, YOLOv12 required the longest training time at 69.3 minutes.

In terms of precision and recall, YOLOv11 achieved the highest precision of 93.41%, indicating excellent accuracy in its detections with minimal false positives. YOLOv8 demonstrated the highest recall at 85.34%, suggesting it is slightly better at capturing all mangoes in the images.

These results suggest that the newer versions of YOLO (v10, v11, and v12) offer significant improvements in inference speed compared to YOLOv8, with YOLOv11 providing the best balance between accuracy and speed. The reduction in inference time from 13.39 seconds (YOLOv8) to around 7 seconds (YOLOv10, v11, and v12) represents approximately a 47% improvement in processing speed, which is crucial for real-time applications in agricultural settings.

The YOLOv11 model offers excellent performance and precision, making it a promising option for precision agriculture applications focused on mango detection. Its fast inference time also makes it suitable for real-world deployment in automated harvesting systems or quality assessment tools. However, all models may face challenges with detecting partially occluded mangoes or those in complex backgrounds with similar colors.

## IV. Conclusions

An exhaustive evaluation and benchmark of four state-of-the-art YOLO object detectors, including YOLOv8, v10, v11, and v12, were conducted for mango detection. The dataset comprised images of mangoes in various conditions, with different lighting conditions, backgrounds, and occlusions, posing substantial challenges to YOLO's detection capabilities.

YOLOv11 emerges as the most robust model among all investigated YOLO versions, with a mean average precision of 91.94% and the best balance between precision and recall as indicated by its F1-score of 0.8810. With respect to real-time implementation and processing speed, YOLOv11 is identified as the optimal option with an inference time of 7.09 seconds, significantly faster than YOLOv8.

In conclusion, the current study substantiates that YOLOv11 represents the most effective choice for implementing a mango detection system that demands high precision and reasonable processing speed. The models may face challenges in detecting mangoes that are partially occluded by leaves or branches, or in complex backgrounds with similar colors. Future research could investigate the potential of training these models to distinguish between different varieties of mangoes or to assess fruit ripeness, further enhancing their utility in agricultural applications.
