# Paddy-Rice-Maturity-Detection

This project was developed as part of the Final Year Engineering Project in the Bachelor of Engineering in Information Technology in 2024. It was a collaborative effort by a team of two members, aiming to develop an automated system for paddy rice maturity assessment using deep learning-based object detection models.

## **Overview**
Paddy rice maturity identification is a crucial agricultural task that impacts grain quality, production optimization, and harvest scheduling. Traditional methods are labor-intensive and subjective, prompting the need for automated solutions. This project presents a **comparative analysis of YOLOv8 and YOLOv4** object detection models for assessing paddy rice maturity using deep learning and digital imaging.

## **Features**
- **Automated detection of paddy rice maturity stages** using deep learning.
- **Comparative analysis of YOLOv8 and YOLOv4** for object detection performance.
- **Training and evaluation of models** on an annotated dataset of paddy rice images.
- **Analysis of detection accuracy, processing speed, and model efficiency**.

## **System Architecture**
The project utilizes **deep learning-based object detection** for maturity assessment. The architecture includes:
- **Dataset collection and preprocessing**: Images annotated with bounding boxes for maturity stage classification.
- **YOLOv4 Model**: Utilizes a convolutional neural network (CNN) for object detection with Darknet or CSPDarknet as backbone networks.
- **YOLOv8 Model**: Enhanced with Cross-Stage Partial Connections (CSPC) and Context Aggregation Modules (CAM) for improved accuracy.
- **Performance Evaluation**: Metrics such as **precision, recall, mean average precision (mAP)**, and inference time are used to assess model effectiveness.

## **Technologies Used**
- **Deep Learning Frameworks**: PyTorch, TensorFlow
- **Programming Languages**: Python
- **Libraries & Tools**:
  - Ultralytics YOLO
  - OpenCV (Image Processing)
  - Google Colab (GPU Training)
  - LabelImg (Image Annotation)

## **Dataset**
- **Collected from various sources** representing different growth stages of paddy rice.
- **5 maturity stages** labeled: Nursery, Transplantation, Panicle Initiation, Panicle Visible, and Mature.
- **Manual annotation using LabelImg** to create ground truth data.

## **Model Training & Evaluation**
- **Training on Google Colab using GPU acceleration**.
- **Optimization with data augmentation**, hyperparameter tuning, and loss function refinement.
- **Evaluation metrics**:
  - **Precision, Recall, mAP@50, mAP@50-95**
  - **Confusion matrix analysis**
  - **F1-score, PR curve, loss curves**

## **Results**
- **YOLOv8 outperformed YOLOv4** in precision (14% vs. 0%), recall (18% vs. 0%), and mAP (10% vs. 0.1%).
- **YOLOv4 struggled with overlapping bounding boxes**, resulting in lower detection performance.
- **Visualization using loss curves, PR curve, and confusion matrices** provided deeper insights.

## **Installation & Setup**
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repository/YOLOv8-vs-YOLOv4-Paddy-Detection.git
   ```
2. **Install dependencies**:
   ```bash
   pip install ultralytics opencv-python torch torchvision
   ```
3. **Prepare dataset**:
   - Download and organize images.
   - Annotate using **LabelImg**.
4. **Train the model**:
   - YOLOv4:
     ```bash
     !./darknet detector train data/train/image_data.data cfg/yolov4_train.cfg yolov4.conv.137 -dont_show
     ```
   - YOLOv8:
     ```bash
     !yolo task=detect mode=train model=yolov8s.pt data=data.yaml epochs=100 imgsz=800 plots=True
     ```
5. **Evaluate the model**:
   ```bash
   !yolo task=detect mode=val model=best.pt data=data.yaml
   ```

## **Future Scope**
- **Expand dataset** for better generalization.
- **Fine-tune YOLOv8 for improved accuracy**.
- **Deploy the model using IoT devices for real-time field analysis**.
- **Integration with drone and satellite imagery** for large-scale agricultural applications.

## **Contributors**
- **T. Nuthna**
- **S. Harshita**
- **Mrs. B. Leelavathy (Guide)** – (Research Supervision)
