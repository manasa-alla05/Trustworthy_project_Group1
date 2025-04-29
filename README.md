
## Overview:

This project demonstrates a trustworthy object detection system using a pre-trained Faster R-CNN model on the COCO dataset. The system detects objects in an uploaded image, draws bounding boxes around them with confidence scores, and provides trustworthiness assessments for each detection.


## Features:

1. Object Detection: Uses Faster R-CNN with ResNet-50 backbone pre-trained on COCO dataset
2. Trustworthiness Assessment: Classifies detections as High/Medium/Low confidence based on prediction scores
3. Visualization: Displays detected objects with bounding boxes, class labels, and confidence information
4. COCO Integration: Automatically downloads and uses COCO dataset annotations for label mapping


## Project Structure:

├── images folder                # images to upload  
├── detection1.jpg               # output   
├── detection2.jpg               # output    
├── README.md                    # Project Documentation  
├── Trustworthy_code.ipynb       # Application logic    


## Requirements:

- Python 3.8+
- PyTorch 1.10+
- Torchvision 0.11+
- OpenCV 4.5+
- PIL
- COCO API (pycocotools)
- requests
- zipfile
- google.colab


## Installation:

1. Clone this repository or upload the notebook to Google Colab
2. Install required packages:
      "pip install torch torchvision opencv-python pycocotools"


## Usage:

1. Run the notebook in Google Colab or Jupyter environment
2. When prompted, upload an image file from the folder given as "images" or you can also download validation2017 COCO dataset images.     # we used the 2017 validation dataset images, but we are unable to upload the whole folder as it has more than 1000 images.
3. The notebook will:

    - Load and preprocess the image
    - Run object detection using Faster R-CNN
    - Filter detections by confidence threshold (default: 0.5)
    - Download COCO annotations if not present
    - Visualize detections with bounding boxes and labels
    - Display trustworthiness assessment for each detection


## Key Components:

1. Model: Faster R-CNN with ResNet-50 FPN backbone
2. Dataset: COCO (Common Objects in Context)
3. Trustworthiness Metrics:
4. High confidence: score > 0.75
5. Medium confidence: 0.5 < score ≤ 0.75
6. Low confidence: score ≤ 0.5


## Output Generation Process:

1. Object Detection Pipeline:
   
    - Model Inference: The pre-trained Faster R-CNN model processes the input image and generates:
        
        - Bounding box coordinates for detected objects
        - Class predictions for each detection
        - Confidence scores (0-1) indicating prediction certainty

    - Threshold Filtering: Only detections with confidence scores above 0.5 are retained for display and analysis

2. Visualization Components:

    - Bounding Boxes: Green rectangles drawn around detected objects

    - Class Labels: Text annotations showing:

        - COCO dataset class names (e.g., "person", "car", "dog")
        - Positioned above each bounding box in blue text

    - Trustworthiness Indicators:

        - High confidence (>0.75): Solid bounding boxes
        - Medium confidence (0.5-0.75): Dashed bounding boxes
        - Confidence scores displayed alongside class labels

3. COCO Annotation Handling:

    - Automatic Download: If annotations not found locally:

        - Downloads COCO annotations zip file
        - Extracts only the validation set annotations
        - Cleans up temporary files
        - Stores annotations at /content/instances_val2017.json

    - Label Mapping: Uses COCO's category IDs to:

        - Convert numeric predictions to human-readable labels
        - Provide accurate object class information

4. Output Display:

For each detection, the notebook displays:

1. The image with bounding boxes
2. Detection number
3. Object class name
4. Confidence score
5. Trustworthiness level

## Source of the code:

1. Chatgpt:
         - To enhance the standard Faster R-CNN model.

## Example Output:

The processed image will display the following annotated elements:

* Bounding Boxes:

    - Solid rectangles around high-confidence detections (score > 0.75)
    - Dashed yellow rectangles around medium-confidence detections (0.5 ≤ score ≤ 0.75)
    - Box thickness proportional to detection confidence.
    

## Acknowledgments:

This project was made possible by the following open-source tools and datasets:

1. Core Technologies:

    - Mask R-CNN architecture for high-quality object detection
    - PyTorch and torchvision for deep learning framework and pre-trained models
    - OpenCV for image processing and visualization capabilities

2. Dataset:

    - COCO (Common Objects in Context) dataset for comprehensive object annotations.

3. Infrastructure:

    - Google Colab for providing free GPU-accelerated notebook environment
    - Python ecosystem for scientific computing


