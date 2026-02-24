# Object detection

Lecturer: Silvia Cascianelli

## Introduction

Given an image, object detection is the task of returning the bounding boxes of the objects in the image, along with their class labels and confidence scores.

### Bounding boxes

Most of the time bounding boxes are axis aligned rectangles, but they can also be rotated rectangles or even polygons (less common).
* modal detection : only the visible part of the object is detected, so the bounding box may be smaller than the actual object.
* amodal detection : the entire object is detected, even if it is partially occluded, so the bounding box may be larger than the visible part of the object.
* bounding cubes: in 3D object detection, the bounding box is a cube that can be rotated in 3D space.
* oriented bounding boxes: rotated rectangles that can capture the orientation of the object.


### Evaluation metrics

* Intersection over Union (IoU) (also known as Jaccard Similarity) : the area of overlap between the predicted bounding box and the ground truth bounding box divided by the area of union between the two boxes. A common threshold for IoU is 0.5, meaning that a predicted box is considered a true positive if its IoU with a ground truth box is greater than 0.5. However, by varying the IoU threshold, we can compute the precision and recall at different levels of localization accuracy, which is the basis for the Average Precision (AP) metric.

* Average Precision (AP) : the area under the precision-recall curve, which is computed by varying the confidence threshold for the predicted boxes. The AP metric summarizes the trade-off between precision and recall at different confidence thresholds, and it is commonly used to evaluate object detection models. The mean Average Precision (mAP) is the average of the AP values across all classes in a multi-class object detection task.

* COCO mAP : the COCO dataset uses a more comprehensive evaluation metric that averages the AP across multiple IoU thresholds (from 0.5 to 0.95 with a step of 0.05) and across all classes. This metric is more stringent than the traditional AP at a single IoU threshold, as it requires the model to perform well across a range of localization accuracies.


## History

* Detection as regression (2014) : the first approach to object detection was to treat it as a regression problem, where the model directly predicts the coordinates of the bounding boxes (regression head) and the class labels (classification head). This approach was simple and fast, but it struggled with small objects and had limited accuracy.

* Sliding window approach (2014) : the model is applied to a sliding window across the image, and the predictions are made for each window. This approach was more accurate than the regression approach, but it was computationally expensive and struggled with scale variation.

* Region proposal approach (2014) : the model first generates a set of region proposals (potential bounding boxes) using a separate algorithm (e.g., Selective Search), and then classifies each proposal using a CNN. This approach was more accurate than the sliding window approach, but it was still computationally expensive and struggled with small objects. (See *Rich feature hierarchies for accurate object detection and semantic segmentation (R-CNN)* - 2014)

* Fast R-CNN (2015) : the issue in R-CNN was that the CNN had to be applied to each region proposal, which was computationally expensive. Fast R-CNN addressed this issue by applying the CNN to the entire image once, and then using a Region of Interest (RoI) pooling layer to extract features for each region proposal. This approach was much faster than R-CNN and achieved state-of-the-art accuracy at the time.

* Faster R-CNN (2015) : the issue in Fast R-CNN was that the region proposals were generated using a separate algorithm, which was still computationally expensive. Faster R-CNN addressed this issue by introducing a Region Proposal Network (RPN) that shares the convolutional features with the detection network, allowing for end-to-end training and much faster inference.

* Focal loss (2017) : the issue in Faster R-CNN was that it struggled with class imbalance, where the number of background examples far outweighed the number of foreground examples. Focal loss addressed this issue by down-weighting the loss for well-classified examples, allowing the model to focus more on hard examples and improving accuracy on small objects.

* You Only Look Once (YOLO) (2016) : the model divides the image into a grid and predicts bounding boxes and class probabilities for each grid cell. This approach is very fast, but it can struggle with small objects and has limited accuracy compared to region proposal approaches.

An evolution of the obj detection problem is to treat it as a sequence prediction problem, where the model generates a sequence of bounding boxes and class labels for each object in the image. This approach is more flexible and can handle variable numbers of objects, but it can be more complex to train and may require more data. 

* End-to-end Object Detection with Transformers (DETR) (2020) : the model uses a transformer architecture to directly predict a set of bounding boxes and class labels for each object in the image, without the need for region proposals or anchor boxes. This approach is more flexible and can handle variable numbers of objects, but it can be more complex to train and may require more data. However, it has shown promising results and has been adopted in many state-of-the-art object detection models.

