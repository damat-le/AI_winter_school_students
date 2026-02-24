# Instance Segmentation

Lecturer: Silvia Cascianelli

## Introduction

Instance segmentation is the task of segmenting each object instance in an image, along with its class label and confidence score. 

Compared to object detection, instance segmentation provides a more detailed understanding of the scene by not only localizing objects with bounding boxes but also delineating their precise shapes.

Compared to semantic segmentation, instance segmentation distinguishes between different instances of the same class, allowing for a more granular understanding of the scene. E.g., in a street scene, instance segmentation can differentiate between multiple cars, while semantic segmentation would label all cars as the same class without distinguishing between them.

## History

* Mask R-CNN (2017) : the model extends Faster R-CNN by adding a branch for predicting a binary mask for each region proposal, allowing for instance segmentation. This approach is accurate and efficient, and it has become a popular baseline for instance segmentation tasks.


---

# Bonus: Interactive Segmentation

Interactive segmentation is a technique where the user provides input (e.g., clicks, scribbles, or bounding boxes) to guide the segmentation process. This approach is particularly useful when dealing with complex scenes or objects that are difficult to segment automatically. The user's input helps the model focus on specific regions of interest and improve segmentation accuracy.

Before deeplearning, a simple interactive segmentation method was the "GrabCut" algorithm, which uses graph cuts to segment an image based on user-provided bounding boxes. The only thing this algorithm was able to do was to separate the foreground from the background, but it was not able to distinguish between different instances of the same class. It was able to do that based on user-provided scribbles.



