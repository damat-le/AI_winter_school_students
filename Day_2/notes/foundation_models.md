# Foundation models for promptable segmentation

Lecturer: Silvia Cascianelli

Models trained on large datasets that can be adapted to a wide range of downstream tasks with minimal fine-tuning, generalize beyond the training data, and can be prompted to perform specific tasks. 

The first example of a foundation model for promptable segmentation is the *Segment Anything Model (SAM)* (2023), which is trained on a large dataset of images and can be prompted to perform various segmentation tasks, such as semantic segmentation, instance segmentation, and interactive segmentation.

SAM accepts various types of prompts, including:
* Point prompts : the user clicks on a point in the image, and the model segments the object at that point.
* Box prompts : the user draws a bounding box around an object, and the model segments the object within the box.
* Text prompts : the user provides a natural language description of the object to be segmented, and the model segments the object based on the description.