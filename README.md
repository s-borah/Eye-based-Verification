# Eye-based-Verification
Proposed contact-less attendance system for organizations by verifying identity using iris and periocular region.

## Abstract:

The proposed method to verify identity for attendance would be to capture images at a distance using Near Infrared Imaging. Before capturing image, user must scan id card barcode to match with database entry. Iris must be clearly visible and image must be high resolution if taken from a distance (so that there is atleast 224x224 pixels in the eye region.

By use of *haarcascade_eye.xml* and *haarcascade_frontalface_default.xml* in **OpenCV** both eyes can be detected and extracted from the captured image. Transfer learning was applied by using an ImageNet trained **ResNet50** Deep Learning Network (easily deployed using [torchvision.models](#https://pytorch.org/docs/stable/torchvision/models.html)) for feature extraction and adding a few linear layers which are then used for verification of the input eye image with the image already stored in the database. By adding a more weight in the cost to false positives, precision of the system was improved.

Further, liveness detection can be incorporated into the system to make it more robust to spoofing.

## Approach:

Transfer Learning was applied in **Pytorch** by freezing all the layers of pre-trained ResNet50 (thus, fine tuning was not performed) and removing the last layer. Two architectures were trained by using either 4 fully connected layers (*model0_100.pth* and *model1_100.pth*) or 3 fully connected layers after ResNet50 (*model2_100.pth* and *model3 reg 6e-5.pth*). In both the cases, features are extracted for both input image and database image using pre-trained network. The outputs are merged and passed through two fully connected layers and finally passed through a logistic regression function for binary classification (*whether the images are from the same user or not*).
