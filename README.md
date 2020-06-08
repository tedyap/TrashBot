# Trash Detection | CleanRobotics

Object Detection on a proprietary dataset of images provided by [CleanRobotics](https://cleanrobotics.com). The idea is to detect and localize objects captured in the staging bin of a CleanRobotics [TrashBot](https://cleanrobotics.com/trashbot/)&trade;, pass it onto the internal controller which will take care of the segregation based on the detected category and the local trash disposal policies.

## Project

Train an object detection model(EfficientDet w/EfficientNet backbone) to detect and tag objects on a custom dataset. Deploy the model to be consumed via an API hosted on the cloud. Quantize model and optimize for deployment on the Google Coral edge device. 