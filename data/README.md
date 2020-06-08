# Data

Data was annotated on the [Supervise.ly](https://supervise.ly) platform. The  Supervise.ly makes the annotations available in its own proprietary format, requiring conversion to MS-COCO before any kind of training.

## COCO conversion

The annotated data was then downloaded in the Supervise.ly format, and converted to the more standard MS-COCO format. The conversion script is heavily inspired by previous scripts written by [Caio Marcellos](https://gist.github.com/caiofcm/0b93b0084669a1287633d9ebf32f3833) and [Sai Peri](https://github.com/speri203/Supervisely2COCO/blob/master/supervisely2coco.py)
