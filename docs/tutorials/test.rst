Test
===================================

Now that you've got your data in a format the model expects, let's train an EfficientDet
to do object detection on our dataset.

Within the root directory of the project, run the following command

.. code-block:: bash

    python trashnet/test.py --path DATA_PATH --pretrained PRETRAINED_PATH --output OUTPUT_PATH

where DATA_PATH is the path to the COCO style dataset that you set up
previously, PRETRAINED_PATH is the path to the pretrained models from
the training loop and OUTPUT_PATH is the path you want to save the 
predicted images to.

For additional hyperparameter choices available during training, use help

.. code-block:: python
    
    python trashnet/test.py --help

When altering the hyperparametrs for the testing loop, to get the best results, try to 
be consistent with the hyperparameter choices for the training loop.