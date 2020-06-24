Train
===================================

Now that you've got your data in a format the model expects, let's train an EfficientDet
to do object detection on our dataset.

Within the root directory of the project, run the following command

.. code-block:: bash

    python trashnet/train.py --num_epochs NUM_EPOCHS --path DATA_PATH

where NUM_EPOCHS is the number of epochs you want to train the network for
and DATA_PATH is the path to the COCO style dataset that you set up
previously.

For additional hyperparameter choices available during training, use help

.. code-block:: bash

    python trashnet/train.py --help

The training loop comes with default hyperparameters that have been tested to work
on the dataset, but feel free to try and experiment.

Depending on the underlying hardware and the number of epochs you're training for,
it can take anywhere from a couple of minutes to a day for the network to finish
training. Go ahead and grab a coffee while the network learns.