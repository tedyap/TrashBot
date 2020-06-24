Generate COCO style dataset
===================================

After downloading and extracting the data, we need to convert it to a
format that the network expects. We'll do that now.

Remember where you extracted your archive. For this tutorial, we'll consider
**~/Desktop/TrashBot** as the extracted location.

Now, inside the root directory of the project, make the build_dataset bash file
an executable. On Linux, you can do it like this:

.. code-block:: bash

        chmod u+x build_dataset.sh

Now, go ahead and run the build_dataset.sh script as follows:

.. code-block:: bash

        ./build_dataset.sh --data_path ~/Desktop/TrashBot --output ~/Desktop/coco

where --output represents where I want the COCO dataset to be stored on disk
and --data_path represents my Supervise.ly extracted archive

That's all! You're all set. Now, go ahead and train a model.