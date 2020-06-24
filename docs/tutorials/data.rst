Download Data
===================================

On the Supervise.ly platform, navigate to the concerned Workspace
and select the project  to work on. Now, download the data as a 
zipped archive, choosing to download the images along with the 
annotations.

Supervise.ly will then run a batch job, generating the necessary
.zip archive. Once it's downloaded, extract using a standard archiver.
To do so using the command line, open a terminal prompt and navigate 
to the download path. Now, type in the following command:

.. code-block:: bash

        unzip -q DOWNLOADED_FILE.zip -d SAVE_PATH
    
where DOWNLOADED_FILE.zip is the name of the downloaded file and 
SAVE_PATH is the destination on the disk.

You're now ready to generate the COCO-style dataset records expected
by the neural network.