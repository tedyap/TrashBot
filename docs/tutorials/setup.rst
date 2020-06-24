Setup
===================================

Welcome! This tutorial will walk you through setting up the project,
creating the necessary environments and installing all the dependencies
in one go.

Start by cloning the project repository from `GitHub <https://github.com/jsaurabh/Trashbot/>`_.
Once you've cloned the repository, enter the directory and visually inspect 
that the clone was successful

On a Linux machine, it would look something like this

.. code-block:: bash

    git clone https://github.com/jsaurabh/TrashBot.git
    cd Trashbot
    ls

Once you've confirmed everything is the way it is supposed to be, go ahead and start
installing all the dependencies

.. code-block:: bash

    conda env update

The command above will setup a conda environment named *trashbot* that will contain all
the dependencies needed for the project. After a couple of minutes, the environments will
be ready. To start using the environment, activate it

.. code-block:: bash

    conda activate trashbot

To change the name of the environment, edit the first line in the *environment.yml* file
in the root of the project directory.