Cloning the repository
======================

To clone the |SR_project| as a stand-alone repository, you can use regular git clone commands. However we recommend
using the :code:`git-repo` tool to clone the repository as part of the ML SDK for Vulkan® suite. The tool is available here:
(|git_repo_tool_url|). This ensures all dependencies are fetched and in a suitable default location on the file
system.

For a minimal build and to initialize only the |SR_project| and its dependencies, run:

.. code-block:: bash

    repo init -u  <server>/ml-sdk-for-vulkan-manifest -g scenario-runner

Alternatively, to initialise the repo structure for the entire ML SDK for Vulkan®, including the Scenario Runner, run:

.. code-block:: bash

    repo init -u  <server>/ml-sdk-for-vulkan-manifest -g all

Once the repo is initialized, you can fetch the contents:

.. code-block:: bash

    repo sync

.. note::
    You must enable long paths on Windows®. To ensure nested submodules do not exceed the maximum long path length, you must clone close to the root directory or use a symlink.

After the sync command completes successfully, you can find the Scenario Runner in :code:`<repo_root>/sw/scenario-runner/`.
You can also find all the dependencies that the Scenario Runner requires in :code:`<repo_root>/dependencies/`.
