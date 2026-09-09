.. SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
.. SPDX-License-Identifier: Apache-2.0

Scenario Runner Python API Reference
************************************

.. warning::

   The Scenario Runner Python API is currently experimental and subject to change,
   including incompatible changes, in future releases.

The Python API is provided by the ``scenario_runner_py`` extension module. It
supports both loading existing JSON scenarios and constructing scenarios
programmatically. Resource uploads and downloads use C-contiguous NumPy arrays.

.. py:currentmodule:: scenario_runner_py

Build and import the bindings
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Build the bindings from a Scenario Runner source checkout:

.. code-block:: shell

   python3 scripts/build.py --build-pylib

The extension is written to ``build/src`` by default. Add that directory to
``PYTHONPATH`` when running a Python application, or configure your IDE to use
it. From the source checkout:

.. code-block:: shell

   export PYTHONPATH="$PWD/build/src${PYTHONPATH:+:$PYTHONPATH}"

Then import the module:

.. code-block:: python

   import scenario_runner_py as sr

Python examples
~~~~~~~~~~~~~~~

Load a JSON scenario, replace an input buffer, and retrieve an output buffer:

.. literalinclude:: samples/python/1_run_json.py
   :language: python

For a programmatic scenario, create resources with a builder and retain the
returned IDs for subsequent transfer operations:

.. literalinclude:: samples/python/2_run_compute.py
   :language: python

Python entry points
~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Purpose
   * - :py:func:`~scenario_runner_py.load_scenario`
     - Load and build an existing JSON scenario file.
   * - :py:class:`~scenario_runner_py.ScenarioJsonFactory`
     - Class-based form of the JSON loading API.
   * - :py:class:`~scenario_runner_py.ScenarioBuilder`
     - Register resources and commands programmatically.
   * - :py:class:`~scenario_runner_py.Scenario`
     - Upload NumPy inputs, execute commands, and download NumPy outputs.

.. contents:: API catalogue
   :local:
   :depth: 1

Scenario creation and execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: scenario_runner_py.load_scenario

.. autoclass:: scenario_runner_py.ScenarioJsonFactory
   :members:

.. autoclass:: scenario_runner_py.Scenario
   :members:

Scenario builder
~~~~~~~~~~~~~~~~

.. autoclass:: scenario_runner_py.ScenarioBuilder
   :members:
   :exclude-members: add_buffer, add_image, add_tensor

Resource creation
^^^^^^^^^^^^^^^^^

The resource methods accept either a resource description object or convenience
arguments. Both forms return the corresponding typed ID.

.. pybind-overloads:: ScenarioBuilder
   :methods: add_buffer, add_image, add_tensor

Python resource identifiers
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Typed IDs are immutable and hashable. They are created by the methods below,
not constructed directly. The ``value`` property exposes the underlying numeric
identifier for inspection or debugging; it is meaningful only within its
scenario.

.. list-table::
   :header-rows: 1
   :widths: 25 30 45

   * - Type
     - Identifies
     - Created by
   * - :py:class:`~scenario_runner_py.BufferId`
     - A buffer
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_buffer` or
       :py:meth:`~scenario_runner_py.Scenario.get_buffer_id`
   * - :py:class:`~scenario_runner_py.ImageId`
     - An image
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_image` or
       :py:meth:`~scenario_runner_py.Scenario.get_image_id`
   * - :py:class:`~scenario_runner_py.TensorId`
     - A tensor
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_tensor` or
       :py:meth:`~scenario_runner_py.Scenario.get_tensor_id`
   * - :py:class:`~scenario_runner_py.ShaderId`
     - Shader data
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_shader`
   * - :py:class:`~scenario_runner_py.RawDataId`
     - Raw binary data
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_raw_data`
   * - :py:class:`~scenario_runner_py.VgfId`
     - A VGF
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_vgf`
   * - :py:class:`~scenario_runner_py.GraphConstantResourceId`
     - A graph constant
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_graph_constant`
   * - :py:class:`~scenario_runner_py.MemoryGroupId`
     - A memory aliasing group
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.create_memory_group`
   * - :py:class:`~scenario_runner_py.ImageBarrierId`
     - An image barrier
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_image_barrier`
   * - :py:class:`~scenario_runner_py.BufferBarrierId`
     - A buffer barrier
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_buffer_barrier`
   * - :py:class:`~scenario_runner_py.TensorBarrierId`
     - A tensor barrier
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_tensor_barrier`
   * - :py:class:`~scenario_runner_py.MemoryBarrierId`
     - A memory barrier
     - :py:meth:`~scenario_runner_py.ScenarioBuilder.add_memory_barrier`

.. automodule:: scenario_runner_py
   :members: BufferId, ImageId, TensorId, ShaderId, RawDataId, VgfId, GraphConstantResourceId, MemoryGroupId, ImageBarrierId, BufferBarrierId, TensorBarrierId, MemoryBarrierId

Resource descriptions
~~~~~~~~~~~~~~~~~~~~~

The following classes have default constructors and writable attributes.
The convenience overloads on :py:class:`ScenarioBuilder` avoid constructing
the buffer, image, and tensor descriptions directly.

.. autoclass:: scenario_runner_py.BufferInfo

.. autoclass:: scenario_runner_py.ImageInfo

.. autoclass:: scenario_runner_py.TensorInfo

.. autoclass:: scenario_runner_py.SamplerSettings

.. autoclass:: scenario_runner_py.RawDataInfo

.. autoclass:: scenario_runner_py.ShaderInfo

.. autoclass:: scenario_runner_py.VgfInfo

.. autoclass:: scenario_runner_py.GraphConstantInfo

.. autoclass:: scenario_runner_py.SpecializationConstant

Use :py:meth:`~scenario_runner_py.SpecializationConstant.from_int32`,
:py:meth:`~scenario_runner_py.SpecializationConstant.from_uint32`, or
:py:meth:`~scenario_runner_py.SpecializationConstant.from_float32` to construct
a value.

.. automethod:: scenario_runner_py.SpecializationConstant.from_int32

.. automethod:: scenario_runner_py.SpecializationConstant.from_uint32

.. automethod:: scenario_runner_py.SpecializationConstant.from_float32

.. autoclass:: scenario_runner_py.SpecializationConstantMap

Python bindings and commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scenario_runner_py.TypedBinding

.. autoclass:: scenario_runner_py.ComputeDispatch

.. autoclass:: scenario_runner_py.DispatchComputeData

.. autoclass:: scenario_runner_py.FragmentAttachment

.. autoclass:: scenario_runner_py.Extent2D

.. autoclass:: scenario_runner_py.DispatchFragmentData

.. autoclass:: scenario_runner_py.PushConstantMap

.. autoclass:: scenario_runner_py.ShaderSubstitution

.. autoclass:: scenario_runner_py.DispatchVgfData

.. autoclass:: scenario_runner_py.DispatchDataGraphData

.. autoclass:: scenario_runner_py.DispatchOpticalFlowData

.. autoclass:: scenario_runner_py.PipelineBarrierData

.. autoclass:: scenario_runner_py.FrameBoundaryData

Python barriers
~~~~~~~~~~~~~~~

.. autoclass:: scenario_runner_py.BaseBarrierInfo

.. autoclass:: scenario_runner_py.SubresourceRange

.. autoclass:: scenario_runner_py.ImageBarrierInfo

.. autoclass:: scenario_runner_py.BufferBarrierInfo

.. autoclass:: scenario_runner_py.TensorBarrierInfo

.. autoclass:: scenario_runner_py.MemoryBarrierInfo

Python runtime options
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: scenario_runner_py.ScenarioOptions

Supported resource formats
~~~~~~~~~~~~~~~~~~~~~~~~~~

The ``Format`` enum uses Python-style names for Vulkan® ``VkFormat`` values;
for example, ``Format.R8Uint`` represents ``VK_FORMAT_R8_UINT``. The supported
tensor formats and file-backed image formats are listed in the :ref:`image`
and :ref:`tensor` sections of the JSON specification. The Python binding
accepts its exposed ``Format`` values when creating resources; the selected
Vulkan® device must support each format for its requested resource usage.

Enumerations
~~~~~~~~~~~~

The following table is generated from the exported Python enumerations.

.. py-enum-table::
