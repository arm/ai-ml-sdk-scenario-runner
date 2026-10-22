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

.. py:module:: scenario_runner_py

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

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [load_scenario]
   :end-before: .. end-api: [load_scenario]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ScenarioJsonFactory]
   :end-before: .. end-api: [ScenarioJsonFactory]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [Scenario]
   :end-before: .. end-api: [Scenario]

Scenario builder
~~~~~~~~~~~~~~~~

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ScenarioBuilder]
   :end-before: .. end-api: [ScenarioBuilder]

Resource creation
^^^^^^^^^^^^^^^^^

The resource methods accept either a resource description object or convenience
arguments. Both forms return the corresponding typed ID.

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [resource_creation]
   :end-before: .. end-api: [resource_creation]

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

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [BufferId]
   :end-before: .. end-api: [BufferId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ImageId]
   :end-before: .. end-api: [ImageId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [TensorId]
   :end-before: .. end-api: [TensorId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ShaderId]
   :end-before: .. end-api: [ShaderId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [RawDataId]
   :end-before: .. end-api: [RawDataId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [VgfId]
   :end-before: .. end-api: [VgfId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [GraphConstantResourceId]
   :end-before: .. end-api: [GraphConstantResourceId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [MemoryGroupId]
   :end-before: .. end-api: [MemoryGroupId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ImageBarrierId]
   :end-before: .. end-api: [ImageBarrierId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [BufferBarrierId]
   :end-before: .. end-api: [BufferBarrierId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [TensorBarrierId]
   :end-before: .. end-api: [TensorBarrierId]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [MemoryBarrierId]
   :end-before: .. end-api: [MemoryBarrierId]

Resource descriptions
~~~~~~~~~~~~~~~~~~~~~

The following classes have default constructors and writable attributes.
The convenience overloads on :py:class:`scenario_runner_py.ScenarioBuilder` avoid constructing
the buffer, image, and tensor descriptions directly.

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [BufferInfo]
   :end-before: .. end-api: [BufferInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ImageInfo]
   :end-before: .. end-api: [ImageInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [TensorInfo]
   :end-before: .. end-api: [TensorInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [SamplerSettings]
   :end-before: .. end-api: [SamplerSettings]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [RawDataInfo]
   :end-before: .. end-api: [RawDataInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ShaderInfo]
   :end-before: .. end-api: [ShaderInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [VgfInfo]
   :end-before: .. end-api: [VgfInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [GraphConstantInfo]
   :end-before: .. end-api: [GraphConstantInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [SpecializationConstant]
   :end-before: .. end-api: [SpecializationConstant]

Use :py:meth:`~scenario_runner_py.SpecializationConstant.from_int32`,
:py:meth:`~scenario_runner_py.SpecializationConstant.from_uint32`, or
:py:meth:`~scenario_runner_py.SpecializationConstant.from_float32` to construct
a value.

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [SpecializationConstantMap]
   :end-before: .. end-api: [SpecializationConstantMap]

Python bindings and commands
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [TypedBinding]
   :end-before: .. end-api: [TypedBinding]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ComputeDispatch]
   :end-before: .. end-api: [ComputeDispatch]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [DispatchComputeData]
   :end-before: .. end-api: [DispatchComputeData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [FragmentAttachment]
   :end-before: .. end-api: [FragmentAttachment]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [Extent2D]
   :end-before: .. end-api: [Extent2D]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [DispatchFragmentData]
   :end-before: .. end-api: [DispatchFragmentData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [PushConstantMap]
   :end-before: .. end-api: [PushConstantMap]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ShaderSubstitution]
   :end-before: .. end-api: [ShaderSubstitution]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ExternalDataGraphBinary]
   :end-before: .. end-api: [ExternalDataGraphBinary]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [DispatchVgfData]
   :end-before: .. end-api: [DispatchVgfData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [DispatchDataGraphBinaryData]
   :end-before: .. end-api: [DispatchDataGraphBinaryData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [DispatchDataGraphData]
   :end-before: .. end-api: [DispatchDataGraphData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [DispatchOpticalFlowData]
   :end-before: .. end-api: [DispatchOpticalFlowData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [PipelineBarrierData]
   :end-before: .. end-api: [PipelineBarrierData]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [FrameBoundaryData]
   :end-before: .. end-api: [FrameBoundaryData]

Python barriers
~~~~~~~~~~~~~~~

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [BaseBarrierInfo]
   :end-before: .. end-api: [BaseBarrierInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [SubresourceRange]
   :end-before: .. end-api: [SubresourceRange]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ImageBarrierInfo]
   :end-before: .. end-api: [ImageBarrierInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [BufferBarrierInfo]
   :end-before: .. end-api: [BufferBarrierInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [TensorBarrierInfo]
   :end-before: .. end-api: [TensorBarrierInfo]

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [MemoryBarrierInfo]
   :end-before: .. end-api: [MemoryBarrierInfo]

Python runtime options
~~~~~~~~~~~~~~~~~~~~~~

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [ScenarioOptions]
   :end-before: .. end-api: [ScenarioOptions]

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

The following reference is generated from the exported Python enumerations.

.. include:: _python_api/api.rst.inc
   :start-after: .. begin-api: [enums]
   :end-before: .. end-api: [enums]
