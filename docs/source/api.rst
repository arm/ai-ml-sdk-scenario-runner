.. SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
.. SPDX-License-Identifier: Apache-2.0

Scenario Runner C++ API
***********************

.. warning::

   The Scenario Runner C++ API is currently experimental and subject to change,
   including incompatible changes, in future releases.

The Scenario Runner API creates and executes Vulkan® scenarios. A scenario can
be constructed from an existing JSON description or defined programmatically.
Both workflows return a :cpp:class:`mlsdk::scenariorunner::Scenario`, which can
accept new in-memory inputs and be executed repeatedly without rebuilding its
resources and pipelines.

Build from JSON
~~~~~~~~~~~~~~~

Use :cpp:class:`mlsdk::scenariorunner::ScenarioJsonFactory` to preserve the
standalone Scenario Runner workflow in an application. Each UID defined for a
memory resource (buffer, image, or tensor) in the JSON file can be resolved to
the corresponding :cpp:type:`mlsdk::scenariorunner::BufferId`,
:cpp:type:`mlsdk::scenariorunner::ImageId`, or
:cpp:type:`mlsdk::scenariorunner::TensorId` for in-memory data transfer. Each
lookup validates that the UID exists and identifies the expected resource type:

.. literalinclude:: samples/4_run_json.cpp
   :language: cpp
   :start-after: API documentation: JSON scenario example begins.
   :end-before: API documentation: JSON scenario example ends.

Relative resource paths in the JSON file are resolved from the scenario file's
directory by default. The factory also accepts explicit working and output
directories. JSON resource ``src`` fields are loaded while the scenario is
created, and resource ``dst`` fields are written after each non-dry run. The
typed ``upload()`` and ``download()`` operations remain available alongside
this file-based input and output.

Build programmatically
~~~~~~~~~~~~~~~~~~~~~~

Use :cpp:func:`mlsdk::scenariorunner::createScenarioBuilder` when the scenario
is assembled by the application. Register resources before commands and retain
the returned IDs. Commands use those IDs for their bindings, and the same IDs
are used to transfer data after the scenario is built:

.. literalinclude:: samples/2_run_compute.cpp
   :language: cpp
   :start-after: API documentation: programmatic scenario example begins.
   :end-before: API documentation: programmatic scenario example ends.

The example has four stages:

1. ``createScenarioBuilder()`` creates the builder. ``addShader()`` and
   ``addBuffer()`` register every resource and return the IDs used below.
2. ``DispatchComputeData`` is configured to identify the shader, bind the
   input and output buffers to descriptor set 0 bindings 0 and 1, and specify
   four X-axis workgroups. ``addDispatchCompute()`` then appends that command
   to the builder's execution sequence. Each programmatic binding explicitly
   provides its Vulkan® descriptor type.
3. ``build()`` consumes the builder and returns an executable scenario. The
   resource IDs remain valid for the scenario's lifetime.
4. ``upload()``, ``run()``, and ``download()`` transfer the application input,
   execute the dispatch, and return the owning output data respectively.

Calling ``build()`` transfers the registered resources and commands to the
scenario. The builder must not be reused afterwards. Upload operations copy the
provided bytes before returning, while download operations return owning data.
For programmatically-created scenarios, ``ShaderInfo::src`` contains immutable
SPIR-V code and ``VgfInfo::src`` contains an immutable VGF view.
``readShaderCode()`` and ``loadVgfView()`` create these values from files;
relative paths are resolved from the application's current working directory.
The JSON factory resolves relative resource paths from the scenario file's
directory and performs these conversions automatically.

API Reference
~~~~~~~~~~~~~

The main entry points are:

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - API
     - Purpose
   * - :cpp:class:`mlsdk::scenariorunner::ScenarioJsonFactory`
     - Build a scenario from an existing JSON scenario file.
   * - :cpp:func:`mlsdk::scenariorunner::createScenarioBuilder`
     - Start defining a scenario programmatically.
   * - :cpp:class:`mlsdk::scenariorunner::ScenarioBuilder`
     - Register resources and commands, then build an executable scenario.
   * - :cpp:class:`mlsdk::scenariorunner::Scenario`
     - Upload inputs, execute commands, and download outputs.

.. contents:: API catalogue
   :local:
   :depth: 1

Scenario construction and execution
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenclass:: mlsdk::scenariorunner::ScenarioBuilder
   :project: ScenarioRunner
   :members:

.. doxygenfunction:: mlsdk::scenariorunner::createScenarioBuilder
   :project: ScenarioRunner

.. doxygenclass:: mlsdk::scenariorunner::ScenarioJsonFactory
   :project: ScenarioRunner
   :members:

.. doxygenclass:: mlsdk::scenariorunner::Scenario
   :project: ScenarioRunner
   :members:

In-memory resource data
^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: mlsdk::scenariorunner::BufferDataView
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::BufferData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ImageDataView
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ImageData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::TensorDataView
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::TensorData
   :project: ScenarioRunner
   :members:

Resource identifiers
^^^^^^^^^^^^^^^^^^^^

ScenarioBuilder returns a distinct opaque ID type for each resource category.
Retain these IDs to define command bindings and transfer resource data. Do not
construct IDs from numeric values.

.. doxygentypedef:: mlsdk::scenariorunner::BufferId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::ImageId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::TensorId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::ShaderId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::RawDataId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::VgfId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::GraphConstantResourceId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::MemoryGroupId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::ImageBarrierId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::BufferBarrierId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::TensorBarrierId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::MemoryBarrierId
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::MemoryResourceId
   :project: ScenarioRunner

Memory resource descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Scenario Runner translates these descriptions to Vulkan® values during runtime
setup. They intentionally provide a smaller API and do not necessarily map
one-to-one to Vulkan® enums or flag combinations.

.. doxygenenum:: mlsdk::scenariorunner::Tiling
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::FilterMode
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::AddressMode
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::BorderColor
   :project: ScenarioRunner

.. doxygentypedef:: mlsdk::scenariorunner::CustomColorValue
   :project: ScenarioRunner

.. doxygenstruct:: mlsdk::scenariorunner::BufferInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::SamplerSettings
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ImageInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::TensorInfo
   :project: ScenarioRunner
   :members:

Shader and graph resource descriptions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: mlsdk::scenariorunner::ShaderType
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::ShaderStage
   :project: ScenarioRunner

.. doxygenunion:: mlsdk::scenariorunner::Constant
   :project: ScenarioRunner

.. doxygenstruct:: mlsdk::scenariorunner::SpecializationConstant
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::SpecializationConstantMap
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::RawDataInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ShaderInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::VgfInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::GraphConstantInfo
   :project: ScenarioRunner
   :members:

Bindings and commands
^^^^^^^^^^^^^^^^^^^^^

.. doxygenstruct:: mlsdk::scenariorunner::TypedBinding
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ComputeDispatch
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::DispatchComputeData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::DispatchFragmentData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ResolvedPushConstantMap
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ResolvedShaderSubstitution
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::DispatchVgfData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::DispatchDataGraphData
   :project: ScenarioRunner
   :members:

.. doxygenenum:: mlsdk::scenariorunner::OpticalFlowGridSize
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::OpticalFlowPerformanceLevel
   :project: ScenarioRunner

.. doxygenstruct:: mlsdk::scenariorunner::DispatchOpticalFlowData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::PipelineBarrierData
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::FrameBoundaryData
   :project: ScenarioRunner
   :members:

Barriers and synchronization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. doxygenenum:: mlsdk::scenariorunner::MemoryAccess
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::PipelineStage
   :project: ScenarioRunner

.. doxygenenum:: mlsdk::scenariorunner::ImageLayout
   :project: ScenarioRunner

.. doxygenstruct:: mlsdk::scenariorunner::SubresourceRange
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::BaseBarrierInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::ImageBarrierInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::BufferBarrierInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::TensorBarrierInfo
   :project: ScenarioRunner
   :members:

.. doxygenstruct:: mlsdk::scenariorunner::MemoryBarrierInfo
   :project: ScenarioRunner
   :members:

Runtime options
^^^^^^^^^^^^^^^

.. doxygenstruct:: mlsdk::scenariorunner::ScenarioOptions
   :project: ScenarioRunner
   :members:
