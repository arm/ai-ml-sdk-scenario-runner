.. Generated from the native Python API; do not edit.

.. begin-api: [ScenarioJsonFactory]

.. py:class:: ScenarioJsonFactory
   :module: scenario_runner_py

   Create scenarios from JSON descriptions.


   .. py:method:: ScenarioJsonFactory.make(scenario_file: os.PathLike | str | bytes, *, work_dir: os.PathLike | str | bytes = Path('.'), output_dir: os.PathLike | str | bytes = Path('.'), options: scenario_runner_py.ScenarioOptions = ScenarioOptions()) -> scenario_runner_py.Scenario
      :module: scenario_runner_py
      :staticmethod:

      Load a JSON scenario and return a ready-to-run :class:`Scenario`.

.. end-api: [ScenarioJsonFactory]

.. begin-api: [Scenario]

.. py:class:: Scenario
   :module: scenario_runner_py

   A built scenario that owns runtime resources and commands. It can be run repeatedly.


   .. py:method:: Scenario.download
      :module: scenario_runner_py

      Overloaded function.

      ``1. download(self: scenario_runner_py.Scenario, id: scenario_runner_py.BufferId) -> numpy.ndarray``

      Return the buffer contents as a ``uint8`` NumPy array.

      ``2. download(self: scenario_runner_py.Scenario, id: scenario_runner_py.TensorId) -> numpy.ndarray``

      Return tensor contents as a NumPy array derived from its format.

      ``3. download(self: scenario_runner_py.Scenario, id: scenario_runner_py.ImageId) -> numpy.ndarray``

      Return image contents as a NumPy array derived from its format. Multi-component images use a packed dtype.


   .. py:method:: Scenario.get_buffer_id(self: scenario_runner_py.Scenario, uid: str) -> scenario_runner_py.BufferId
      :module: scenario_runner_py

      Return the buffer ID associated with a JSON resource UID.


   .. py:method:: Scenario.get_image_id(self: scenario_runner_py.Scenario, uid: str) -> scenario_runner_py.ImageId
      :module: scenario_runner_py

      Return the image ID associated with a JSON resource UID.


   .. py:method:: Scenario.get_tensor_id(self: scenario_runner_py.Scenario, uid: str) -> scenario_runner_py.TensorId
      :module: scenario_runner_py

      Return the tensor ID associated with a JSON resource UID.


   .. py:method:: Scenario.run(self: scenario_runner_py.Scenario, *, repeat_count: typing.SupportsInt | typing.SupportsIndex = 1, dry_run: bool = False) -> None
      :module: scenario_runner_py

      Execute the command sequence.


   .. py:method:: Scenario.upload
      :module: scenario_runner_py

      Overloaded function.

      ``1. upload(self: scenario_runner_py.Scenario, id: scenario_runner_py.BufferId, data: numpy.ndarray) -> None``

      Copy a C-contiguous NumPy array into the buffer selected by ``id``.

      ``2. upload(self: scenario_runner_py.Scenario, id: scenario_runner_py.TensorId, data: numpy.ndarray) -> None``

      Copy a C-contiguous NumPy array into the tensor selected by ``id``.

      ``3. upload(self: scenario_runner_py.Scenario, id: scenario_runner_py.ImageId, data: numpy.ndarray) -> None``

      Copy one mip level from a C-contiguous NumPy array into the image selected by ``id``.

.. end-api: [Scenario]

.. begin-api: [ScenarioBuilder]

.. py:class:: ScenarioBuilder()
   :module: scenario_runner_py

   Define resources and commands, then build a :class:`Scenario`.


   .. py:method:: ScenarioBuilder.add_buffer_barrier(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.BufferBarrierInfo) -> scenario_runner_py.BufferBarrierId
      :module: scenario_runner_py

      Register a :class:`BufferBarrierInfo` and return its barrier ID.


   .. py:method:: ScenarioBuilder.add_dispatch_compute(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.DispatchComputeData) -> None
      :module: scenario_runner_py

      Append a :class:`DispatchComputeData` command.


   .. py:method:: ScenarioBuilder.add_dispatch_data_graph(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.DispatchDataGraphData) -> None
      :module: scenario_runner_py

      Append a :class:`DispatchDataGraphData` command.


   .. py:method:: ScenarioBuilder.add_dispatch_data_graph_binary(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.DispatchDataGraphBinaryData) -> None
      :module: scenario_runner_py

      Append a :class:`DispatchDataGraphBinaryData` command.


   .. py:method:: ScenarioBuilder.add_dispatch_fragment(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.DispatchFragmentData) -> None
      :module: scenario_runner_py

      Append a :class:`DispatchFragmentData` command.


   .. py:method:: ScenarioBuilder.add_dispatch_optical_flow(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.DispatchOpticalFlowData) -> None
      :module: scenario_runner_py

      Append a :class:`DispatchOpticalFlowData` command.


   .. py:method:: ScenarioBuilder.add_dispatch_vgf(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.DispatchVgfData) -> None
      :module: scenario_runner_py

      Append a :class:`DispatchVgfData` command.


   .. py:method:: ScenarioBuilder.add_frame_boundary(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.FrameBoundaryData) -> None
      :module: scenario_runner_py

      Append a :class:`FrameBoundaryData` command.


   .. py:method:: ScenarioBuilder.add_graph_constant(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.GraphConstantInfo) -> scenario_runner_py.GraphConstantResourceId
      :module: scenario_runner_py

      Register :class:`GraphConstantInfo` and return its ID.


   .. py:method:: ScenarioBuilder.add_image_barrier(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.ImageBarrierInfo) -> scenario_runner_py.ImageBarrierId
      :module: scenario_runner_py

      Register an :class:`ImageBarrierInfo` and return its barrier ID.


   .. py:method:: ScenarioBuilder.add_memory_barrier(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.MemoryBarrierInfo) -> scenario_runner_py.MemoryBarrierId
      :module: scenario_runner_py

      Register a :class:`MemoryBarrierInfo` and return its barrier ID.


   .. py:method:: ScenarioBuilder.add_pipeline_barrier(self: scenario_runner_py.ScenarioBuilder, command: scenario_runner_py.PipelineBarrierData) -> None
      :module: scenario_runner_py

      Append a :class:`PipelineBarrierData` command.


   .. py:method:: ScenarioBuilder.add_raw_data(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.RawDataInfo) -> scenario_runner_py.RawDataId
      :module: scenario_runner_py

      Register :class:`RawDataInfo` and return its ID.


   .. py:method:: ScenarioBuilder.add_resource_to_memory_group(self: scenario_runner_py.ScenarioBuilder, group: scenario_runner_py.MemoryGroupId, resource: object) -> None
      :module: scenario_runner_py

      Add a buffer, image, or tensor ID to an existing aliasing group.


   .. py:method:: ScenarioBuilder.add_shader(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.ShaderInfo) -> scenario_runner_py.ShaderId
      :module: scenario_runner_py

      Register :class:`ShaderInfo` and return its ID.


   .. py:method:: ScenarioBuilder.add_tensor_barrier(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.TensorBarrierInfo) -> scenario_runner_py.TensorBarrierId
      :module: scenario_runner_py

      Register a :class:`TensorBarrierInfo` and return its barrier ID.


   .. py:method:: ScenarioBuilder.add_vgf(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.VgfInfo) -> scenario_runner_py.VgfId
      :module: scenario_runner_py

      Register :class:`VgfInfo` and return its ID.


   .. py:method:: ScenarioBuilder.build(self: scenario_runner_py.ScenarioBuilder, *, options: scenario_runner_py.ScenarioOptions = ScenarioOptions()) -> scenario_runner_py.Scenario
      :module: scenario_runner_py

      Consume the builder and return a ready-to-run :class:`Scenario`. The builder cannot be reused.


   .. py:method:: ScenarioBuilder.create_memory_group(self: scenario_runner_py.ScenarioBuilder) -> scenario_runner_py.MemoryGroupId
      :module: scenario_runner_py

      Create an aliasing group and return its ID.

.. end-api: [ScenarioBuilder]

.. begin-api: [BufferInfo]

.. py:class:: BufferInfo()
   :module: scenario_runner_py


   .. py:property:: BufferInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: BufferInfo.memory_offset
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: BufferInfo.size
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [BufferInfo]

.. begin-api: [ImageInfo]

.. py:class:: ImageInfo()
   :module: scenario_runner_py


   .. py:property:: ImageInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: ImageInfo.format
      :module: scenario_runner_py

      :type: Format


      Writable.


   .. py:property:: ImageInfo.is_color_attachment
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ImageInfo.is_input
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ImageInfo.is_sampled
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ImageInfo.is_storage
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ImageInfo.memory_offset
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: ImageInfo.mips
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: ImageInfo.sampler_settings
      :module: scenario_runner_py

      :type: SamplerSettings


      Writable.


   .. py:property:: ImageInfo.shape
      :module: scenario_runner_py

      :type: list[int]


      Writable.


   .. py:property:: ImageInfo.target_format
      :module: scenario_runner_py

      :type: Format


      Writable.


   .. py:property:: ImageInfo.tiling
      :module: scenario_runner_py

      :type: Tiling | None


      Writable.

.. end-api: [ImageInfo]

.. begin-api: [TensorInfo]

.. py:class:: TensorInfo()
   :module: scenario_runner_py


   .. py:property:: TensorInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: TensorInfo.descriptor_buffer_capture_replay
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: TensorInfo.format
      :module: scenario_runner_py

      :type: Format


      Writable.


   .. py:property:: TensorInfo.memory_offset
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: TensorInfo.shape
      :module: scenario_runner_py

      :type: list[int]


      Writable.


   .. py:property:: TensorInfo.sparsity_dimension
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: TensorInfo.tiling
      :module: scenario_runner_py

      :type: Tiling


      Writable.

.. end-api: [TensorInfo]

.. begin-api: [SamplerSettings]

.. py:class:: SamplerSettings()
   :module: scenario_runner_py


   .. py:property:: SamplerSettings.address_mode_u
      :module: scenario_runner_py

      :type: AddressMode


      Writable.


   .. py:property:: SamplerSettings.address_mode_v
      :module: scenario_runner_py

      :type: AddressMode


      Writable.


   .. py:property:: SamplerSettings.address_mode_w
      :module: scenario_runner_py

      :type: AddressMode


      Writable.


   .. py:property:: SamplerSettings.border_color
      :module: scenario_runner_py

      :type: BorderColor


      Writable.


   .. py:property:: SamplerSettings.custom_border_color
      :module: scenario_runner_py

      :type: tuple[float, float, float, float] | tuple[int, int, int, int]


      Writable.


   .. py:property:: SamplerSettings.mag_filter
      :module: scenario_runner_py

      :type: FilterMode


      Writable.


   .. py:property:: SamplerSettings.min_filter
      :module: scenario_runner_py

      :type: FilterMode


      Writable.


   .. py:property:: SamplerSettings.mip_filter
      :module: scenario_runner_py

      :type: FilterMode


      Writable.

.. end-api: [SamplerSettings]

.. begin-api: [RawDataInfo]

.. py:class:: RawDataInfo()
   :module: scenario_runner_py


   .. py:property:: RawDataInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: RawDataInfo.src
      :module: scenario_runner_py

      :type: str


      Writable.

.. end-api: [RawDataInfo]

.. begin-api: [ShaderInfo]

.. py:class:: ShaderInfo()
   :module: scenario_runner_py


   .. py:property:: ShaderInfo.build_options
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: ShaderInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: ShaderInfo.entry
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: ShaderInfo.include_dirs
      :module: scenario_runner_py

      :type: list[str]


      Writable.


   .. py:method:: ShaderInfo.load_source(self: scenario_runner_py.ShaderInfo, path: str) -> None
      :module: scenario_runner_py

      Load or compile a shader file into ``src``.


   .. py:property:: ShaderInfo.push_constants_size
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: ShaderInfo.shader_type
      :module: scenario_runner_py

      :type: ShaderType


      Writable.


   .. py:property:: ShaderInfo.specialization_constants
      :module: scenario_runner_py

      :type: list[SpecializationConstant]


      Writable.


   .. py:property:: ShaderInfo.src
      :module: scenario_runner_py

      :type: list[int] | None


      Writable.


   .. py:property:: ShaderInfo.stage
      :module: scenario_runner_py

      :type: ShaderStage


      Writable.

.. end-api: [ShaderInfo]

.. begin-api: [VgfInfo]

.. py:class:: VgfInfo()
   :module: scenario_runner_py


   .. py:property:: VgfInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:method:: VgfInfo.load_source(self: scenario_runner_py.VgfInfo, path: str) -> None
      :module: scenario_runner_py

      Load a VGF file into ``src``.


   .. py:property:: VgfInfo.push_constants_size
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: VgfInfo.specialization_constant_maps
      :module: scenario_runner_py

      :type: list[SpecializationConstantMap]


      Writable.

.. end-api: [VgfInfo]

.. begin-api: [GraphConstantInfo]

.. py:class:: GraphConstantInfo()
   :module: scenario_runner_py


   .. py:property:: GraphConstantInfo.data
      :module: scenario_runner_py

      :type: list[int]


      Writable.


   .. py:property:: GraphConstantInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: GraphConstantInfo.dims
      :module: scenario_runner_py

      :type: list[int]


      Writable.


   .. py:property:: GraphConstantInfo.format
      :module: scenario_runner_py

      :type: Format


      Writable.

.. end-api: [GraphConstantInfo]

.. begin-api: [SpecializationConstant]

.. py:class:: SpecializationConstant
   :module: scenario_runner_py


   .. py:method:: SpecializationConstant.from_float32(id: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsFloat | typing.SupportsIndex) -> scenario_runner_py.SpecializationConstant
      :module: scenario_runner_py
      :staticmethod:


   .. py:method:: SpecializationConstant.from_int32(id: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsInt | typing.SupportsIndex) -> scenario_runner_py.SpecializationConstant
      :module: scenario_runner_py
      :staticmethod:


   .. py:method:: SpecializationConstant.from_uint32(id: typing.SupportsInt | typing.SupportsIndex, value: typing.SupportsInt | typing.SupportsIndex) -> scenario_runner_py.SpecializationConstant
      :module: scenario_runner_py
      :staticmethod:


   .. py:property:: SpecializationConstant.id
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [SpecializationConstant]

.. begin-api: [SpecializationConstantMap]

.. py:class:: SpecializationConstantMap()
   :module: scenario_runner_py


   .. py:property:: SpecializationConstantMap.shader_target
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: SpecializationConstantMap.specialization_constants
      :module: scenario_runner_py

      :type: list[SpecializationConstant]


      Writable.

.. end-api: [SpecializationConstantMap]

.. begin-api: [TypedBinding]

.. py:class:: TypedBinding(set: typing.SupportsInt | typing.SupportsIndex, binding: typing.SupportsInt | typing.SupportsIndex, resource: object, descriptor_type: scenario_runner_py.DescriptorType, *, lod: typing.SupportsInt | typing.SupportsIndex | None = None)
   :module: scenario_runner_py


   .. py:property:: TypedBinding.binding
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: TypedBinding.descriptor_type
      :module: scenario_runner_py

      :type: DescriptorType


      Writable.


   .. py:property:: TypedBinding.lod
      :module: scenario_runner_py

      :type: int | None


      Writable.


   .. py:property:: TypedBinding.resource
      :module: scenario_runner_py

      :type: BufferId | ImageId | TensorId


      Writable.


   .. py:property:: TypedBinding.set
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [TypedBinding]

.. begin-api: [ComputeDispatch]

.. py:class:: ComputeDispatch()
   :module: scenario_runner_py


   .. py:property:: ComputeDispatch.group_count_x
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: ComputeDispatch.group_count_y
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: ComputeDispatch.group_count_z
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: ComputeDispatch.profile_name
      :module: scenario_runner_py

      :type: str


      Writable.

.. end-api: [ComputeDispatch]

.. begin-api: [DispatchComputeData]

.. py:class:: DispatchComputeData(shader: scenario_runner_py.ShaderId)
   :module: scenario_runner_py


   .. py:property:: DispatchComputeData.bindings
      :module: scenario_runner_py

      :type: list[TypedBinding]


      Writable.


   .. py:property:: DispatchComputeData.compute_dispatch
      :module: scenario_runner_py

      :type: ComputeDispatch


      Writable.


   .. py:property:: DispatchComputeData.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: DispatchComputeData.implicit_barrier
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: DispatchComputeData.push_data
      :module: scenario_runner_py

      :type: RawDataId | None


      Writable.


   .. py:property:: DispatchComputeData.shader
      :module: scenario_runner_py

      :type: ShaderId


      Writable.

.. end-api: [DispatchComputeData]

.. begin-api: [FragmentAttachment]

.. py:class:: FragmentAttachment(resource: scenario_runner_py.ImageId, *, lod: typing.SupportsInt | typing.SupportsIndex | None = None)
   :module: scenario_runner_py


   .. py:property:: FragmentAttachment.lod
      :module: scenario_runner_py

      :type: int | None


      Writable.


   .. py:property:: FragmentAttachment.resource
      :module: scenario_runner_py

      :type: ImageId


      Writable.

.. end-api: [FragmentAttachment]

.. begin-api: [Extent2D]

.. py:class:: Extent2D(arg0: typing.SupportsInt | typing.SupportsIndex, arg1: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py


   .. py:property:: Extent2D.height
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: Extent2D.width
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [Extent2D]

.. begin-api: [DispatchFragmentData]

.. py:class:: DispatchFragmentData(vertex_shader: scenario_runner_py.ShaderId, fragment_shader: scenario_runner_py.ShaderId)
   :module: scenario_runner_py


   .. py:property:: DispatchFragmentData.bindings
      :module: scenario_runner_py

      :type: list[TypedBinding]


      Writable.


   .. py:property:: DispatchFragmentData.color_attachments
      :module: scenario_runner_py

      :type: list[FragmentAttachment]


      Writable.


   .. py:property:: DispatchFragmentData.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: DispatchFragmentData.fragment_shader
      :module: scenario_runner_py

      :type: ShaderId


      Writable.


   .. py:property:: DispatchFragmentData.implicit_barrier
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: DispatchFragmentData.push_data
      :module: scenario_runner_py

      :type: RawDataId | None


      Writable.


   .. py:property:: DispatchFragmentData.render_extent
      :module: scenario_runner_py

      :type: Extent2D | None


      Writable.


   .. py:property:: DispatchFragmentData.vertex_shader
      :module: scenario_runner_py

      :type: ShaderId


      Writable.

.. end-api: [DispatchFragmentData]

.. begin-api: [PushConstantMap]

.. py:class:: PushConstantMap(push_data: scenario_runner_py.RawDataId, shader_target: str)
   :module: scenario_runner_py


   .. py:property:: PushConstantMap.push_data
      :module: scenario_runner_py

      :type: RawDataId


      Writable.


   .. py:property:: PushConstantMap.shader_target
      :module: scenario_runner_py

      :type: str


      Writable.

.. end-api: [PushConstantMap]

.. begin-api: [ShaderSubstitution]

.. py:class:: ShaderSubstitution(shader: scenario_runner_py.ShaderId, target: str)
   :module: scenario_runner_py


   .. py:property:: ShaderSubstitution.shader
      :module: scenario_runner_py

      :type: ShaderId


      Writable.


   .. py:property:: ShaderSubstitution.target
      :module: scenario_runner_py

      :type: str


      Writable.

.. end-api: [ShaderSubstitution]

.. begin-api: [ExternalDataGraphBinary]

.. py:class:: ExternalDataGraphBinary(binary: scenario_runner_py.RawDataId)
   :module: scenario_runner_py


   .. py:property:: ExternalDataGraphBinary.binary
      :module: scenario_runner_py

      :type: RawDataId


      Writable.


   .. py:property:: ExternalDataGraphBinary.pipeline_identifier
      :module: scenario_runner_py

      :type: list[int] | None


      Writable.


   .. py:property:: ExternalDataGraphBinary.use_application_storage
      :module: scenario_runner_py

      :type: bool


      Writable.

.. end-api: [ExternalDataGraphBinary]

.. begin-api: [DispatchVgfData]

.. py:class:: DispatchVgfData(vgf: scenario_runner_py.VgfId)
   :module: scenario_runner_py


   .. py:property:: DispatchVgfData.bindings
      :module: scenario_runner_py

      :type: list[TypedBinding]


      Writable.


   .. py:property:: DispatchVgfData.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: DispatchVgfData.implicit_barrier
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: DispatchVgfData.push_constants
      :module: scenario_runner_py

      :type: list[PushConstantMap]


      Writable.


   .. py:property:: DispatchVgfData.shader_substitutions
      :module: scenario_runner_py

      :type: list[ShaderSubstitution]


      Writable.


   .. py:property:: DispatchVgfData.vgf
      :module: scenario_runner_py

      :type: VgfId


      Writable.

.. end-api: [DispatchVgfData]

.. begin-api: [DispatchDataGraphBinaryData]

.. py:class:: DispatchDataGraphBinaryData(binary: scenario_runner_py.RawDataId)
   :module: scenario_runner_py

   Describe a standalone precompiled graph dispatch.


   .. py:property:: DispatchDataGraphBinaryData.binary_data
      :module: scenario_runner_py

      :type: ExternalDataGraphBinary


      Writable.


   .. py:property:: DispatchDataGraphBinaryData.bindings
      :module: scenario_runner_py

      :type: list[TypedBinding]


      Writable.


   .. py:property:: DispatchDataGraphBinaryData.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: DispatchDataGraphBinaryData.implicit_barrier
      :module: scenario_runner_py

      :type: bool


      Writable.

.. end-api: [DispatchDataGraphBinaryData]

.. begin-api: [DispatchDataGraphData]

.. py:class:: DispatchDataGraphData(graph_shader: scenario_runner_py.ShaderId)
   :module: scenario_runner_py

   Describe a dispatch of a SPIR-V graph shader.

   Create a dispatch for ``graph_shader``.


   .. py:property:: DispatchDataGraphData.bindings
      :module: scenario_runner_py

      :type: list[TypedBinding]

      Resource bindings used by the graph shader.


      Writable.


   .. py:property:: DispatchDataGraphData.debug_name
      :module: scenario_runner_py

      :type: str

      Optional name used in diagnostics.


      Writable.


   .. py:property:: DispatchDataGraphData.graph_constants
      :module: scenario_runner_py

      :type: list[GraphConstantResourceId]

      Graph constants applied to the dispatch.


      Writable.


   .. py:property:: DispatchDataGraphData.graph_shader
      :module: scenario_runner_py

      :type: ShaderId

      The graph shader to dispatch.


      Writable.


   .. py:property:: DispatchDataGraphData.implicit_barrier
      :module: scenario_runner_py

      :type: bool

      Whether to insert the default synchronization barrier.


      Writable.

.. end-api: [DispatchDataGraphData]

.. begin-api: [DispatchOpticalFlowData]

.. py:class:: DispatchOpticalFlowData(search: scenario_runner_py.TypedBinding, reference: scenario_runner_py.TypedBinding, output: scenario_runner_py.TypedBinding)
   :module: scenario_runner_py


   .. py:property:: DispatchOpticalFlowData.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: DispatchOpticalFlowData.execution_flags
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: DispatchOpticalFlowData.grid_size
      :module: scenario_runner_py

      :type: OpticalFlowGridSize


      Writable.


   .. py:property:: DispatchOpticalFlowData.height
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: DispatchOpticalFlowData.hint_motion_vectors
      :module: scenario_runner_py

      :type: TypedBinding | None


      Writable.


   .. py:property:: DispatchOpticalFlowData.implicit_barrier
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: DispatchOpticalFlowData.mean_flow_l1_norm_hint
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: DispatchOpticalFlowData.output_cost
      :module: scenario_runner_py

      :type: TypedBinding | None


      Writable.


   .. py:property:: DispatchOpticalFlowData.output_image
      :module: scenario_runner_py

      :type: TypedBinding


      Writable.


   .. py:property:: DispatchOpticalFlowData.performance_level
      :module: scenario_runner_py

      :type: OpticalFlowPerformanceLevel


      Writable.


   .. py:property:: DispatchOpticalFlowData.search_image
      :module: scenario_runner_py

      :type: TypedBinding


      Writable.


   .. py:property:: DispatchOpticalFlowData.template_image
      :module: scenario_runner_py

      :type: TypedBinding


      Writable.


   .. py:property:: DispatchOpticalFlowData.width
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [DispatchOpticalFlowData]

.. begin-api: [PipelineBarrierData]

.. py:class:: PipelineBarrierData()
   :module: scenario_runner_py


   .. py:property:: PipelineBarrierData.buffer_barriers
      :module: scenario_runner_py

      :type: list[BufferBarrierId]


      Writable.


   .. py:property:: PipelineBarrierData.image_barriers
      :module: scenario_runner_py

      :type: list[ImageBarrierId]


      Writable.


   .. py:property:: PipelineBarrierData.memory_barriers
      :module: scenario_runner_py

      :type: list[MemoryBarrierId]


      Writable.


   .. py:property:: PipelineBarrierData.tensor_barriers
      :module: scenario_runner_py

      :type: list[TensorBarrierId]


      Writable.

.. end-api: [PipelineBarrierData]

.. begin-api: [FrameBoundaryData]

.. py:class:: FrameBoundaryData()
   :module: scenario_runner_py


   .. py:property:: FrameBoundaryData.buffers
      :module: scenario_runner_py

      :type: list[BufferId]


      Writable.


   .. py:property:: FrameBoundaryData.images
      :module: scenario_runner_py

      :type: list[ImageId]


      Writable.


   .. py:property:: FrameBoundaryData.tensors
      :module: scenario_runner_py

      :type: list[TensorId]


      Writable.

.. end-api: [FrameBoundaryData]

.. begin-api: [BaseBarrierInfo]

.. py:class:: BaseBarrierInfo()
   :module: scenario_runner_py


   .. py:property:: BaseBarrierInfo.debug_name
      :module: scenario_runner_py

      :type: str


      Writable.


   .. py:property:: BaseBarrierInfo.dst_access
      :module: scenario_runner_py

      :type: MemoryAccess


      Writable.


   .. py:property:: BaseBarrierInfo.dst_stages
      :module: scenario_runner_py

      :type: list[PipelineStage]


      Writable.


   .. py:property:: BaseBarrierInfo.src_access
      :module: scenario_runner_py

      :type: MemoryAccess


      Writable.


   .. py:property:: BaseBarrierInfo.src_stages
      :module: scenario_runner_py

      :type: list[PipelineStage]


      Writable.

.. end-api: [BaseBarrierInfo]

.. begin-api: [SubresourceRange]

.. py:class:: SubresourceRange()
   :module: scenario_runner_py


   .. py:property:: SubresourceRange.base_array_layer
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: SubresourceRange.base_mip_level
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: SubresourceRange.layer_count
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: SubresourceRange.level_count
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [SubresourceRange]

.. begin-api: [ImageBarrierInfo]

.. py:class:: ImageBarrierInfo()
   :module: scenario_runner_py


   .. py:property:: ImageBarrierInfo.image
      :module: scenario_runner_py

      :type: ImageId


      Writable.


   .. py:property:: ImageBarrierInfo.new_layout
      :module: scenario_runner_py

      :type: ImageLayout


      Writable.


   .. py:property:: ImageBarrierInfo.old_layout
      :module: scenario_runner_py

      :type: ImageLayout


      Writable.


   .. py:property:: ImageBarrierInfo.range
      :module: scenario_runner_py

      :type: SubresourceRange


      Writable.

.. end-api: [ImageBarrierInfo]

.. begin-api: [BufferBarrierInfo]

.. py:class:: BufferBarrierInfo()
   :module: scenario_runner_py


   .. py:property:: BufferBarrierInfo.buffer
      :module: scenario_runner_py

      :type: BufferId


      Writable.


   .. py:property:: BufferBarrierInfo.offset
      :module: scenario_runner_py

      :type: int


      Writable.


   .. py:property:: BufferBarrierInfo.size
      :module: scenario_runner_py

      :type: int


      Writable.

.. end-api: [BufferBarrierInfo]

.. begin-api: [TensorBarrierInfo]

.. py:class:: TensorBarrierInfo()
   :module: scenario_runner_py


   .. py:property:: TensorBarrierInfo.tensor
      :module: scenario_runner_py

      :type: TensorId


      Writable.

.. end-api: [TensorBarrierInfo]

.. begin-api: [MemoryBarrierInfo]

.. py:class:: MemoryBarrierInfo()
   :module: scenario_runner_py

.. end-api: [MemoryBarrierInfo]

.. begin-api: [ScenarioOptions]

.. py:class:: ScenarioOptions()
   :module: scenario_runner_py

   Configure scenario creation and output.


   .. py:property:: ScenarioOptions.capture_frame
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ScenarioOptions.clear_pipeline_cache
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ScenarioOptions.disabled_extensions
      :module: scenario_runner_py

      :type: list[str]


      Writable.


   .. py:property:: ScenarioOptions.enable_gpu_debug_markers
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ScenarioOptions.enable_pipeline_caching
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ScenarioOptions.enable_robustness_features
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ScenarioOptions.fail_on_pipeline_cache_miss
      :module: scenario_runner_py

      :type: bool


      Writable.


   .. py:property:: ScenarioOptions.graph_profiling_dump_dir
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.


   .. py:property:: ScenarioOptions.neural_debug_database_dump_dir
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.


   .. py:property:: ScenarioOptions.neural_statistics_dump_dir
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.


   .. py:property:: ScenarioOptions.neural_statistics_mode
      :module: scenario_runner_py

      :type: NeuralAcceleratorStatisticsMode


      Writable.


   .. py:property:: ScenarioOptions.perf_counters_path
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.


   .. py:property:: ScenarioOptions.pipeline_cache_path
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.


   .. py:property:: ScenarioOptions.profiling_path
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.


   .. py:property:: ScenarioOptions.session_rams_dump_dir
      :module: scenario_runner_py

      :type: os.PathLike


      Writable.

.. end-api: [ScenarioOptions]

.. begin-api: [BufferId]

.. py:class:: BufferId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: BufferId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [BufferId]

.. begin-api: [ImageId]

.. py:class:: ImageId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: ImageId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [ImageId]

.. begin-api: [TensorId]

.. py:class:: TensorId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: TensorId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [TensorId]

.. begin-api: [ShaderId]

.. py:class:: ShaderId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: ShaderId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [ShaderId]

.. begin-api: [RawDataId]

.. py:class:: RawDataId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: RawDataId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [RawDataId]

.. begin-api: [VgfId]

.. py:class:: VgfId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: VgfId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [VgfId]

.. begin-api: [GraphConstantResourceId]

.. py:class:: GraphConstantResourceId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: GraphConstantResourceId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [GraphConstantResourceId]

.. begin-api: [MemoryGroupId]

.. py:class:: MemoryGroupId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: MemoryGroupId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [MemoryGroupId]

.. begin-api: [ImageBarrierId]

.. py:class:: ImageBarrierId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: ImageBarrierId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [ImageBarrierId]

.. begin-api: [BufferBarrierId]

.. py:class:: BufferBarrierId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: BufferBarrierId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [BufferBarrierId]

.. begin-api: [TensorBarrierId]

.. py:class:: TensorBarrierId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: TensorBarrierId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [TensorBarrierId]

.. begin-api: [MemoryBarrierId]

.. py:class:: MemoryBarrierId
   :module: scenario_runner_py

   An immutable, hashable typed resource ID scoped to a scenario.


   .. py:property:: MemoryBarrierId.value
      :module: scenario_runner_py

      :type: int


      Return the numeric identifier, which is meaningful only within its scenario.


      Read-only.

.. end-api: [MemoryBarrierId]

.. begin-api: [load_scenario]

.. py:function:: load_scenario(scenario_file: os.PathLike | str | bytes, *, work_dir: os.PathLike | str | bytes = Path('.'), output_dir: os.PathLike | str | bytes = Path('.'), options: scenario_runner_py.ScenarioOptions = ScenarioOptions()) -> scenario_runner_py.Scenario
   :module: scenario_runner_py

   Load a JSON scenario and return a ready-to-run :class:`Scenario`.

   Relative resource paths use the scenario file's parent directory unless ``work_dir`` is supplied.

.. end-api: [load_scenario]

.. begin-api: [enums]

.. py:class:: AddressMode(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   ClampBorder

   ClampEdge

   Repeat

   MirroredRepeat

   Unknown


   .. py:attribute:: AddressMode.ClampBorder
      :module: scenario_runner_py
      :value: <AddressMode.ClampBorder: 0>


   .. py:attribute:: AddressMode.ClampEdge
      :module: scenario_runner_py
      :value: <AddressMode.ClampEdge: 1>


   .. py:attribute:: AddressMode.MirroredRepeat
      :module: scenario_runner_py
      :value: <AddressMode.MirroredRepeat: 3>


   .. py:attribute:: AddressMode.Repeat
      :module: scenario_runner_py
      :value: <AddressMode.Repeat: 2>


   .. py:attribute:: AddressMode.Unknown
      :module: scenario_runner_py
      :value: <AddressMode.Unknown: 4>


   .. py:property:: AddressMode.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: AddressMode.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: BorderColor(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   FloatTransparentBlack

   FloatOpaqueBlack

   FloatOpaqueWhite

   IntTransparentBlack

   IntOpaqueBlack

   IntOpaqueWhite

   FloatCustomExt

   IntCustomExt

   Unknown


   .. py:attribute:: BorderColor.FloatCustomExt
      :module: scenario_runner_py
      :value: <BorderColor.FloatCustomExt: 6>


   .. py:attribute:: BorderColor.FloatOpaqueBlack
      :module: scenario_runner_py
      :value: <BorderColor.FloatOpaqueBlack: 1>


   .. py:attribute:: BorderColor.FloatOpaqueWhite
      :module: scenario_runner_py
      :value: <BorderColor.FloatOpaqueWhite: 2>


   .. py:attribute:: BorderColor.FloatTransparentBlack
      :module: scenario_runner_py
      :value: <BorderColor.FloatTransparentBlack: 0>


   .. py:attribute:: BorderColor.IntCustomExt
      :module: scenario_runner_py
      :value: <BorderColor.IntCustomExt: 7>


   .. py:attribute:: BorderColor.IntOpaqueBlack
      :module: scenario_runner_py
      :value: <BorderColor.IntOpaqueBlack: 4>


   .. py:attribute:: BorderColor.IntOpaqueWhite
      :module: scenario_runner_py
      :value: <BorderColor.IntOpaqueWhite: 5>


   .. py:attribute:: BorderColor.IntTransparentBlack
      :module: scenario_runner_py
      :value: <BorderColor.IntTransparentBlack: 3>


   .. py:attribute:: BorderColor.Unknown
      :module: scenario_runner_py
      :value: <BorderColor.Unknown: 8>


   .. py:property:: BorderColor.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: BorderColor.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: DescriptorType(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   StorageBuffer

   StorageImage

   CombinedImageSampler

   TensorArm


   .. py:attribute:: DescriptorType.CombinedImageSampler
      :module: scenario_runner_py
      :value: <DescriptorType.CombinedImageSampler: 1>


   .. py:attribute:: DescriptorType.StorageBuffer
      :module: scenario_runner_py
      :value: <DescriptorType.StorageBuffer: 7>


   .. py:attribute:: DescriptorType.StorageImage
      :module: scenario_runner_py
      :value: <DescriptorType.StorageImage: 3>


   .. py:attribute:: DescriptorType.TensorArm
      :module: scenario_runner_py
      :value: <DescriptorType.TensorArm: 1000460000>


   .. py:property:: DescriptorType.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: DescriptorType.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: FilterMode(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Linear

   Nearest

   Unknown


   .. py:attribute:: FilterMode.Linear
      :module: scenario_runner_py
      :value: <FilterMode.Linear: 0>


   .. py:attribute:: FilterMode.Nearest
      :module: scenario_runner_py
      :value: <FilterMode.Nearest: 1>


   .. py:attribute:: FilterMode.Unknown
      :module: scenario_runner_py
      :value: <FilterMode.Unknown: 2>


   .. py:property:: FilterMode.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: FilterMode.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: ImageLayout(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   General

   TensorAliasing

   Undefined

   Unknown


   .. py:attribute:: ImageLayout.General
      :module: scenario_runner_py
      :value: <ImageLayout.General: 0>


   .. py:attribute:: ImageLayout.TensorAliasing
      :module: scenario_runner_py
      :value: <ImageLayout.TensorAliasing: 1>


   .. py:attribute:: ImageLayout.Undefined
      :module: scenario_runner_py
      :value: <ImageLayout.Undefined: 2>


   .. py:attribute:: ImageLayout.Unknown
      :module: scenario_runner_py
      :value: <ImageLayout.Unknown: 3>


   .. py:property:: ImageLayout.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: ImageLayout.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: MemoryAccess(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   ComputeShaderWrite

   MemoryWrite

   MemoryRead

   GraphWrite

   GraphRead

   ComputeShaderRead

   Unknown


   .. py:attribute:: MemoryAccess.ComputeShaderRead
      :module: scenario_runner_py
      :value: <MemoryAccess.ComputeShaderRead: 5>


   .. py:attribute:: MemoryAccess.ComputeShaderWrite
      :module: scenario_runner_py
      :value: <MemoryAccess.ComputeShaderWrite: 0>


   .. py:attribute:: MemoryAccess.GraphRead
      :module: scenario_runner_py
      :value: <MemoryAccess.GraphRead: 4>


   .. py:attribute:: MemoryAccess.GraphWrite
      :module: scenario_runner_py
      :value: <MemoryAccess.GraphWrite: 3>


   .. py:attribute:: MemoryAccess.MemoryRead
      :module: scenario_runner_py
      :value: <MemoryAccess.MemoryRead: 2>


   .. py:attribute:: MemoryAccess.MemoryWrite
      :module: scenario_runner_py
      :value: <MemoryAccess.MemoryWrite: 1>


   .. py:attribute:: MemoryAccess.Unknown
      :module: scenario_runner_py
      :value: <MemoryAccess.Unknown: 6>


   .. py:property:: MemoryAccess.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: MemoryAccess.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: NeuralAcceleratorStatisticsMode(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Disabled

   Statistics0

   Statistics1


   .. py:attribute:: NeuralAcceleratorStatisticsMode.Disabled
      :module: scenario_runner_py
      :value: <NeuralAcceleratorStatisticsMode.Disabled: 0>


   .. py:attribute:: NeuralAcceleratorStatisticsMode.Statistics0
      :module: scenario_runner_py
      :value: <NeuralAcceleratorStatisticsMode.Statistics0: 1>


   .. py:attribute:: NeuralAcceleratorStatisticsMode.Statistics1
      :module: scenario_runner_py
      :value: <NeuralAcceleratorStatisticsMode.Statistics1: 2>


   .. py:property:: NeuralAcceleratorStatisticsMode.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: NeuralAcceleratorStatisticsMode.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: OpticalFlowGridSize(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Invalid

   OneByOne

   TwoByTwo

   FourByFour

   EightByEight


   .. py:attribute:: OpticalFlowGridSize.EightByEight
      :module: scenario_runner_py
      :value: <OpticalFlowGridSize.EightByEight: 3>


   .. py:attribute:: OpticalFlowGridSize.FourByFour
      :module: scenario_runner_py
      :value: <OpticalFlowGridSize.FourByFour: 2>


   .. py:attribute:: OpticalFlowGridSize.Invalid
      :module: scenario_runner_py
      :value: <OpticalFlowGridSize.Invalid: 4294967295>


   .. py:attribute:: OpticalFlowGridSize.OneByOne
      :module: scenario_runner_py
      :value: <OpticalFlowGridSize.OneByOne: 0>


   .. py:attribute:: OpticalFlowGridSize.TwoByTwo
      :module: scenario_runner_py
      :value: <OpticalFlowGridSize.TwoByTwo: 1>


   .. py:property:: OpticalFlowGridSize.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: OpticalFlowGridSize.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: OpticalFlowPerformanceLevel(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Invalid

   Unknown

   Slow

   Medium

   Fast


   .. py:attribute:: OpticalFlowPerformanceLevel.Fast
      :module: scenario_runner_py
      :value: <OpticalFlowPerformanceLevel.Fast: 3>


   .. py:attribute:: OpticalFlowPerformanceLevel.Invalid
      :module: scenario_runner_py
      :value: <OpticalFlowPerformanceLevel.Invalid: 4294967295>


   .. py:attribute:: OpticalFlowPerformanceLevel.Medium
      :module: scenario_runner_py
      :value: <OpticalFlowPerformanceLevel.Medium: 2>


   .. py:attribute:: OpticalFlowPerformanceLevel.Slow
      :module: scenario_runner_py
      :value: <OpticalFlowPerformanceLevel.Slow: 1>


   .. py:attribute:: OpticalFlowPerformanceLevel.Unknown
      :module: scenario_runner_py
      :value: <OpticalFlowPerformanceLevel.Unknown: 0>


   .. py:property:: OpticalFlowPerformanceLevel.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: OpticalFlowPerformanceLevel.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: PipelineStage(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Graph

   Compute

   Graphics

   All

   Unknown


   .. py:attribute:: PipelineStage.All
      :module: scenario_runner_py
      :value: <PipelineStage.All: 3>


   .. py:attribute:: PipelineStage.Compute
      :module: scenario_runner_py
      :value: <PipelineStage.Compute: 1>


   .. py:attribute:: PipelineStage.Graph
      :module: scenario_runner_py
      :value: <PipelineStage.Graph: 0>


   .. py:attribute:: PipelineStage.Graphics
      :module: scenario_runner_py
      :value: <PipelineStage.Graphics: 2>


   .. py:attribute:: PipelineStage.Unknown
      :module: scenario_runner_py
      :value: <PipelineStage.Unknown: 4>


   .. py:property:: PipelineStage.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: PipelineStage.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: ShaderStage(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Unknown

   Compute

   Vertex

   Fragment


   .. py:attribute:: ShaderStage.Compute
      :module: scenario_runner_py
      :value: <ShaderStage.Compute: 1>


   .. py:attribute:: ShaderStage.Fragment
      :module: scenario_runner_py
      :value: <ShaderStage.Fragment: 3>


   .. py:attribute:: ShaderStage.Unknown
      :module: scenario_runner_py
      :value: <ShaderStage.Unknown: 0>


   .. py:attribute:: ShaderStage.Vertex
      :module: scenario_runner_py
      :value: <ShaderStage.Vertex: 2>


   .. py:property:: ShaderStage.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: ShaderStage.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: ShaderType(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Unknown

   SpirV

   Glsl

   Hlsl


   .. py:attribute:: ShaderType.Glsl
      :module: scenario_runner_py
      :value: <ShaderType.Glsl: 2>


   .. py:attribute:: ShaderType.Hlsl
      :module: scenario_runner_py
      :value: <ShaderType.Hlsl: 3>


   .. py:attribute:: ShaderType.SpirV
      :module: scenario_runner_py
      :value: <ShaderType.SpirV: 1>


   .. py:attribute:: ShaderType.Unknown
      :module: scenario_runner_py
      :value: <ShaderType.Unknown: 0>


   .. py:property:: ShaderType.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: ShaderType.value
      :module: scenario_runner_py

      :type: int



      Read-only.


.. py:class:: Tiling(value: typing.SupportsInt | typing.SupportsIndex)
   :module: scenario_runner_py

   Members:

   Optimal

   Linear

   Unknown


   .. py:attribute:: Tiling.Linear
      :module: scenario_runner_py
      :value: <Tiling.Linear: 1>


   .. py:attribute:: Tiling.Optimal
      :module: scenario_runner_py
      :value: <Tiling.Optimal: 0>


   .. py:attribute:: Tiling.Unknown
      :module: scenario_runner_py
      :value: <Tiling.Unknown: 2>


   .. py:property:: Tiling.name
      :module: scenario_runner_py

      :type: str



      Read-only.


   .. py:property:: Tiling.value
      :module: scenario_runner_py

      :type: int



      Read-only.

.. end-api: [enums]

.. begin-api: [resource_creation]

.. py:method:: ScenarioBuilder.add_buffer
   :module: scenario_runner_py

   Overloaded function.

   ``1. add_buffer(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.BufferInfo) -> scenario_runner_py.BufferId``

   Register a buffer and return its ID.

   ``2. add_buffer(self: scenario_runner_py.ScenarioBuilder, size: typing.SupportsInt | typing.SupportsIndex, *, debug_name: str = '', memory_offset: typing.SupportsInt | typing.SupportsIndex = 0) -> scenario_runner_py.BufferId``

   Register a buffer and return its ID.


.. py:method:: ScenarioBuilder.add_image
   :module: scenario_runner_py

   Overloaded function.

   ``1. add_image(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.ImageInfo) -> scenario_runner_py.ImageId``

   Register an image and return its ID.

   ``2. add_image(self: scenario_runner_py.ScenarioBuilder, shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], format: scenario_runner_py.Format, *, debug_name: str = '', is_input: bool = False, is_sampled: bool = False, is_storage: bool = False, is_color_attachment: bool = False, mips: typing.SupportsInt | typing.SupportsIndex = 1, tiling: scenario_runner_py.Tiling | None = None, memory_offset: typing.SupportsInt | typing.SupportsIndex = 0) -> scenario_runner_py.ImageId``

   Register an image and return its ID. The convenience overload uses ``format`` as both source and target format.


.. py:method:: ScenarioBuilder.add_tensor
   :module: scenario_runner_py

   Overloaded function.

   ``1. add_tensor(self: scenario_runner_py.ScenarioBuilder, info: scenario_runner_py.TensorInfo) -> scenario_runner_py.TensorId``

   Register a tensor and return its ID.

   ``2. add_tensor(self: scenario_runner_py.ScenarioBuilder, shape: collections.abc.Sequence[typing.SupportsInt | typing.SupportsIndex], format: scenario_runner_py.Format, *, debug_name: str = '', sparsity_dimension: typing.SupportsInt | typing.SupportsIndex = -1, descriptor_buffer_capture_replay: bool = False, tiling: scenario_runner_py.Tiling = <Tiling.Linear: 1>, memory_offset: typing.SupportsInt | typing.SupportsIndex = 0) -> scenario_runner_py.TensorId``

   Register a tensor and return its ID.

.. end-api: [resource_creation]
