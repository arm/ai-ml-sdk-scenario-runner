Scenario Runner architecture
============================

A scenario describes resources and an ordered sequence of commands. Scenario
Runner (SR) turns that description into Vulkan® resources and pipelines, records
the commands, submits them to a queue and waits for completion. It supports file-based
execution through the command-line tool and embedded execution through C++ and
Python. Both use the same native construction and execution machinery.

Construction and execution are separate. A built ``Scenario`` retains its
resources and pipelines, allowing an application to upload new inputs, execute
again and download results without rebuilding the scenario. The public C++ API
is experimental; its compatibility status and examples are described in
:doc:`api`, with Python entry points in :doc:`python_api`.

.. contents::
   :local:
   :depth: 2

.. _scenario-construction:

Construction paths
------------------

There are two ways to construct a scenario:

1. The command-line tool passes a JSON scenario to ``ScenarioJsonFactory::make()``.
   An embedded application can call the same factory directly. The factory parses
   the description, resolves shapes and paths, and populates a native builder with
   resources, memory groups, initial data, output destinations and commands.

2. An application can use ``createScenarioBuilder()`` to register resources
   and append commands directly. The Python bindings wrap the same native builder
   and scenario interfaces; they do not introduce a separate execution backend.
   Programmatic construction does not require an intermediate JSON file.

The JSON factory performs shape resolution before populating the builder. Its
SPIR-V™ graph shape pass can use concrete interface tensor shapes, propagate
inferred shapes and retain rewritten modules and VGF resource-shape overrides.
This is a JSON construction step, not automatic shape inference on every call
to the programmatic builder. Programmatic descriptions must supply the resource
metadata and module inputs needed by their preparation path.

Resources must be registered before commands that refer to them. Registration
returns typed IDs, such as ``BufferId`` or ``TensorId``, which identify resources
within the scenario. A binding connects a resource ID to a descriptor location;
the ID itself is not a descriptor set or binding number. JSON UIDs provide a
separate naming layer: the factory resolves them to typed IDs, and a built
JSON scenario exposes typed lookups for buffer, image and tensor UIDs.

``build()`` consumes the builder's definition and returns a ``Scenario``.
The builder becomes immutable when its build data is taken, before runtime
construction completes; it cannot be reused even if that construction fails.
The scenario owns the transferred resource definitions and command sequence.

.. uml:: puml/scenario-construction.puml
   :align: center
   :caption: JSON and programmatic construction converge on one built scenario.
   :alt: Sequence showing JSON factory and direct builder paths followed by upload, run and download.

Building the scenario establishes its device context, creates runtime resources,
initializes their data and prepares the command sequence and pipelines. These
steps happen before ``build()`` or the JSON factory returns. Scenario dispatch
command recording and submission are handled by ``Compute`` during ``run()``;
initial resource transfers can already submit work during construction.

For JSON input, resource ``src`` fields supply initial file data and ``dst``
fields identify output files. Resource metadata, such as image dimensions,
determines how that data is interpreted. Relative input paths are resolved from
the scenario file's directory by default; the factory also accepts explicit
working and output directories. Output resources are saved after a non-dry
``run()`` completes its requested iterations, rather than after every individual
dispatch. See :doc:`json` for the description format and :doc:`api` for factory
path handling.

Applications can also call typed ``upload()`` and ``download()`` operations on
the built scenario. Upload copies the supplied data before returning; download
returns owning data. Retained typed IDs let a programmatic caller transfer data
without string lookup. File input/output and explicit transfers can coexist.

.. _scenario-responsibilities:

Internal responsibilities
-------------------------

``ScenarioImpl`` orchestrates construction and execution behind the public
``Scenario`` interface. Its managers separate declarative resource information,
concrete runtime objects and allocation grouping. This lets both construction
paths use the same resource and command setup.

.. uml:: puml/scenario-components.puml
   :align: center
   :caption: Scenario-owned state and the command execution boundary.
   :alt: Scenario owns the context and resource managers and Compute, which owns pipelines and commands and uses the context.

``Context`` selects the physical device and creates the Vulkan® environment,
including the device, queues and enabled features. Command requirements and
options determine its setup before runtime resource creation.
``ResourceManager`` retains definitions indexed by typed ID, ``DataManager``
retains concrete resources and barrier objects, and ``GroupManager`` tracks
allocation groups and their shared memory managers.

``Compute`` owns the registered execution commands, descriptors, command-buffer
and submission state, and ``Pipeline`` wrappers. Each pipeline owns the Vulkan®
objects for its dispatch, including graph sessions and their memory when needed.
The name ``Compute`` refers to this execution orchestrator, which also handles
graph, fragment and optical-flow work.

.. _scenario-workload-library-boundary:

Workload Library boundary
^^^^^^^^^^^^^^^^^^^^^^^^^

Scenario Runner currently creates dispatch pipelines through ``Compute`` and
registers them there for execution. It also links Workload Library and contains an
internal bridge for recording a prepared workload into its command stream, but
the current dispatch constructors neither create workloads through the bridge
nor call its registration entry point. The bridge provides these roles:

- ``WorkloadRuntime`` wraps the existing scenario context and queue.
- ``WorkloadExecution`` owns a workload, session and prepared binding snapshot,
  retains its runtime and records into a supplied command buffer.
- ``WorkloadBindings`` binds existing ``DataManager`` resources through borrowed
  Vulkan® handles. Scenario Runner retains the resources and image-layout tracking.

``Compute::registerWorkloadExecution()`` can retain such an execution alongside
other registered commands, including an optional trailing implicit barrier.
Its recording support leaves submission and completion waits with Scenario Runner.

.. _scenario-resource-model:

Resources, data and ownership
-----------------------------

Resource construction separates three things: a definition describes a resource object,
a runtime wrapper owns its Vulkan® handles and tracks its state, and a resource
memory manager owns the backing allocation. Typed resource IDs connect the first
two. Two IDs can identify distinct objects that share allocation storage; they
do not become the same resource or descriptor binding.

Construction order
^^^^^^^^^^^^^^^^^^

``ScenarioImpl::setupResources()`` first creates runtime wrappers for the declared
buffers, images, tensors and raw data, and retains the VGF views. It then expands
resources needed by VGF dispatches and adds their memory-group relationships.
Only after that expansion does it finalize the groups.

Buffer and image setup precedes tensor setup. Setup creates the Vulkan® objects,
collects memory requirements and, for images, records layout information needed
by aliasing tensors. Runtime barrier objects are then created. Resource allocation
and memory binding follow, including image and tensor views. Initial data is
uploaded after these allocations exist; pipeline and command setup follows the
initialization phase. Constructing wrappers alone does not allocate every
resource's backing memory.

``Buffer`` represents untyped storage bytes. ``Image`` combines image metadata,
image/view objects, optional sampler state and tracked layout. ``Tensor`` combines
shape, element format, tiling and optional strides with a tensor object and view.
``RawData`` instead retains a mapped source file and provides CPU-side byte views,
for example for graph binaries, or a NumPy interpretation for push data. It is not a separately allocated
shader-visible buffer.

Graph constants are another kind of input: ``GraphConstantInfo`` owns its bytes
and tensor metadata as part of the resource definition. Shader modules and VGF
views are also construction inputs, separate from the buffers, images and tensors
that carry runtime input/output data.

.. _scenario-memory-groups:

Memory groups and aliasing
^^^^^^^^^^^^^^^^^^^^^^^^^^

A memory group associates buffer, image or tensor IDs with one shared
``ResourceMemoryManager``. The group manager and participating wrappers retain
shared references to that manager; the allocation is not exclusively owned by
one resource. A resource outside a group receives its own manager.

Each resource contributes its required allocation extent and permitted memory
types. The manager takes the maximum covered end, accounting for offsets, and
intersects the memory-type masks. Allocation therefore happens after all members
have contributed requirements. It is not the sum of their logical data sizes.
The first allocation request creates device-local memory and host-visible,
coherent staging storage; later members bind to the same backing device memory.

.. uml:: puml/scenario-resources.puml
   :align: center
   :caption: Distinct buffer objects can share allocation storage through a memory group.
   :alt: Definitions create two runtime buffers whose shared memory manager owns backing and staging allocations; each buffer has its own offset.

The diagram uses two buffers as a simple example. Their declared offsets can
select overlapping or disjoint regions. ``Buffer::setup()`` checks offset
alignment against the buffer's Vulkan® memory requirements. Images bind at their
declared memory offset; tensors bind at their declared offset plus the image
subresource offset recorded by the shared manager, when present. Allocation size,
alignment, compatible memory types and resource layout all matter; equal logical
byte counts alone do not establish a valid alias.

An image in a multi-member memory group must have a single mip level;
``Image::setup()`` rejects aliased images with multiple mips. The mip-chain
upload support described below applies to images outside these shared groups.

For linear image/tensor sharing, image subresource pitches supply the tensor's
strides. Tensor setup checks that its innermost dimension matches the image's
component count and incorporates row pitch and element size. The tensor keeps its
logical shape, but its strides must account for padding in the shared image
memory. Optimal tiling follows a different representation and requires the
appropriate device support and layout handling. Image setup respects explicit
tiling and otherwise selects supported tiling based on the requested operations.
Do not infer cross-resource synchronization from shared allocation ownership.

The JSON interface expresses groups with ``memory_group.id`` and optional
``offset``; the programmatic builder creates groups and registers their members
by typed ID. Follow the single-source initialization rule in
:ref:`scenario-usage-memory-aliasing`:
provide a ``src`` file for only one member of a shared JSON group. This avoids
independent initial uploads overwriting the same storage. It is an input contract,
not a guarantee that every invalid alias combination is rejected before Vulkan®
object creation. That section contains image/tensor and offset examples.

.. _scenario-resource-transfers:

Initialization and explicit transfers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The JSON factory reads initial buffer, tensor and image file data into owning
initialization records. Scenario construction uploads those records after memory
binding. For unaliased JSON buffers and tensors without ``src``, the factory
fills them with zero. For images without ``src``, initialization either uploads
zeros for an unaliased image or establishes the layout of an aliased image
without overwriting its peer's storage. These JSON initialization policies are
separate from the programmatic builder: registering a buffer or tensor does not
provide its application input. Upload that data explicitly before consuming it.

Explicit transfers operate on existing resources. Buffer upload checks the byte
count. Tensor upload checks logical shape, optional format and logical byte size;
image upload checks shape, optional format, mip information and data size. Image
uploads containing fewer mip levels than the resource can generate missing levels
using linear-filtered blits, provided the format and tiling support them. Full
mip-chain input avoids that generation requirement.

C++ upload views borrow the caller's bytes for the synchronous call; the
scenario does not retain that pointer. Download returns owning data: buffers as
bytes, tensors as logical-sized bytes plus metadata, and images as packed
base-mip data plus metadata. Tensor download can repack a padded four-dimensional
layout with four explicit strides; otherwise it copies a contiguous logical-sized
region. It does not generally convert arbitrary tiled tensor storage. Storage
pitches and allocation sizes should not substitute for logical transfer sizes.

Python upload requires a C-contiguous NumPy array and passes its storage through
the native transfer call. It copies data into the resource rather than retaining
the array as device storage. Python download creates and fills a new owning NumPy
array; it does not expose mapped device memory. Buffer results use byte elements,
while image and tensor result types derive from their formats. Shape and byte
checks apply, but the upload wrapper does not supply a format for native format
validation. See :doc:`python_api` for the Python-facing transfer interface.

File ``dst`` output is implemented by downloading the resource and serializing
its contents after execution. Explicit download is available independently of
whether a resource has a file destination. These synchronous interfaces do not
establish support for concurrent transfers or overlapping scenario executions.

.. _scenario-resource-backing:

VGF and shader backing storage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``ShaderInfo::src`` retains shared immutable SPIR-V™ words; copying a shader
definition shares that backing storage. ``VgfInfo::src`` similarly retains a
shared const ``VgfView``. A file-loaded VGF view owns its memory mapping and the
decoders that borrow from it. Views of decoded module or constant data remain
tied to that backing storage; creating a decoder does not copy all file bytes.
The scenario's definitions and runtime VGF entries retain the shared view, so
the caller need not keep its original shared-pointer variable alive after build.

VGF resource expansion adds internal runtime resources before groups are
finalized. It records the mapping from VGF model-resource table indices to newly
created SR resource IDs. Alias groups from the file are mapped into runtime groups,
including external resources selected by each dispatch's bindings. VGF model
interface binding IDs, model-resource indices, SR typed IDs and shader descriptor
locations are distinct identities. Their dispatch mapping is handled during VGF
preparation, not by treating their numeric values as interchangeable.

.. raw:: html

   <style>
   table.scenario-ownership td, table.scenario-ownership th { white-space: normal; overflow-wrap: anywhere; }
   </style>

.. list-table:: Resource ownership at the API boundary
   :header-rows: 1
   :widths: 28 72
   :class: scenario-ownership

   * - State
     - Owner and lifetime
   * - Built state
     - Runtime resources, allocations, pipelines and sessions remain available through the scenario's synchronous runs and transfers.
   * - Shader and VGF sources
     - Definitions retain shared backing objects; a VGF view owns its mapping and borrowed-data decoders.
   * - Constants and raw data
     - Graph constant definitions own byte vectors. RawData owns a mapping; its byte views borrow that mapping.
   * - Upload arguments
     - C++ byte views and Python array storage must remain valid during the call; upload copies before returning.
   * - Download results
     - C++ data vectors and Python arrays own their copied results independently of the scenario.

.. _scenario-dispatch-preparation:

Dispatch preparation
--------------------

``ScenarioImpl::setupRuntimeCommands()`` visits the scenario command sequence
and selects a preparation path for each dispatch. ``Compute`` owns the resulting
``Pipeline`` objects and registered commands. Pipeline creation determines code,
layouts and session state; command registration connects those objects to the
concrete resources. Neither step submits the dispatch to the GPU.

.. list-table:: Dispatch inputs and prepared state
   :header-rows: 1
   :widths: 28 32 40
   :class: scenario-ownership

   * - Command
     - Input
     - Preparation and requirements
   * - Compute
     - Compute shader and typed bindings.
     - Compute pipeline and dispatch geometry; requires suitable compute execution support.
   * - Fragment
     - Vertex and fragment shaders, bindings and color attachments.
     - Graphics pipeline and rendering state; graphics queue and dynamic rendering support. Attachment extents must agree.
   * - VGF
     - VGF view, external bindings and code overrides.
     - Ordered graph/compute segment preparation; requirements follow the selected segment paths.
   * - Graph module
     - SPIR-V™ graph, tensor/image bindings and graph constants.
     - Graph pipeline, session and session memory; data-graph and resource capabilities required.
   * - Graph binary
     - Mapped binary, optional pipeline identifier and tensor/image bindings.
     - Binary-backed graph pipeline and session; requires data-graph binary support and compatible binary input.
   * - Optical flow
     - Search, template and output images; optional hints/cost and algorithm options.
     - Optical-flow graph pipeline and session; optical-flow capability and valid image configuration required.

The paths share Vulkan® resource management but have different construction
inputs. Graph module and graph binary commands accept tensor and image resources,
not arbitrary buffer bindings. Fragment commands use vertex and fragment stages;
attachment images must have consistent mip-adjusted extents, or a render extent
must be supplied when there are no attachments. Device and queue requirements
are derived from the selected commands before device creation. Build-time
compiler availability is separate from device support for those commands.

Shader loading through ``readShaderCode()`` accepts SPIR-V™ words directly or
compiles GLSL through glslang using the shader's compute, vertex or fragment stage. HLSL uses the
optional compute compiler backend. Embedded VGF compute code follows its own
selection path below. Source compilation and graph pipeline creation are separate
operations; GLSL text is not a graph module input.

.. _scenario-vgf-preparation:

VGF sequence expansion and bindings
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

One VGF dispatch can expand into several prepared pipelines and dispatches.
``createVgfPipeline()`` iterates segment indices in sequence order. Each segment
refers to a module; the module supplies its kind, entry point and code, while
the segment supplies invocation-specific descriptor mapping and dispatch data.
A module can therefore be referenced by more than one segment. Segment order
is preserved when those dispatches are registered with ``Compute``.

.. uml:: puml/scenario-vgf-dispatch.puml
   :align: center
   :caption: Each VGF segment resolves resources and selects graph or compute preparation.
   :alt: Flow iterating VGF segments, mapping model resources to runtime IDs, selecting graph or compute pipeline preparation and appending commands in sequence order.

Binding resolution first builds a map keyed by VGF model-resource table index.
It starts with the internal resources created during resource expansion.
External dispatch bindings use VGF model-interface binding IDs to identify
inputs or outputs and associate them with SR typed resource IDs. The resolver
checks the selected resource against the interface metadata and replaces the
corresponding entry in that map. An external binding is not interpreted as a
segment-local descriptor location.

For each segment descriptor entry, ``VgfView::resolveBindings()`` looks up its
model-resource index in the completed map and emits a ``TypedBinding`` with the
segment's descriptor set, binding number and descriptor type, plus the resolved
SR resource ID. A missing required resource fails resolution. This two-stage
mapping lets a model interface and different shader segments use different
binding locations for the same resource. The resource model's internal IDs do
not have to equal any of those locations. See :ref:`scenario-resource-backing`
for the backing-storage and intermediate-resource boundary.

Compute segments select a module-name-matched shader substitution first. The
substitution can replace embedded code as well as fill a code-free module.
Without a substitution, the path prefers embedded SPIR-V™, then GLSL, then HLSL;
missing code or unavailable HLSL support fails preparation. GLSL/HLSL compilation
here produces a compute shader. The VGF resource's push-constant range size and
module-targeted specialization constants are applied to the selected shader
metadata. Push payloads are separately selected by module name from dispatch
push mappings and read from retained raw data; declaring a range does not supply
its bytes. Compute dispatch dimensions come from the segment.

For graph segments, a per-segment external binary selection takes precedence over
the graph module path. A segment explicitly marked as an external binary requires
that selection. Selections must name valid graph segments. Without a selection,
the graph path creates a shader module from the VGF graph SPIR-V™ and supplies
its entry point, concrete resource descriptions and graph constants. VGF constant
bindings connect graph constant IDs to payload records, including format, shape
and optional sparsity metadata. These are pipeline-creation inputs, separate
from runtime descriptor resources and session working memory. Standalone graph
commands instead assign sequential graph constant IDs in the order of the
referenced constant definitions.

.. _scenario-pipeline-session-state:

Pipeline layouts, sessions and binaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

``Pipeline`` creates descriptor set layouts from typed bindings and a pipeline
layout from those layouts. Compute and graphics paths also use declared push
ranges and specialization inputs when creating their shader stages. Registered
commands refer to the resulting pipeline and concrete descriptor resources.
Graph resource descriptions additionally carry the tensor shape, format, strides
or image-layout metadata needed for graph pipeline creation.

A graph ``Pipeline`` owns its Vulkan® graph pipeline session. Session setup queries
memory bind points, allocates memory for nonzero memory requirements and binds
those allocations to the session. These allocations belong to the pipeline's
session state; they are separate from the resource-group allocations described
above and from graph constant payloads. Ordinary session allocations request
device-local memory; the optional session-memory dump mode instead requests
host-visible, coherent storage. Optical-flow sessions also request their
optical-flow cache state through session creation flags.

Graph binary preparation reads a ``RawData`` mapping rather than a shader module.
It checks the public binary header and device-required data alignment, passes
an optional pipeline identifier and ``useApplicationStorage`` selection, and
checks pipeline creation success. The scenario retains the mapped source through
its raw-data resource, including when application storage is selected. Binary
validation does not establish that a binary is executable on every device.
The binary path creates descriptor/pipeline layouts and a graph session like
the module path, but bypasses Vulkan® pipeline caches.

Neural statistics collection is configured while creating the graph pipeline
and session. The collection request enables the pipeline statistics creation
input; the selected statistics mode is supplied to session creation. Both graph
module and graph binary preparation, including VGF graph segments, pass these
options to the common session setup. The required feature/extension support is
checked while establishing the context.

Statistics use their own session bind point. For a nonzero statistics memory
requirement, the pipeline allocates host-visible, coherent memory and retains
that storage alongside the ordinary session allocations.
This storage is not a public scenario tensor or a graph constant. Pipeline and
session ownership keep it available for later result inspection. Debug database
extraction is distinct from enabling runtime statistics collection;
:ref:`scenario-usage-neural-statistics` describes the options.

.. _scenario-pipeline-cache:

Pipeline-cache boundary
^^^^^^^^^^^^^^^^^^^^^^^

When enabled, the scenario creates a shared ``PipelineCache`` wrapper during
command setup and supplies it to eligible pipeline constructors. The wrapper
can load cache bytes from a file, checks the cache header and device identifiers,
and can clear the saved cache before construction. It saves updated cache data
after setup. Its fail-on-miss policy is active when a valid existing cache was
loaded; missing or empty cache files do not by themselves prove a cache miss.

Standalone graph binary commands and VGF segments with external binary selections
bypass this cache and reject the fail-on-pipeline-cache-miss option. Retaining
built ``Pipeline`` objects across scenario runs is separate from using a Vulkan®
pipeline cache.

.. _scenario-execution:

Recording, submission and repeated runs
---------------------------------------

``Scenario::run()`` executes one iteration; ``run(repeatCount, dryRun)`` requires
a positive repeat count and executes iterations sequentially. Each non-dry
iteration records the retained CPU command list into fresh Vulkan® command
buffers. ``Compute`` owns the command pool, queue reference and fence. It resets
the timestamp query pool, records pipeline/descriptor bindings, push constants,
dispatches and barriers in command order, submits the final buffer and waits for
its fence before returning. Frame-boundary commands can introduce additional
submissions and waits during recording, as described below.

.. uml:: puml/scenario-execution.puml
   :caption: A built scenario records and waits for each execution iteration.
   :alt: Application calls run; Scenario resets execution state, Compute records ordered commands, submits to Vulkan® and waits, then Scenario collects profiling and saves final results.

A compute dispatch records ``dispatch``; graph and optical-flow dispatches record
``dispatchDataGraphARM`` using their prepared graph sessions. Fragment dispatches
record dynamic rendering and a full-screen triangle draw. VGF segments have
already expanded into individual dispatches during construction.

Dependencies and image layouts
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Command order alone does not specify all Vulkan® memory dependencies. Dispatches enable implicit barriers by default, registering a trailing global
memory barrier with
all-command source/destination stages and memory read/write access. This barrier
does not perform image-layout transitions. An explicit pipeline-barrier command
can combine global memory barriers, buffer ranges, image subresource ranges and
tensor barriers, with the configured source/destination access and stage scopes.
``Compute`` records them through ``pipelineBarrier2``; tensor barriers are supplied
through the tensor dependency extension structure. Image barriers additionally
carry the declared old/new layouts. Correct dependencies and alias usage still
depend on the scenario's resource accesses and layout choices.

Initial uploads perform their resource transitions during construction or an
explicit upload. Before a non-dry iteration, a scenario containing aliased
optimal tensors also runs its alias-layout preparation: it checks group tiling
consistency, identifies resources used by commands, and requests appropriate
image transitions for tensor aliases or image use. This is separate from the
trailing global memory barrier. Sharing an allocation does not itself establish
an execution dependency between its aliases.

Frames and reuse
^^^^^^^^^^^^^^^^

When the frame-boundary extension is enabled, a frame-boundary command retains
the referenced image, buffer and tensor handles as frame metadata. Encountering
it during recording ends the current command buffer, submits with that metadata,
waits for the fence, resets the fence and begins another buffer. The remaining
commands continue there; the final buffer is submitted after recording finishes.
If frame-boundary commands are present, the first recording also submits an
initialization boundary before visiting the command list. Its frame ID starts
at zero, and IDs advance at each boundary across later iterations and calls.
These markers can divide a run into several frames; a frame is not necessarily
one repeat iteration. Without extension support, setup warns and skips the
frame-boundary command.

After a completed iteration, the next iteration or later ``run`` call resets the
command pool and fence state. For images whose definitions explicitly select
optimal tiling, it also resets the wrapper's tracked layout to undefined; this
bookkeeping operation does not submit a transition. Runtime resources,
allocations, pipelines, sessions and the registered CPU command list remain.
Reset does not reload initial inputs or explicitly clear resource storage.
Buffer data, including intervening uploads, can be reused across calls. Image
content preservation remains subject to layout transitions and alias usage;
a transition from undefined does not preserve previous contents. Applications
can upload new data and download results between calls without rebuilding.
Command buffers are recorded again; completed buffers are released rather than
replayed. The API provides synchronous execution here, with no parallel-run or
thread-safety guarantee implied by retaining this state.

.. _scenario-results:

Results and diagnostics
-----------------------

Outputs and timing
^^^^^^^^^^^^^^^^^^

``run()`` returns after execution and result handling; it does not return a
separate execution-result object.

After all requested iterations complete, ``saveResults`` saves the pipeline
cache, stores resources with configured output destinations, then performs
requested neural, graph-profiling and session-memory dumps. File output therefore
reflects the final iteration. Explicit ``download`` calls use the transfer paths
described in :ref:`scenario-resource-transfers` and return owning CPU data.

CPU performance counters measure scoped host work such as setup, recording,
submission, fence waiting and result saving, in microseconds. Counters aggregate host scopes by category; Time to Inference sums the scopes
marked for inclusion in that total. These host measurements are distinct from
device execution timestamps. With timestamp profiling
requested, setup registers query pairs around dispatches and creates a query
pool. After each iteration's completion wait, profiling reads the timestamps and
device timestamp period and associates them with dispatch names. A VGF sequence
contributes a separately named dispatch per segment. The profiling writer
accumulates iteration records and writes the file on the last iteration. Graph
session-memory requirements are reported alongside timing information, without
requiring execution to produce that memory-usage report.

Dry runs
^^^^^^^^

A dry run still constructs the scenario, resources, pipelines and sessions, and
initial resource uploads can already submit transfer work. It skips the run's
command recording/submission, timestamp retrieval, resource output files and
all post-run neural/graph/session dumps. It still saves enabled pipeline caches,
CPU counters and requested profiling memory-usage data. Repeat iteration/reset
and capture hooks still execute; dry run is not a check that avoids all device
work or all filesystem output.

Optional dumps and capture
^^^^^^^^^^^^^^^^^^^^^^^^^^

Neural debug database extraction queries a graph-pipeline property and writes
nonempty database data; it does not read execution statistics memory. Runtime
statistics instead read the pipeline's retained statistics session allocation
after execution, using the selected statistics mode (0 or 1) for the dump. Both
operate on graph pipelines, including graph module and binary paths described
above. Missing statistics-property data is skipped; a missing statistics memory
allocation produces a warning and no statistics file for that pipeline.

Requests for either neural database or statistics dumps require the neural
accelerator statistics extension to be supported and enabled. Context setup
rejects unsupported requests before execution, including in dry-run mode.
Graph-profiling dumps query a separate optional pipeline property; unavailable,
empty or failed property queries warn and skip that pipeline. Optional session
RAM dumps map the session allocations prepared for that debugging mode. See
:ref:`scenario-usage-neural-statistics` for options and mode selection.

Frame capture is another independent facility. When requested, the scenario
places capture begin/end hooks around each iteration. A build with RenderDoc
support uses an already loaded RenderDoc API; unavailable capture support warns
and leaves the hooks inactive. Vulkan® frame-boundary metadata and timestamp
profiling do not require a successful RenderDoc capture.

Validation and failure boundaries
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

JSON parsing and shape/resource resolution can reject a description before
construction. Builder and resource setup validate typed references, metadata,
allocation requirements and transfer compatibility as their paths consume them.
Context, shader compilation, pipeline and session creation can fail before a
scenario is ready. Upload, recording, queue submission and fence waiting can
also fail; result transfer, property extraction and file handling form later
failure points. These phases are not a complete device-compatibility preflight,
and failure does not promise rollback or a reusable partially executed scenario.
For example, normal profiling needs a populated timestamp query pool; requesting
it for a command stream with no profiled dispatches can fail when reading results.

Use :doc:`usage` for CLI/JSON configuration, :doc:`api` and :doc:`python_api` for
programmatic contracts, and :doc:`limitations` for supported behavior. As explained
in :doc:`security`, scenarios, workload binaries and resources must come from
trusted sources; Scenario Runner is not a sandbox or process-security boundary.
