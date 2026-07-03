Neural accelerator statistics
-----------------------------

Scenario Runner supports ``VK_ARM_data_graph_neural_accelerator_statistics`` for data graph pipelines. When this extension and its ``dataGraphNeuralAcceleratorStatistics`` feature are available, Scenario Runner can request neural accelerator debug database and statistics output from graph pipelines.

Use ``--neural-debug-database-dump-dir`` to dump the neural accelerator debug database for graph pipelines that expose one. Use ``--neural-statistics-dump-dir`` to enable neural accelerator statistics collection and dump the resulting statistics memory after execution. ``--neural-statistics-mode`` selects the statistics mode, using ``0`` for ``VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS_0_ARM`` and ``1`` for ``VK_NEURAL_ACCELERATOR_STATISTICS_MODE_STATISTICS_1_ARM``.

If neural accelerator dumps are requested and ``VK_ARM_data_graph_neural_accelerator_statistics`` is not supported or enabled, Scenario Runner fails during startup before creating the Vulkan® device.

Graph profiling dumps
---------------------

Use ``--emulation-layer-profiling-dump-dir`` to request Emulation layer graph profiling payload from data graph pipelines using the custom Emulation layer property token ``0x7ffffffe`` in ``vkGetDataGraphPipelineAvailablePropertiesARM``. This option is Emulation layer specific and best-effort: Scenario Runner warns and continues if the property is unavailable or returns no data. When data is returned, Scenario Runner writes one file per graph pipeline with the name ``Graph_Pipeline_<idx>_EL_Performance.json``. The payload is stored exactly as returned, without JSON validation or pretty-printing.
