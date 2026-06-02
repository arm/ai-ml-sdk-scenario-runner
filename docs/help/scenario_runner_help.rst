Usage: ./scenario-runner [--help] [--version] --scenario VAR [--output VAR] [--profiling-dump-path VAR] [--pipeline-caching] [--clear-pipeline-cache] [--cache-path VAR] [--neural-debug-database-dump-dir VAR] [--fail-on-pipeline-cache-miss] [--emulation-layer-profiling-dump-dir VAR] [--neural-statistics-dump-dir VAR] [--neural-statistics-mode VAR] [--perf-counters-dump-path VAR] [--log-level VAR] [--wait-for-key-stroke-before-run] [--dry-run] [--disable-extension VAR...]... [--enable-gpu-debug-markers] [--session-memory-dump-dir VAR] [--repeat VAR] [--capture-frame] [--pause-on-exit]

Optional arguments:
  -h, --help                            shows help message and exits
  -v, --version                         prints version information and exits
  --scenario                            file to load the scenario from. File should be in JSON format [required]
  --output                              output folder
  --profiling-dump-path                 path to save runtime profiling
  --pipeline-caching                    enable the pipeline caching
  --clear-pipeline-cache                clear pipeline cache
  --cache-path                          set pipeline cache location [default: "/tmp"]
  --neural-debug-database-dump-dir      path to dump Neural Accelerator Debug Database [nargs=0..1] [default: ""]
  --fail-on-pipeline-cache-miss         ensure an error is generated on a pipeline cache miss
  --emulation-layer-profiling-dump-dir  path to dump Emulation Layer graph profiling data [nargs=0..1] [default: ""]
  --neural-statistics-dump-dir          path to dump Neural Accelerator Statistics [nargs=0..1] [default: ""]
  --neural-statistics-mode              set neural accelerator statistics mode to 0 or 1 [nargs=0..1] [default: "1"]
  --perf-counters-dump-path             path to save performance counter stats [default: ""]
  --log-level                           set logging level [default: info]
  --wait-for-key-stroke-before-run      wait for a key stroke before run
  --dry-run                             setup pipelines but skip the actual execution
  --disable-extension                   specify extensions to disable out of the following: VK_EXT_custom_border_color, VK_EXT_frame_boundary, VK_ARM_data_graph_neural_accelerator_statistics, VK_KHR_maintenance5, VK_KHR_deferred_host_operations [nargs: 1 or more] [may be repeated]
  --enable-gpu-debug-markers            enable GPU debug markers
  --session-memory-dump-dir             path to dump the contents of the sessions ram after inference completes
  --repeat                              optional repeat count for scenario execution
  --capture-frame                       enable RenderDoc integration for frame capturing
  --pause-on-exit                       pause before exiting
