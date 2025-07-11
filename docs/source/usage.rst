Usage
=============

.. include:: ../generated/README.md
    :parser: myst_parser.sphinx_
    :start-after: ### Usage
    :end-before: ## License

The following command is an example run command that reads a JSON file and executes it:

.. code-block:: bash

  ./build/scenario-runner --scenario=JSON_FILE

Where:

- ``--scenario``: File to load the scenario from. The file must be in JSON
  format. If the resources in the JSON_FILE are not specified with absolute paths,
  their relative paths will be resolved against the parent directory of the JSON_FILE.

Tensor/Image aliasing
---------------------

It is useful for some resources to share the same underlying memory. Sharing memory allows resources to share inputs or outputs between shaders and VGF workloads.

To enable the Tensor/Image aliasing feature, in the scenario json file, a tensor must have a ``alias_target`` field with the ``resource_ref`` contained inside pointing at the ``uid`` of the aliased image. The tensor must not have an ``src`` field.

The following example shows you how to setup a tensor resource that alias an image resource.

.. code-block:: json

  {
      "commands": [],
      "resources": [
          {
              "image": {
                  "shader_access": "readonly",
                  "dims": [ 1, 64, 10, 1 ],
                  "format": "VK_FORMAT_R32_SFLOAT",
                  "src": "input.dds",
                  "uid": "input_image",
              }
          },
          {
              "tensor": {
                  "shader_access": "writeonly",
                  "dims": [ 1, 10, 64, 1 ],
                  "format": "VK_FORMAT_R32_SFLOAT",
                  "dst": "output.npy",
                  "uid": "output_tensor",
                  "alias_target": {
                      "resource_ref" : "input_image"
                  }
              }
          }
      ]
  }

This example performs no calculations. However, the example reads in the input image data and saves it to the ``output.npy`` file. If the image has padding added to its data, the image padding is discarded when saving to the NumPy file.

The following example is a more realistic usage of the Tensor/Image aliasing feature. The example has a preprocessing shader which has images as its input and output. The example then uses this image output as the input for a VGF dispatch, which has tensors for its input and output. The image outputs of the preprocessing shader stage are then aliased to tensors which are used as input of the VGF dispatch stage.

.. code-block:: json

  {
    "commands": [
      {
        "dispatch_compute": {
          "bindings": [
            { "set": 0, "id": 0, "resource_ref": "input_image" },
            { "set": 0, "id": 1, "resource_ref": "output_image" }
          ],
          "shader_ref": "image_shader",
          "rangeND": [64, 64, 1]
        }
      },
      {
        "dispatch_graph": {
          "bindings": [
            { "set": 0, "id": 2, "resource_ref": "input_tensor" },
            { "set": 0, "id": 3, "resource_ref": "output_tensor" }
          ],
          "shader_ref": "tensor_vgf",
          "rangeND": [64, 64, 1]
        }
      }
    ],
    "resources": [
      {
        "shader": {
          "uid": "image_shader",
          "src": "imageShader.spv",
          "entry": "main",
          "type": "SPIR-V",
        }
      },
      {
        "graph": {
          "uid": "tensor_vgf",
          "src": "tensorVgf.vgf",
        }
      },
      {
        "image": {
          "uid": "input_image",
          "dims": [1, 64, 64, 1],
          "shader_access": "image_read",
          "format": "VK_FORMAT_R16G16B16A16_SFLOAT"
        }
      },
      {
        "image": {
          "uid": "output_image",
          "dims": [1, 64, 64, 1],
          "mips": false,
          "format": "VK_FORMAT_R16G16B16A16_SFLOAT",
          "shader_access": "writeonly",
        }
      },
      {
        "tensor": {
          "uid": "input_tensor",
          "dims": [1, 64, 64, 4],
          "format": "VK_FORMAT_R16_UINT",
          "shader_access": "readwrite",
          "alias_target": {
            "resource_ref": "output_image"
          },
        }
      },
      {
        "tensor": {
          "uid": "output_tensor",
          "dims": [1, 64, 64, 4],
          "format": "VK_FORMAT_R16_UINT",
          "shader_access": "readwrite",
          "dst": "outputTensor.npy"
        }
      }
    ]
  }

For this example, the scenario runner automatically inserts a memory barrier between the shader and graph dispatches. The memory barrier allows the output image data to be correctly shared with the "input_tensor" resource. Only single component data types are allowed in NumPy files, therefore, you can use the VK_FORMAT_R16_UINT type and multiplying the innermost "dims" value by 4 to approximate the VK_FORMAT_R16G16B16A16_SFLOAT type that the images use.

In general, the innermost dimension of the tensor must match the number of components of the image data type. The size of the tensor data type must also match the size of the image data type component.

Using Emulation and Validation layers
-------------------------------------

You can use the ML SDK Emulation Layer to enable the Scenario Runner to run on platforms which do not support the Tensor and Graph Vulkan® extensions. However, you must have already built the Emulation layer. To enable the Emulation layer on Linux, set the following environment variables:
 - LD_LIBRARY_PATH=path/to/build/sw/vulkan-ml-emulation-layer/build/graph/:path/to/build/sw/vulkan-ml-emulation-layer/build/tensor/:$LD_LIBRARY_PATH
 - VK_LAYER_PATH=path/to/build/sw/vulkan-ml-emulation-layer/build/graph/layer.d:path/to/build/sw/vulkan-ml-emulation-layer/build/tensor/layer.d
 - VK_INSTANCE_LAYERS=VK_LAYER_ML_Graph_Emulation:VK_LAYER_ML_Tensor_Emulation

To check for correct usage of the Scenario Runner's Vulkan® API calls, you can use the Vulkan® Validation Layers. To enable the Vulkan® Validation Layers on Linux, set the following environment variables:
 - LD_LIBRARY_PATH={PATH_TO_VALIDATION_LAYERS}/build:$LD_LIBRARY_PATH
 - VK_LAYER_PATH={PATH_TO_VALIDATION_LAYERS}/build/layers
 - VK_INSTANCE_LAYERS=VK_LAYER_KHRONOS_validation
