from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

from pipeline.components.transform.component import Transform

_pipeline_name = 'cifar10_pipeline_pytorch'
_cifar10_root = os.path.join(os.environ['HOME'], "examples/cifar10")
_data_root = ""
_pipeline_root = os.path.join(os.environ['HOME'], "pipeline")
_module_file = os.path.join(_cifar10_root, 'cifar10_utils_pytorch.py')


def _create_pipeline(pipeline_name, pipeline_root, data_root, module_file):
    transform = Transform(input_dict = {}, output_dict = {}, exec_properties = {"module_file": str(_module_file)})


if __name__ == "__main__":
    _create_pipeline(
        pipeline_name =_pipeline_name,
        pipeline_root =_pipeline_root,
        data_root =_data_root,
        module_file =_module_file)