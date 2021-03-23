from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pipeline.dsl.components.base import base_executor
from pipeline.components.util import udf_utils

class Executor(base_executor.BaseExecutor):

    def Do(self, input_dict, output_dict, exec_properties):
        print("Executor is called from component!!!")
        preprocess_fn = udf_utils.get_fn(exec_properties, 'preprocess_fn')
        preprocess_fn("test")

