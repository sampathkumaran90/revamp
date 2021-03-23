from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Dict, Optional, Text, Union

from pipeline.dsl.components.base.base_component import BaseComponent
from pipeline.components.transform.executor import Executor

class Transform(BaseComponent):

    def __init__(self, input_dict, output_dict, exec_properties):
        super(Transform, self).__init__()
        self.call_execute_fn(input_dict, output_dict, exec_properties)

    def call_execute_fn(self, input_dict, output_dict, exec_properties):
        print("Executor is being called from this function!!!")
        Executor().Do(input_dict, output_dict, exec_properties)
