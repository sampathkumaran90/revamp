import abc
import inspect
from typing import Any, Dict, Optional, Text
from six import with_metaclass
from abc import abstractmethod
from pipeline.utils import abc_utils
import pipeline.types


class BaseComponent(with_metaclass(abc.ABCMeta, object)):

    def __init__(self, custom_executor_spec = None, instance_name = None):

        SPEC_CLASS = abc_utils.abstract_property()
        EXECUTOR_SPEC = abc_utils.abstract_property()

    @abc.abstractmethod
    def call_execute_fn(self, input_dict, output_dict, exec_properties):
        pass