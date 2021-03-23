from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import json
import os
import sys
from typing import Any, Dict, List, Optional, Text
from six import with_metaclass


class BaseExecutor(with_metaclass(abc.ABCMeta, object)):

    @abc.abstractmethod
    def Do(self, input_dict, output_dict, exec_properties):
        pass