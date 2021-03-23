from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any, Dict, Optional, Text, Type
from six import with_metaclass

def _abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)


class BaseNode(with_metaclass(abc.ABC, json_utils.Jsonable)):
  



