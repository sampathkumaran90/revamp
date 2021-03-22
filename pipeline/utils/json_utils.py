from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import importlib
import inspect
import json
from typing import Any, Dict, List, Text, Type, Union

from six import with_metaclass

class Jsonable(with_metaclass(abc.ABCMeta, object)):
  """Base class for serializing and deserializing objects to/from JSON.
  The default implementation assumes that the subclass can be restored by
  updating `self.__dict__` without invoking `self.__init__` function.. If the
  subclass cannot hold the assumption, it should
  override `to_json_dict` and `from_json_dict` to customize the implementation.
  """

  def to_json_dict(self) -> Dict[Text, Any]:
    """Convert from an object to a JSON serializable dictionary."""
    return self.__dict__

  @classmethod
  def from_json_dict(cls, dict_data: Dict[Text, Any]) -> Any:
    """Convert from dictionary data to an object."""
    instance = cls.__new__(cls)
    instance.__dict__ = dict_data
    return instance


JsonableValue = Union[bool, bytes, float, int, Jsonable, message.Message, Text,
                      Type]
JsonableList = List[JsonableValue]
JsonableDict = Dict[Union[bytes, Text], Union[JsonableValue, JsonableList]]
JsonableType = Union[JsonableValue, JsonableList, JsonableDict]