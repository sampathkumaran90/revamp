from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from typing import Any


def abstract_property() -> Any:
  """Returns an abstract property for use in an ABC abstract class."""
  return abc.abstractmethod(lambda: None)