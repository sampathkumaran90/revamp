from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Any, Callable, Dict, Optional, Text

from pipeline.utils import import_utils

# Key for module file path.
_MODULE_FILE_KEY = 'module_file'
# Key for module python path.
_MODULE_PATH_KEY = 'module_path'


def get_fn(exec_properties: Dict[Text, Any],
           fn_name: Text) -> Callable[..., Any]:
  """Loads and returns user-defined function."""

  has_module_file = bool(exec_properties.get(_MODULE_FILE_KEY))
  has_module_path = bool(exec_properties.get(_MODULE_PATH_KEY))
  has_fn = bool(exec_properties.get(fn_name))

  if has_module_path:
    module_path = exec_properties[_MODULE_PATH_KEY]
    return import_utils.import_func_from_module(module_path, fn_name)
  elif has_module_file:
    if has_fn:
      return import_utils.import_func_from_source(
          exec_properties[_MODULE_FILE_KEY], exec_properties[fn_name])
    else:
      return import_utils.import_func_from_source(
          exec_properties[_MODULE_FILE_KEY], fn_name)
  elif has_fn:
    fn_path_split = exec_properties[fn_name].split('.')
    return import_utils.import_func_from_module('.'.join(fn_path_split[0:-1]),
                                                fn_path_split[-1])
  else:
    raise ValueError(
        'Neither module file or user function have been supplied in `exec_properties`.'
    )


def try_get_fn(exec_properties: Dict[Text, Any],
               fn_name: Text) -> Optional[Callable[..., Any]]:
  """Loads and returns user-defined function if exists."""
  try:
    return get_fn(exec_properties, fn_name)
  except (ValueError, AttributeError):
    # ValueError: module file or user function is unset.
    # AttributeError: the function doesn't exist in the module.
    return None