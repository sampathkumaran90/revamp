import abc
import inspect
from typing import Any, Dict, Optional, Text
from six import with_metaclass
from abc import abstractmethod


class BaseComponent(with_metaclass(abc.ABCMeta, )):

    @abstractmethod
    def preprocess(self, preprocess_file):
        pass

    