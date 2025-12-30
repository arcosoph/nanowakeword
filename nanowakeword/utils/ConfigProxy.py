# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

# IMPORTS
import collections.abc

class ConfigProxy(collections.abc.Mapping):
    """
    A powerful, transparent proxy for configuration dictionaries.
    Acts almost exactly like a read-only dictionary (supports .keys(), .values(), .items(), etc.).
    Transparently tracks which configuration values are accessed.
    Returns nested dictionaries also as ConfigProxy objects for deep tracking.
    """
    def __init__(self, data: dict, root_proxy=None, prefix=""):
        # Keeping self._data "private" so it is not directly accessed
        self.__data = data
        self._root = root_proxy if root_proxy is not None else self
        self._prefix = prefix
        
        if self._root is self:
            self._used_params = {}
            self._accessed_keys = set()

    def _track_access(self, key, value):
        full_key = self._prefix + key
        # Only the final values ​​will be tracked, not the entire dictionary.
        if not isinstance(value, (dict, ConfigProxy)):
            if full_key not in self._root._accessed_keys:
                self._root._used_params[full_key] = value
                self._root._accessed_keys.add(full_key)
    

    def __getitem__(self, key):
        if key not in self.__data:
            raise KeyError(f"Key '{self._prefix}{key}' not found in configuration.")
        
        value = self.__data[key]
        self._track_access(key, value) 
        
        if isinstance(value, dict):
            new_prefix = f"{self._prefix}{key}."
            return ConfigProxy(value, root_proxy=self._root, prefix=new_prefix)
        
        return value

    def __iter__(self):
        return iter(self.__data)

    def __len__(self):
        return len(self.__data)
    
    def get(self, key: str, default=None):
        """
        Gets a value, returning a default if the key is not found.
        Tracks the access of the key and the returned value.
        """
        if key in self.__data:
            return self[key]
        else:
            self._track_access(key, default)
            
            if isinstance(default, dict):
                new_prefix = f"{self._prefix}{key}."
                return ConfigProxy(default, root_proxy=self._root, prefix=new_prefix)

            return default
            
    def __setitem__(self, key, value):
        self.__data[key] = value
        self._track_access(key, value) 

    def report(self) -> dict:
        """Returns a dictionary of all parameters that were accessed."""
        return self._root._used_params
        
    def to_dict(self) -> dict:
        """

        Recursively converts the ConfigProxy and all nested proxies back to a standard Python dictionary.
        This is useful when a library or function specifically requires a real dict.
        """
        plain_dict = {}
        for key, value in self.items():
            if isinstance(value, ConfigProxy):
                plain_dict[key] = value.to_dict()
            else:
                plain_dict[key] = value
        return plain_dict

    def __repr__(self):
        return f"ConfigProxy(prefix='{self._prefix}', data={self.__data})"