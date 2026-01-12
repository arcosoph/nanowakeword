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

import collections.abc

class ConfigProxy(collections.abc.Mapping):
    """
    A powerful, transparent proxy for configuration dictionaries that supports
    access tracking, deep nesting, and behaves correctly as both a mapping
    and a numeric/string type when it wraps a final value.
    """
    def __init__(self, data, root_proxy=None, prefix=""):
        # Use a non-standard name to avoid conflicts with potential keys in data
        self._internal_data = data
        self._internal_root = root_proxy if root_proxy is not None else self
        self._internal_prefix = prefix
        
        if self._internal_root is self:
            self._internal_used_params = {}
            self._internal_accessed_keys = set()

    def _track_access(self, key, value):
        full_key = self._internal_prefix + key
        # Only track leaf nodes (final values), not entire dictionaries.
        if not isinstance(value, dict):
            if full_key not in self._internal_root._internal_accessed_keys:
                self._internal_root._internal_used_params[full_key] = value
                self._internal_root._internal_accessed_keys.add(full_key)
    
    def __getitem__(self, key):
        if key not in self._internal_data:
            raise KeyError(f"Key '{self._internal_prefix}{key}' not found in configuration.")
        
        value = self._internal_data[key]
        self._track_access(key, value)
        
        # If the value is another dictionary, wrap it in a new ConfigProxy
        if isinstance(value, dict):
            new_prefix = f"{self._internal_prefix}{key}."
            return ConfigProxy(value, root_proxy=self._internal_root, prefix=new_prefix)
        
        return value

    def __iter__(self):
        return iter(self._internal_data)

    def __len__(self):
        return len(self._internal_data)

    def get(self, key: str, default=None):
        if key in self._internal_data:
            return self[key]
        else:
            self._track_access(key, default)
            if isinstance(default, dict):
                new_prefix = f"{self._internal_prefix}{key}."
                return ConfigProxy(default, root_proxy=self._internal_root, prefix=new_prefix)
            return default
            
    def __setitem__(self, key, value):
        self._internal_data[key] = value
        self._track_access(key, value)

    def report(self) -> dict:
        """Returns a dictionary of all parameters that were accessed."""
        return self._internal_root._internal_used_params
        
    def to_dict(self) -> dict:
        """
        Recursively converts the ConfigProxy back to a standard Python dictionary.
        """
        plain_dict = {}
        for key, value in self.items():
            if isinstance(value, ConfigProxy):
                plain_dict[key] = value.to_dict()
            else:
                plain_dict[key] = value
        return plain_dict

    def __repr__(self):
        return f"ConfigProxy(prefix='{self._internal_prefix}', data={self._internal_data})"

    def _get_leaf_value(self):
        """Helper to get the raw value if this proxy wraps a single item, not a dict."""
        if isinstance(self._internal_data, dict):
            raise TypeError(f"This ConfigProxy wraps a dictionary and cannot be treated as a single value. Path: '{self._internal_prefix}'")
        return self._internal_data

    def __int__(self):
        """Allows casting to int: int(proxy)"""
        return int(self._get_leaf_value())

    def __float__(self):
        """Allows casting to float: float(proxy)"""
        return float(self._get_leaf_value())

    def __str__(self):
        """Allows casting to string: str(proxy)"""
        # If it's a dict, show the dict string representation, otherwise the value's.
        if isinstance(self._internal_data, dict):
            return str(self._internal_data)
        return str(self._get_leaf_value())

    def __add__(self, other):
        """Handles: proxy + other"""
        return self._get_leaf_value() + other

    def __radd__(self, other):
        """Handles: other + proxy (This is what sum() uses!)"""
        return other + self._get_leaf_value()