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


import sys
from io import StringIO
from nanowakeword.utils.logger import print_table


class DynamicTable:
    def __init__(self, config_proxy, title="Training Configuration", enabled: bool = True):
        self._proxy = config_proxy
        self._title = title
        self._print_func = print_table 
        self._last_state = {}
        self._last_table_height = 0

        self._is_enabled = enabled


    def _move_cursor_up(self, lines):
        """Moves the terminal cursor up by a specified number of lines."""
        if lines > 0:
            sys.stdout.write(f'\x1b[{lines}A')
            sys.stdout.flush()

    def _clear_from_cursor(self):
        """Clears the screen from the current cursor position to the end."""
        sys.stdout.write('\x1b[J')
        sys.stdout.flush()

    def update(self, force_print=False):
        """
        Updates the table in-place, but only if it's enabled.
        """
        if not self._is_enabled:
            return 

        current_state = self._proxy.report()
        
        if current_state != self._last_state or force_print:
            self._move_cursor_up(self._last_table_height)
            self._clear_from_cursor()

            display_config = {}
            keys_to_exclude = {
                "positive_data_path", "negative_data_path", "background_paths",
                "rir_paths", "output_dir", "force_verify"
            }
            sorted_keys = sorted(current_state.keys())
            for key in sorted_keys:
                if key not in keys_to_exclude and current_state[key] is not None:
                    display_config[key] = str(current_state[key])
            
            old_stdout = sys.stdout
            sys.stdout = string_io = StringIO()
            self._print_func(display_config, self._title)
            table_string = string_io.getvalue()
            sys.stdout = old_stdout 
            
            self._last_table_height = table_string.count('\n') + 1

            print(table_string, end='')
            
            self._last_state = current_state.copy()
