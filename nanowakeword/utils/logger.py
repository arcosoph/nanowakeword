# Copyright 2025 Arcosoph. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box


console = Console()

def print_banner():
    """Prints the minimalist, professional NanoWakeWord banner."""
    banner_art = """
  _   _               __          __   _     __          __           _ 
 | \ | |              \ \        / /  | |    \ \        / /          | |
 |  \| | __ _ _ __   __\ \  /\  / /_ _| | ____\ \  /\  / /__  _ __ __| |
 | . ` |/ _` | '_ \ / _ \ \/  \/ / _` | |/ / _ \ \/  \/ / _ \| '__/ _` |
 | |\  | (_| | | | | (_) \  /\  / (_| |   <  __/\  /\  / (_) | | | (_| |
 |_| \_|\__,_|_| |_|\___/ \/  \/ \__,_|_|\_\___| \/  \/ \___/|_|  \__,_|

"""

    version = "1.0.0" 
    
    console.print(f"\n[bold cyan]{banner_art}[/bold cyan] [bold]NanoWakeWord[/bold] v{version}")
    console.print("-" * 40)


def print_step_header(step_num, title):
    """Prints a clean step header."""
    console.print(f"\n[bold]STEP {step_num}: {title}[/bold]")
    console.print("=" * (len(title) + 8))


def print_info(message, indent=0):
    """Prints an informational message with indentation."""
    console.print(f"{' ' * indent}[blue]INFO:[/blue] {message}")


def print_key_value(key, value, indent=2):
    """Prints a clean, copy-paste friendly key-value pair."""
    console.print(f"{' ' * indent}{key:<25}: {value}")


def print_final_report_header():
    console.print("\n" + "="*40)
    console.print("[bold green]TRAINING COMPLETE - FINAL REPORT[/bold green]")
    console.print("="*40)


def print_table(data_dict, title):
    """Prints a minimalist table for statistics (with nice border)."""
    table = Table(
        title=title,
        show_header=True,
        header_style="bold bright_blue",
        box=box.ROUNDED 
    )
    table.add_column("Parameter", style="white", no_wrap=True)
    table.add_column("Value", style="red")
    
    for key, value in data_dict.items():
        table.add_row(key, str(value))
        
    console.print(table)




                                                                                                                              
                                                                                                                              
#                                                                                                                           ,,  
# `7MN.   `7MF'                         `7MMF'     A     `7MF'      `7MM          `7MMF'     A     `7MF'                  `7MM  
#   MMN.    M                             `MA     ,MA     ,V          MM            `MA     ,MA     ,V                      MM  
#   M YMb   M  ,6"Yb.  `7MMpMMMb.  ,pW"Wq. VM:   ,VVM:   ,V ,6"Yb.    MM  ,MP'.gP"Ya VM:   ,VVM:   ,V ,pW"Wq.`7Mb,od8  ,M""bMM  
#   M  `MN. M 8)   MM    MM    MM 6W'   `Wb MM.  M' MM.  M'8)   MM    MM ;Y  ,M'   Yb MM.  M' MM.  M'6W'   `Wb MM' "',AP    MM  
#   M   `MM.M  ,pm9MM    MM    MM 8M     M8 `MM A'  `MM A'  ,pm9MM    MM;Mm  8M"""""" `MM A'  `MM A' 8M     M8 MM    8MI    MM  
#   M     YMM 8M   MM    MM    MM YA.   ,A9  :MM;    :MM;  8M   MM    MM `Mb.YM.    ,  :MM;    :MM;  YA.   ,A9 MM    `Mb    MM  
# .JML.    YM `Moo9^Yo..JMML  JMML.`Ybmd9'    VF      VF   `Moo9^Yo..JMML. YA.`Mbmmd'   VF      VF    `Ybmd9'.JMML.   `Wbmd"MML.
                                                                                                                              
                                                                                                                              