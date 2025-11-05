# ==============================================================================
#  NanoWakeWord — Lightweight Intelligent Wake Word Detection
#  © 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0:
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Distributed on an "AS IS" basis, without warranties or conditions of any kind.
#  For details, visit the project repository:
#      https://github.com/arcosoph/nanowakeword
# ==============================================================================



from rich.console import Console
from rich.table import Table
from rich.text import Text
from rich import box


console = Console()

def print_banner():
    """Prints the minimalist, professional NanoWakeWord banner."""
    banner_art = r"""
  _   _               __          __   _     __          __           _ 
 | \ | |              \ \        / /  | |    \ \        / /          | |
 |  \| | __ _ _ __   __\ \  /\  / /_ _| | ____\ \  /\  / /__  _ __ __| |
 | . ` |/ _` | '_ \ / _ \ \/  \/ / _` | |/ / _ \ \/  \/ / _ \| '__/ _` |
 | |\  | (_| | | | | (_) \  /\  / (_| |   <  __/\  /\  / (_) | | | (_| |
 |_| \_|\__,_|_| |_|\___/ \/  \/ \__,_|_|\_\___| \/  \/ \___/|_|  \__,_|

"""

 
    
    console.print(f"\n[bold cyan]{banner_art}[/bold cyan]")
    # console.print("-" * 40)


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
    """
    Prints a professional, monochrome, and theme-agnostic table.
    This classic black-and-white style offers maximum readability and a timeless look.
    """
    table = Table(
        title=f"[blue]{title}[/blue]", # A standout, inverted title
        show_header=True,
        # header_style="bold white",           # Simple, bold white for headers
        box=box.SQUARE,                        # A clean, minimalist box style
        border_style="dim"                     # A subtle border
    )

    # `bright_black` is the most theme-adaptive color for text.
    # It appears as light gray on dark terminals and dark gray on light terminals.
    table.add_column("Parameter", style="bright_black", no_wrap=True, justify="lift")

    # `white` is a strong, neutral color for the most important data.
    table.add_column("Value")
    
    for key, value in data_dict.items():
        value_str = str(value)
        if len(value_str) > 80:
            value_str = value_str[:77] + "..."
        table.add_row(key, value_str)
        
    console.print(table)

                                                                                                                              
                                                                                                                              
#                                                                                                                           ,,  
# `7MN.   `7MF'                         `7MMF'     A     `7MF'      `7MM          `7MMF'     A     `7MF'                  `7MM  
#   MMN.    M                             `MA     ,MA     ,V          MM            `MA     ,MA     ,V                      MM  
#   M YMb   M  ,6"Yb.  `7MMpMMMb.  ,pW"Wq. VM:   ,VVM:   ,V ,6"Yb.    MM  ,MP'.gP"Ya VM:   ,VVM:   ,V ,pW"Wq.`7Mb,od8  ,M""bMM  
#   M  `MN. M 8)   MM    MM    MM 6W'   `Wb MM.  M' MM.  M'8)   MM    MM ;Y  ,M'   Yb MM.  M' MM.  M'6W'   `Wb MM' "',AP    MM  
#   M   `MM.M  ,pm9MM    MM    MM 8M     M8 `MM A'  `MM A'  ,pm9MM    MM;Mm  8M"""""" `MM A'  `MM A' 8M     M8 MM    8MI    MM  
#   M     YMM 8M   MM    MM    MM YA.   ,A9  :MM;    :MM;  8M   MM    MM `Mb.YM.    ,  :MM;    :MM;  YA.   ,A9 MM    `Mb    MM  
# .JML.    YM `Moo9^Yo..JMML  JMML.`Ybmd9'    VF      VF   `Moo9^Yo..JMML. YA.`Mbmmd'   VF      VF    `Ybmd9'.JMML.   `Wbmd"MML.
                                                                                                                              
                                                                                                                              