"""Task-related utilities for benchmarking."""

import os
import inspect
import importlib
from typing import Dict, Type

import hydrax.tasks as tasks
from hydrax.task_base import Task


def get_all_tasks() -> Dict[str, Type[Task]]:
    """Dynamically find all task classes.

    Returns:
        Dictionary mapping task names to task classes.
    """
    task_dict = {}

    # Iterate through all modules in the tasks package
    for file in os.listdir(os.path.dirname(tasks.__file__)):
        if file.endswith(".py") and not file.startswith("__"):
            module_name = file[:-3]
            module = importlib.import_module(f"hydrax.tasks.{module_name}")

            # Find classes in the module that are Tasks
            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Task)
                    and obj != Task
                    and name != "Task"
                ):
                    task_dict[name] = obj

    return task_dict
