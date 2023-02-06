import inspect
from functools import partial, update_wrapper
from typing import Any, Callable, Optional


class MlFlowBaseLogger:
    """MLFlow base logger.

    Args:
        function (callable): function to wrap
    """
    def __init__(self, function: Callable):
        update_wrapper(self, function)
        self.function = function

    @property
    def is_member_function(self):
        return self._get_function_type(self.function)

    def execute(
        self,
        obj: Optional[Any] = None,
        *args,
        **kwargs,
    ):
        if self.is_member_function:
            outputs = self.function(obj, *args, **kwargs)
        else:
            outputs = self.function(*args, **kwargs)
        return outputs

    def _get_function_type(self, function: Callable):
        """To make it works as good as deco in normal functions and class functions."""
        signature = inspect.signature(function)
        return 'self' in signature.parameters

    def _call_impl(
        self,
        obj=None,
        *args,
        **kwargs,
    ):
        return self.execute(*args, **kwargs)

    def __call__(
        self,
        obj=None,
        *args,
        **kwargs,
    ):
        """Used for acting as decorator to deco main calling function.

        Must call self._set_experiment at begining.
        """
        return self._call_impl(
            obj=None,
            *args,
            **kwargs,
        )

    def __get__(self, obj, objtype):
        return partial(self._call_impl, obj)
