import time
import typing as t
from contextlib import ContextDecorator

import mlflow
from loguru import logger


class TimeProfiler(ContextDecorator):
    """Time profiling
    Usage:
    1) As context manager:
        ```
        with TimeProfiler('my_task', log_every=2) as t:
            my_value = do_something()
            # optional
            t.record_value('my_value', 123)
        ```
    2) As Decorator
        ```
        @TimerProfiler('my_task', log_every=2)
        def my_func():
            my_value = do_something()
            return
        ```
    """
    _counters = {}  # To support log_every

    def __init__(
        self,
        name: str,
        log_to_mlflow: bool = True,
        log_every: int = 1,
        mlflow_sync: bool = False,
    ):
        self.name = f'{name} ({log_every})'
        self.log_to_mlflow = log_to_mlflow
        self.log_every = log_every
        self.mlflow_sync = mlflow_sync
        if self.name not in TimeProfiler._counters:
            TimeProfiler._counters[self.name] = 0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = time.perf_counter()
        time_comsume = self.end - self.start
        current_function_call_step = TimeProfiler._counters[self.name] // self.log_every

        if (TimeProfiler._counters[self.name] % self.log_every) == 0:
            if self.log_to_mlflow:
                mlflow.log_metric(
                    f'{self.name}_time',
                    time_comsume,
                    step=current_function_call_step,
                    synchronous=self.mlflow_sync,
                )
            else:
                logger.info(f'{self.name} time: {time_comsume}')
        TimeProfiler._counters[self.name] += 1

    def __call__(self, func):
        def wrapper(*args, **kwargs):
            with self:
                return func(*args, **kwargs)
        return wrapper

    def record_value(self, attr_name: str, value: t.Union[int, float, str]):
        """Support recording other output value if needed.
        """
        current_function_call_step = TimeProfiler._counters[self.name] // self.log_every
        if (TimeProfiler._counters[self.name] % self.log_every) == 0:
            if self.log_to_mlflow:
                mlflow.log_metric(
                    f'{self.name}_{attr_name}',
                    value,
                    step=current_function_call_step,
                    synchronous=self.mlflow_sync,
                )
            else:
                logger.info(f'{self.name} {attr_name}: {value}')
