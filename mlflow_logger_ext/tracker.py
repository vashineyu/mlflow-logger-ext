import inspect
import mlflow
from typing import Callable, Union

from .base import MlFlowBaseLogger


class TrackMetric(MlFlowBaseLogger):
    def __init__(
        self,
        function: Callable,
        names: Union[list, str],
        collect: Union[list, str, None] = None,
    ):
        super().__init__(function=function)
        self.names = names if isinstance(names, list) else [names]
        if collect:
            self.collect = collect if isinstance(collect, list) else [collect]
        else:
            self.collect = None

    def _call_impl(self, obj=None, *args, **kwargs):
        outputs = self.execute(obj, *args, **kwargs)

        if not isinstance(outputs, (tuple, list)):
            outputs = [outputs]

        metric_dict = dict(zip(self.names, outputs))
        metric_dict = {k: float(v) for k, v in metric_dict.items()}
        if self.collect:
            metric_dict = {
                metric_name: metric_value for metric_name, metric_value in metric_dict.items()
                if metric_name in self.collect
            }
        mlflow.log_metrics(metric_dict)
        return outputs


class TrackParam(MlFlowBaseLogger):
    def __init__(
        self,
        function: Callable,
        collect: Union[list, str, None] = None,
    ):
        super().__init__(function)
        if collect:
            self.collect = collect if isinstance(collect, list) else [collect]
        else:
            self.collect = None

    def _call_impl(self, obj=None, *args, **kwargs):
        outputs = self.execute(obj, *args, **kwargs)
        function_signature = inspect.signature(self.function)
        params = {
            k: v.default for k, v in function_signature.parameters.items()
        }
        params.update(kwargs)
        if self.is_member_function:
            params.pop('self')

        if self.collect:
            params = {k: v for k, v in params.items() if k in self.collect}

        mlflow.log_params(params)
        return outputs
