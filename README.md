# mlflow-logger-extension
Extension of [MLFlow](https://mlflow.org/) tracking.  
Why `mlflow-logger-extension`: To record experiments with MlFlow with minimum code intervention using decorators.

## Installation
[Dependency List](./pyproject.toml)  

### From pip
```bash
$ pip install mlflow-logger-ext
```

### From source
```bash
$ git clone ...
$ cd ...
$ pip install .
```

## Usage
Some example use case can be learned from [testcases](mlflow_logger_ext/tests/test_mlflow.py).

### Tracker
0. Setup recording infomation
```python
from mlflow_logger_ext.utils import Teleport, set_experiment

teleport = Teleport(
    experiment_name='my-awesome-experiment',
    tracking_url='path-to-local-dir-or-remote-mlflow-server-port',
    artifact_location='path-to-artifact-location',
)
set_experiment(teleport)
```

**A standard method to use `mlflow` to record parameters and metrics**
```python
import mlflow
def train(arg1: int, arg2: str, arg3: float):
    mlflow.log_params(
        {
            'arg1': arg1,
            'arg2': arg2,
            'arg3': arg3,
        },
    )
    some_metrics = train_epoch(...)
    mlflow.log_metric(
        key='my-metric1',
        value=some_metrics[0],
    )
    mlflow.log_metric(
        key='my-metric2',
        value=some_metrics[1],
    )
```

**With logger extension**
```python
from mlflow_logger_ext import mlflow_wrapper

@mlflow_wrapper.param()
def train(arg1: int, arg2: str, arg3: float):
    some_metric = train_epoch(...)

@mlflow_wrapper.metric(names=['my-metric1', 'my_metric2'])
def train_epoch(...) -> tuple[float, float]:
    ...
    return some_metric1, some_metric2
```
As you can see, user can simply add the decorator on the top of original functions (`train` and `train_epoch`) -- **NO NEED TO INSERT TRACKING CODES INTO ORIGINAL FUNCTION!!**

### More details
1. Input parameters and output metrics can be recorded by spcification using `collect`. For example,
```python
@mlflow_wrapper.param(collect=['params3'])
def pseudofunction(
    params3: dict,
    params4: dict,
    *args,
    **kwargs,
) -> tuple[float, float]:
    return np.random.random(), np.random.random()
```
Your function has many input parameters, but you only want mlflow to record `params3`

2. The `mlflow_wrapper.param` and `mlflow_wrapper.metric` can be stacked. For example
```python
@mlflow_wrapper.metric(names=['log-val3', 'log-val4'], collect=['log-val4'])
@mlflow_wrapper.param(collect=['params3'])
def pseudofunction(
    params3: dict,
    params4: dict,
    *args,
    **kwargs,
) -> tuple[float, float]:
    return np.random.random(), np.random.random()
```
You want to record both inputs and outputs of a single function. Just stack them together.


### TimeProfiler
Log the function or block execution time and record the information using MLFlow

0. Setup environment
Just follow the normal MLFlow setting

1. Loging the execution time of a block
```python
from ... import TimeProfiler


for i in range(100):
    with TimeProfiler('my-block-name', log_every=10):
        ... # do something
```
This will tell the logger to log every `10` iterations. (Activate the logger when touch the profiler `log_every` times.)


2. Logging the execution time of a function - Just add a decorator to the function (or Class function)
```python
from ... import TimeProfiler

@TimeProfiler('my-function-name', log_every=1)
def my_awesome_function(...)
    ... # do something
    return

x = my_awesome_function(...)
```