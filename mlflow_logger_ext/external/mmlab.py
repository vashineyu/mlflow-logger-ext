"""Extension for MMCV MLFLOW logger
Raw: mmcv/runner/hooks/logger/mlflow.py

The original MlflowLoggerHook only record records during training.
This moudle makes
- key parameters in the `config.py` be recorded.
- treat config.py as artifact to store.

To use this module, use
custom_hooks = dict(
    imports=['mlflow_logger_ext.external.mmlab'],
    allow_failed_imports=False,
)

"""
import sys
import warnings
from glob import glob
from typing import Optional

import mlflow


try:
    from mmcv.runner.dist_utils import master_only
    from mmcv.runner.hooks import HOOKS
    from mmcv.runner.hooks.logger import LoggerHook
    from mmcv.runner.iter_based_runner import IterBasedRunner
    from mmcv.utils import Config
except ImportError:
    warnings.warn(
        'To use MLFlowTrack, you have to install MMCV first',
        ImportWarning,
    )


if sys.version_info.minor < 8:
    from importlib_metadata import version
else:
    from importlib.metadata import version


@HOOKS.register_module(name='MlflowLoggerHook', force=True)
class MlFlowTrack(LoggerHook):
    """
    Hook for mlflow tracking

    Notes: Additional Dependencies
        It requires 'mlflow' package. Make sure you have installed it.

    Args:
        exp_name (str): Experiment name. Should be set as your project name.
        run_id (Optional[str]): Attach logger to an exist record. Default is `None` which means this logger
            will create a new record.
        log_model (bool): Record model file as artifact (not recommend). Default is `False`.
        tags (Optional[dict]): Key-value pairs for custom tags.
        additional_params (Optional[dict]): Additional parameters to be recorded.
        tracking_uri (Optional[str]): Destination to the tracking URI (ex. http://0.0.0.0:8050).
        artifact_location (Optional[str]): A path to a local directory or remote s3 service (s3://[bucket-name]).
    """
    def __init__(
        self,
        exp_name: str,
        run_id: Optional[str] = None,
        log_model: bool = False,
        tags: Optional[dict] = None,
        additional_params: Optional[dict] = None,
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.exp_name = exp_name
        self.run_id = run_id
        self.log_model = log_model
        self.tracking_uri = tracking_uri
        self.artifact_location = artifact_location
        self.tags = tags or {}
        self.additional_params = additional_params or {}

    @master_only
    def before_run(self, runner):
        super().before_run(runner)

        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)

        if self.artifact_location and mlflow.get_experiment_by_name(self.exp_name) is None:
            mlflow.create_experiment(self.exp_name, self.artifact_location)

        mlflow.set_experiment(self.exp_name)

        if self.run_id:
            mlflow.start_run(self.run_id)

        mlflow.set_tags(self.get_tags())

        # Automatically set flags via runner type.
        # TODO: does this work ? self.by_epoch = not isintance(runner, IterBasedRunner)
        if isinstance(runner, IterBasedRunner):
            self.by_epoch = False

        config = self.get_config(runner)

        params = {
            'model_arch': config['model']['type'],
            'backbone': config['model']['backbone']['type'],
            'samples_per_gpu': config['data']['samples_per_gpu'],
            'optimizer': config['optimizer']['type'],
            'init_lr': config['optimizer']['lr'],
            'num_epochs': runner.max_epochs,
            'num_iters': runner.max_iters,
            **self.additional_params,
        }

        mlflow.log_params(params)

    @master_only
    def log(self, runner):
        metrics = self.get_loggable_tags(runner)
        if metrics:
            mlflow.log_metrics(metrics, step=self.get_iter(runner))

    @master_only
    def after_run(self, runner):
        if self.log_model:
            mlflow.pytorch.log_model(runner.model, 'models')

        # get saved config path as artifact
        work_dir = self.get_config(runner)['work_dir']
        config_path = glob(f'{work_dir}/*.py')[0]

        mlflow.log_artifact(config_path)

    def get_config(self, runner):
        return Config.fromstring(runner.meta['config'], '.py').to_dict()

    def get_tags(self):
        tags = {
            'mmcv-full': version('mmcv-full'),
            'torch': version('torch'),
        }
        tags.update(self.tags)
        return tags
