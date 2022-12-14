import builtins
from dataclasses import asdict, dataclass
from typing import Any, Optional

import mlflow


@dataclass
class Teleport:
    experiment_name: str
    tracking_url: str
    artifact_location: Optional[str] = None

    def asdict(self) -> dict[str, Any]:
        return asdict(self)


def get_mlflow_env() -> Teleport:
    return Teleport(
        experiment_name=builtins.MLFLOW_EXPERIMENT_NAME,
        tracking_url=builtins.MLFLOW_TRACKING_URL,
        artifact_location=builtins.MLFLOW_ARTIFACT_LOCATION,
    )


def set_experiment(teleport: Teleport):
    _set_mlflow_env(teleport)  # set cross-module variables
    print(f'Set tracking environment variables: {teleport}')
    mlflow.set_tracking_uri(teleport.tracking_url)
    fetched_experiment_name = mlflow.get_experiment_by_name(
        teleport.experiment_name,
    )
    if teleport.artifact_location and fetched_experiment_name is None:
        mlflow.create_experiment(
            teleport.experiment_name,
            teleport.artifact_location,
        )
    mlflow.set_experiment(teleport.experiment_name)


def _set_mlflow_env(teleport: Teleport):
    # https://stackoverflow.com/questions/142545/how-to-make-a-cross-module-variable
    builtins.MLFLOW_EXPERIMENT_NAME = teleport.experiment_name
    builtins.MLFLOW_TRACKING_URL = teleport.tracking_url
    builtins.MLFLOW_ARTIFACT_LOCATION = teleport.artifact_location
