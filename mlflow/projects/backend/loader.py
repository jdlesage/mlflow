import entrypoints
import logging

from mlflow.projects.backend import databricks, local

ENTRYPOINT_GROUP_NAME = "mlflow.mlproject_backend"

__logger__ = logging.getLogger(__name__)


# Statically register backends defined in mlflow
MLFLOW_BACKENDS = {"databricks": databricks.DatabricksProjectBackend,
                   "local": local.LocalBackend}


def load_kubernetes_backend():
    try:
        from mlflow.projects.backend import kubernetes
        MLFLOW_BACKENDS["kubernetes"] = kubernetes.KubernetesBackend
    except ModuleNotFoundError:
        __logger__.info("Ignore kubernetes backend")
        # kubernetes not installed. Ignore this backend


def load_backend(backend_name):
    # Static backends
    if not backend_name:
        return local.LocalBackend()

    load_kubernetes_backend()
    if backend_name in MLFLOW_BACKENDS:
        return MLFLOW_BACKENDS[backend_name]()

    # backends from plugin
    try:
        backend_builder = entrypoints.get_single(ENTRYPOINT_GROUP_NAME,
                                                 backend_name).load()
        return backend_builder()
    except entrypoints.NoSuchEntryPoint:
        # TODO Should be a error when all backends are migrated here
        available_entrypoints = entrypoints.get_group_all(
            ENTRYPOINT_GROUP_NAME)
        available_plugins = [
            entrypoint.name for entrypoint in available_entrypoints]
        __logger__.warning("Backend '%s' is not available. Available plugins are %s",
                           backend_name, available_plugins + list(MLFLOW_BACKENDS.keys()))
        raise

    return None
