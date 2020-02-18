from __future__ import absolute_import
import logging
import docker
import time
import os
from threading import RLock
from datetime import datetime
import yaml

import kubernetes
from kubernetes.config.config_exception import ConfigException

from mlflow.projects.backend.abstract_backend import AbstractBackend
from mlflow.exceptions import ExecutionException
from mlflow.projects.submitted_run import SubmittedRun
from mlflow.projects import utils
from mlflow.entities import RunStatus

from shlex import split
from six.moves import shlex_quote as quote

_logger = logging.getLogger(__name__)


def push_image_to_registry(image_tag):
    client = docker.from_env()
    _logger.info("=== Pushing docker image %s ===", image_tag)
    for line in client.images.push(repository=image_tag, stream=True, decode=True):
        if 'error' in line and line['error']:
            raise ExecutionException("Error while pushing to docker registry: "
                                     "{error}".format(error=line['error']))
    return client.images.get_registry_data(image_tag).id


def _get_kubernetes_job_definition(project_name, image_tag, image_digest,
                                   command, env_vars, job_template):
    container_image = image_tag + '@' + image_digest
    timestamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S-%f')
    job_name = "{}-{}".format(project_name, timestamp)
    _logger.info("=== Creating Job %s ===", job_name)
    environment_variables = [{'name': k, 'value': v} for k, v in env_vars.items()]
    job_template['metadata']['name'] = job_name
    job_template['spec']['template']['spec']['containers'][0]['name'] = project_name
    job_template['spec']['template']['spec']['containers'][0]['image'] = container_image
    job_template['spec']['template']['spec']['containers'][0]['command'] = command
    if 'env' not in job_template['spec']['template']['spec']['containers'][0].keys():
        job_template['spec']['template']['spec']['containers'][0]['env'] = []
    job_template['spec']['template']['spec']['containers'][0]['env'] += environment_variables
    return job_template


def _get_run_command(entrypoint_command):
    formatted_command = []
    for cmd in entrypoint_command:
        formatted_command.extend([quote(s) for s in split(cmd)])
    return formatted_command


def _load_kube_context(context=None):
    try:
        # trying to load either the context passed as arg or, if None,
        # the one provided as env var `KUBECONFIG` or in `~/.kube/config`
        kubernetes.config.load_kube_config(context=context)
    except (IOError, ConfigException) as e:
        _logger.debug('Error loading kube context "%s": %s', context, e)
        _logger.info('No valid kube config found, using in-cluster configuration')
        kubernetes.config.load_incluster_config()


def run_kubernetes_job(project_name, active_run, image_tag, image_digest, command, env_vars,
                       kube_context=None, job_template=None):
    job_template = _get_kubernetes_job_definition(project_name,
                                                  image_tag,
                                                  image_digest,
                                                  _get_run_command(command),
                                                  env_vars,
                                                  job_template)
    job_name = job_template['metadata']['name']
    job_namespace = job_template['metadata']['namespace']
    _load_kube_context(context=kube_context)
    api_instance = kubernetes.client.BatchV1Api()
    api_instance.create_namespaced_job(namespace=job_namespace,
                                       body=job_template, pretty=True)
    return KubernetesSubmittedRun(active_run.info.run_id, job_name, job_namespace)


def _parse_kubernetes_config(backend_config):
    """
    Creates build context tarfile containing Dockerfile and project code, returning path to tarfile
    """
    if not backend_config:
        raise ExecutionException("Backend_config file not found.")
    kube_config = backend_config.copy()
    if 'kube-job-template-path' not in backend_config.keys():
        raise ExecutionException("'kube-job-template-path' attribute must be specified in "
                                 "backend_config.")
    kube_job_template = backend_config['kube-job-template-path']
    if os.path.exists(kube_job_template):
        with open(kube_job_template, 'r') as job_template:
            yaml_obj = yaml.safe_load(job_template.read())
        kube_job_template = yaml_obj
        kube_config['kube-job-template'] = kube_job_template
    else:
        raise ExecutionException("Could not find 'kube-job-template-path': {}".format(
            kube_job_template))
    if 'kube-context' not in backend_config.keys():
        _logger.debug("Could not find kube-context in backend_config."
                      " Using current context or in-cluster config.")
    if 'repository-uri' not in backend_config.keys():
        raise ExecutionException("Could not find 'repository-uri' in backend_config.")
    return kube_config


class KubernetesBackend(AbstractBackend):
    def run(self, active_run, uri, entry_point, parameters, backend_config):
        env_vars = utils.generate_env_vars_to_attach_to_run(active_run)
        docker_image = backend_config.pop('image')
        command_args = backend_config.pop('command_args')
        kube_config = _parse_kubernetes_config(backend_config)
        image_digest = push_image_to_registry(docker_image.tags[0])
        return run_kubernetes_job(
            backend_config['project_name'],
            active_run,
            docker_image.tags[0],
            image_digest,
            command_args,
            env_vars,
            kube_config.get('kube-context', None),
            kube_config['kube-job-template']
        )

    @property
    def name(self):
        return "kubernetes"


class KubernetesSubmittedRun(SubmittedRun):
    """
    Instance of SubmittedRun corresponding to a Kubernetes Job run launched to run an MLflow
    project.
    :param mlflow_run_id: ID of the MLflow project run.
    :param job_name: Kubernetes job name.
    :param job_namespace: Kubernetes job namespace.
    """
    # How often to poll run status when waiting on a run
    POLL_STATUS_INTERVAL = 5

    def __init__(self, mlflow_run_id, job_name, job_namespace):
        super(KubernetesSubmittedRun, self).__init__()
        self._mlflow_run_id = mlflow_run_id
        self._job_name = job_name
        self._job_namespace = job_namespace
        self._status = RunStatus.SCHEDULED
        self._status_lock = RLock()

    @property
    def run_id(self):
        return self._mlflow_run_id

    def wait(self):
        kube_api = kubernetes.client.BatchV1Api()
        while not RunStatus.is_terminated(self._update_status(kube_api)):
            time.sleep(self.POLL_STATUS_INTERVAL)

        return self._status == RunStatus.FINISHED

    def _update_status(self, kube_api=kubernetes.client.BatchV1Api()):
        api_response = kube_api.read_namespaced_job_status(name=self._job_name,
                                                           namespace=self._job_namespace,
                                                           pretty=True)
        status = api_response.status
        with self._status_lock:
            if RunStatus.is_terminated(self._status):
                return self._status
            if self._status == RunStatus.SCHEDULED:
                if api_response.status.start_time is None:
                    _logger.info("Waiting for Job to start")
                else:
                    _logger.info("Job started.")
                    self._status = RunStatus.RUNNING
            if status.conditions is not None:
                for condition in status.conditions:
                    if condition.status == "True":
                        _logger.info(condition.message)
                        if condition.type == "Failed":
                            self._status = RunStatus.FAILED
                        elif condition.type == "Complete":
                            self._status = RunStatus.FINISHED
        return self._status

    def get_status(self):
        status = self._status
        return status if RunStatus.is_terminated(status) else self._update_status()

    def cancel(self):
        with self._status_lock:
            if not RunStatus.is_terminated(self._status):
                _logger.info("Cancelling job.")
                kube_api = kubernetes.client.BatchV1Api()
                kube_api.delete_namespaced_job(name=self._job_name,
                                               namespace=self._job_namespace,
                                               body=kubernetes.client.V1DeleteOptions(),
                                               pretty=True)
                self._status = RunStatus.KILLED
                _logger.info("Job cancelled.")
            else:
                _logger.info("Attempting to cancel a job that is already terminated.")
