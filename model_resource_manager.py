import os
import logging
import requests
import asyncio
from pathlib import Path
from kubernetes import client, config
from kubernetes.stream import stream
from concurrent.futures import ThreadPoolExecutor

# ------------------------------------------------------------
# Logging Configuration
# ------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ------------------------------------------------------------
# Environment Configuration
# ------------------------------------------------------------
NAMESPACE = os.environ.get("NAMESPACE")
DEPLOYMENT_NAME = os.environ.get("DEPLOYMENT_NAME")
MAX_RSS_GB = float(os.environ.get("MAX_RSS_GB", 48.0))
ADMIN_PORT = int(os.environ.get("ADMIN_PORT", 7070))
STATS_ENDPOINT = os.environ.get("STATS_ENDPOINT", "/stats")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() in ("true", "1", "yes")
MAINTENANCE_FILE_PATH = os.environ.get("MAINTENANCE_FILE_PATH", "/app/dist/maintenance/maintenance_file")

# Model cleanup configuration
MODELS_DIR = os.environ.get("MODELS_DIR", "/app/dist/path")
MIN_SIZE_MB = float(os.environ.get("MIN_SIZE_MB", 10))
MAX_DAYS_OLD = int(os.environ.get("MAX_DAYS_OLD", 5))
MODELS_DIR_MAX_SIZE_GB = float(os.environ.get("MODELS_DIR_MAX_SIZE_GB", 100))  # Typically aligned with emptyDir 100Gi

# Pod label configuration
PREDICTOR_APP_LABEL = os.environ.get("PREDICTOR_APP_LABEL", "ml-pod-generic")

# Maintenance command
MAINTENANCE_CMD = ["/bin/sh", "-c", f"touch {MAINTENANCE_FILE_PATH}"]

# ------------------------------------------------------------
# Execute command synchronously in pod
# ------------------------------------------------------------
def exec_command_in_pod(core_v1, pod_name, namespace, command):
    """Run a shell command inside a Kubernetes pod and return the output."""
    try:
        response = stream(
            core_v1.connect_get_namespaced_pod_exec,
            name=pod_name,
            namespace=namespace,
            command=["/bin/sh", "-c", command],
            stderr=True,
            stdin=False,
            stdout=True,
            tty=False
        )
        return response
    except client.ApiException as e:
        logging.error(f"Kubernetes API error while executing command in pod {pod_name}: {e}")
        return ""
    except Exception as e:
        logging.error(f"Unexpected error while executing command in pod {pod_name}: {e}")
        return ""

# ------------------------------------------------------------
# Get RSS memory usage from pod
# ------------------------------------------------------------
def get_pod_rss(pod_ip):
    """Fetch the RSS memory (in GB) of a pod via its admin/stats endpoint."""
    if not pod_ip:
        logging.error("Pod IP cannot be empty")
        return 0

    stats_url = f"http://{pod_ip}:{ADMIN_PORT}{STATS_ENDPOINT}"
    try:
        response = requests.get(stats_url, timeout=15)
        response.raise_for_status()
        rss = response.json().get("stats", {}).get("self.mem:rss")
        if rss is None:
            logging.error(f"No RSS data in stats response from {stats_url}")
            return 0
        return float(rss) / (1024 * 1024)  # Convert KB → GB
    except requests.RequestException as e:
        logging.error(f"Could not fetch stats from {stats_url}: {e}")
        return 0
    except (ValueError, KeyError) as e:
        logging.error(f"Could not parse RSS from stats response at {stats_url}: {e}")
        return 0

# ------------------------------------------------------------
# Get directory size inside pod
# ------------------------------------------------------------
def get_directory_size_gb(pod_name, namespace, core_v1):
    """Calculate total size of MODELS_DIR in GB for a pod."""
    check_size_cmd = f"du -sb {MODELS_DIR} | cut -f1"
    try:
        resp = exec_command_in_pod(core_v1, pod_name, namespace, check_size_cmd)
        total_bytes = int(resp.strip())
        return total_bytes / (1024 * 1024 * 1024)
    except (ValueError, AttributeError) as e:
        logging.error(f"Error parsing directory size for pod {pod_name}: {e}")
        return 0
    except Exception as e:
        logging.error(f"Error getting directory size for pod {pod_name}: {e}")
        return 0

# ------------------------------------------------------------
# Async exec command in pod
# ------------------------------------------------------------
async def exec_command_in_pod_async(core_v1, pod_name, namespace, command, timeout_seconds=30):
    """Execute a command in a pod asynchronously."""
    try:
        resp = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: stream(
                core_v1.connect_get_namespaced_pod_exec,
                name=pod_name,
                namespace=namespace,
                command=command,
                stderr=True,
                stdin=False,
                stdout=True,
                tty=False,
                _request_timeout=timeout_seconds
            )
        )
        return True, resp.strip()
    except client.ApiException as e:
        logging.error(f"API error executing command in pod {pod_name}: {e}")
        return False, str(e)
    except Exception as e:
        logging.error(f"Error executing command in pod {pod_name}: {e}")
        return False, str(e)

# ------------------------------------------------------------
# Cleanup Logic
# ------------------------------------------------------------
async def cleanup_old_models():
    """Clean up old/large model folders in predictor pods concurrently."""
    if not NAMESPACE or not PREDICTOR_APP_LABEL:
        logging.error("NAMESPACE or PREDICTOR_APP_LABEL environment variable not set")
        return

    if DRY_RUN:
        logging.info("DRY_RUN mode enabled. No changes will be made.")

    try:
        # Load Kube config
        try:
            config.load_incluster_config()
        except config.ConfigException:
            config.load_kube_config()

        core_v1 = client.CoreV1Api()

        # Find predictor pods
        label_selector = f"app={PREDICTOR_APP_LABEL}"
        logging.info(f"Looking for pods with label: {label_selector}")

        pods = core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector)
        if not pods.items:
            logging.error(f"No predictor pods found with label {label_selector}")
            return

        logging.info(f"Found {len(pods.items)} matching pods")

        running_pods = [pod for pod in pods.items if pod.status.phase == "Running" and pod.status.pod_ip]
        if not running_pods:
            logging.error("No running predictor pods found with valid IPs")
            return

        async def cleanup_pod(pod):
            pod_name = pod.metadata.name
            logging.info(f"Processing cleanup for pod {pod_name}")

            check_dir_cmd = f"[ -d {MODELS_DIR} ] && echo 'exists' || echo 'not found'"
            success, resp = await exec_command_in_pod_async(core_v1, pod_name, NAMESPACE, ["/bin/sh", "-c", check_dir_cmd])
            if not success or resp.strip() != 'exists':
                logging.warning(f"{MODELS_DIR} does not exist in pod {pod_name}. Skipping.")
                return

            current_size_gb = get_directory_size_gb(pod_name, NAMESPACE, core_v1)
            if current_size_gb <= MODELS_DIR_MAX_SIZE_GB:
                logging.info(f"Pod {pod_name}: Size {current_size_gb:.2f}GB < threshold {MODELS_DIR_MAX_SIZE_GB}GB. Skipping cleanup.")
                return

            logging.info(f"Pod {pod_name}: Starting cleanup. Current size: {current_size_gb:.2f}GB")

            min_size_kb = MIN_SIZE_MB * 1024
            cleanup_cmd = [
                "/bin/sh", "-c",
                f"cd {MODELS_DIR} && "
                f"find . -mindepth 1 -maxdepth 1 -type d -print0 | "
                f"du --files0-from=- -s | sort -k1nr,1 | awk '$1 > {min_size_kb}' | cut -f2 | "
                f"xargs -I{{}} sh -c 'find {{}} -type f -mtime -{MAX_DAYS_OLD} -printf \"%H\\n\" | grep -q . || echo {{}}' | "
                f"xargs -I{{}} rm -rf {{}}"
            ]

            if DRY_RUN:
                logging.info(f"Pod {pod_name}: Dry run — would execute cleanup command: {cleanup_cmd}")
                return

            success, resp = await exec_command_in_pod_async(core_v1, pod_name, NAMESPACE, cleanup_cmd)
            if not success:
                logging.error(f"Pod {pod_name}: Cleanup failed: {resp}")
                return

            final_size_gb = get_directory_size_gb(pod_name, NAMESPACE, core_v1)
            logging.info(f"Pod {pod_name}: Cleanup complete. New size: {final_size_gb:.2f}GB")

        await asyncio.gather(*[cleanup_pod(p) for p in running_pods], return_exceptions=True)

    except Exception as e:
        logging.error(f"Error during cleanup: {e}")

def main():
    """Main entry point."""
    required_vars = {
        "NAMESPACE": NAMESPACE,
        "DEPLOYMENT_NAME": DEPLOYMENT_NAME,
        "MAINTENANCE_FILE_PATH": MAINTENANCE_FILE_PATH,
        "PREDICTOR_APP_LABEL": PREDICTOR_APP_LABEL,
        "MODELS_DIR": MODELS_DIR
    }
    missing_vars = [k for k, v in required_vars.items() if not v]
    if missing_vars:
        logging.error(f"Missing environment variables: {', '.join(missing_vars)}")
        return

    # Validate numeric values
    if MAX_RSS_GB <= 0 or MIN_SIZE_MB <= 0 or MAX_DAYS_OLD <= 0 or MODELS_DIR_MAX_SIZE_GB <= 0:
        logging.error("Numeric environment variables must be positive.")
        return

    # Run model cleanup
    asyncio.run(cleanup_old_models())

    logging.info(f"Starting pod restart check for deployment '{DEPLOYMENT_NAME}' in namespace '{NAMESPACE}'.")

    # Load Kubernetes config
    try:
        config.load_incluster_config()
    except config.ConfigException:
        config.load_kube_config()

    apps_v1 = client.AppsV1Api()
    core_v1 = client.CoreV1Api()

    try:
        deployment = apps_v1.read_namespaced_deployment(name=DEPLOYMENT_NAME, namespace=NAMESPACE)
        desired_replicas = deployment.spec.replicas
        actual_replicas = deployment.status.ready_replicas or 0

        rolling_update = getattr(deployment.spec.strategy, "rolling_update", None)
        max_unavailable_str = getattr(rolling_update, "max_unavailable", 1)
        if isinstance(max_unavailable_str, str) and "%" in max_unavailable_str:
            max_unavailable_pods = int(desired_replicas * (int(max_unavailable_str.strip("%")) / 100.0))
        else:
            max_unavailable_pods = int(max_unavailable_str or 1)

        currently_unavailable = desired_replicas - actual_replicas
        allowed_restarts = max(0, max_unavailable_pods - currently_unavailable)
        if allowed_restarts == 0:
            logging.info("Deployment fully available but 0 restarts allowed. Setting to 1.")
            allowed_restarts = 1

        logging.info(f"Deployment status: Desired={desired_replicas}, Actual={actual_replicas}, MaxUnavailable={max_unavailable_pods}")
        logging.info(f"Allowed restarts this cycle: {allowed_restarts}")

        if allowed_restarts <= 0:
            logging.warning("No restarts allowed in this cycle.")
            return

        # Identify pods with high memory
        label_selector = f"app.kubernetes.io/name={deployment.spec.selector.match_labels.get('app.kubernetes.io/name', PREDICTOR_APP_LABEL)}"
        logging.info(f"Looking for pods with label selector: {label_selector}")
        pods = core_v1.list_namespaced_pod(namespace=NAMESPACE, label_selector=label_selector)
        logging.info(f"Found {len(pods.items)} pods matching selector")

        pods_to_restart = []
        for pod in pods.items:
            if pod.status.phase == "Running" and pod.status.pod_ip:
                rss_gb = get_pod_rss(pod.status.pod_ip)
                logging.info(f"Pod {pod.metadata.name} has RSS: {rss_gb:.2f} GB")
                if rss_gb > MAX_RSS_GB:
                    pods_to_restart.append(pod.metadata.name)
            else:
                logging.warning(f"Pod {pod.metadata.name} is not running or missing IP.")

        if not pods_to_restart:
            logging.info("No pods exceed memory threshold.")
            return

        if DRY_RUN:
            logging.info(f"Dry run mode: Would restart pods {pods_to_restart}")
            return

        restarted_count = 0
        for pod_name in pods_to_restart:
            if restarted_count >= allowed_restarts:
                logging.warning("Reached restart limit for this cycle.")
                break

            try:
                logging.info(f"Putting pod {pod_name} into maintenance mode...")
                resp = stream(core_v1.connect_get_namespaced_pod_exec,
                              name=pod_name,
                              namespace=NAMESPACE,
                              command=MAINTENANCE_CMD,
                              stderr=True, stdin=False,
                              stdout=True, tty=False)
                logging.info(f"Pod {pod_name} maintenance response: {resp}")
                restarted_count += 1
            except client.ApiException as e:
                logging.error(f"Failed to put pod {pod_name} in maintenance: {e}")

        logging.info(f"Completed restart sequence. Restarted {restarted_count} pod(s).")

    except client.ApiException as e:
        logging.error(f"Kubernetes API error: {e}")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
