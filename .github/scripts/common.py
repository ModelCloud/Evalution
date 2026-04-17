import os
import re
import subprocess
import time
import urllib.parse
import urllib.error
import urllib.request
import json
from pathlib import Path
from device_smi import Device


GPU_DISABLED_MARKER = re.compile(r"^# GPU=-1\s*$", re.MULTILINE)


def now_ms() -> int:
    return time.time_ns() // 1_000_000


def fetch_text(url: str, *, timeout: float, suppress_error: bool = False) -> str:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if suppress_error:
            print(f"Request failed for {url}: {exc}")
            return ""
        raise


def fetch_with_retry(url: str, *, timeout: float, retries: int, retry_delay: float) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return fetch_text(url, timeout=timeout)
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)
    if last_error is not None:
        print(f"Request failed after retries: {last_error}")
    return ""


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def request_json(
    url: str,
    *,
    method: str = "GET",
    body: dict[str, object] | None = None,
    timeout: float,
) -> object | None:
    data = None
    headers: dict[str, str] = {}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"

    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
        if not raw.strip():
            return None
        return json.loads(raw)


def request_json_with_retry(
    url: str,
    *,
    method: str,
    body: dict[str, object] | None,
    timeout: float,
    retries: int,
    retry_delay: float,
) -> object | None:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            return request_json(url, method=method, body=body, timeout=timeout)
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                time.sleep(retry_delay)
    if last_error is not None:
        print(f"Request failed after retries: {last_error}")
    return None


def append_github_env(name: str, value: str) -> None:
    _append_github_file(os.environ.get("GITHUB_ENV"), name, value)


def append_github_output(name: str, value: str) -> None:
    _append_github_file(os.environ.get("GITHUB_OUTPUT"), name, value)


def _append_github_file(target: str | None, name: str, value: str) -> None:
    if not target:
        return
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(f"{name}={value}\n")


def normalize_test_file(test_file: str) -> str:
    value = test_file.strip()
    if not value:
        raise ValueError("test file must not be empty")
    if value.startswith("tests/") and value.endswith(".py"):
        return value
    if value.endswith(".py"):
        return f"tests/{value.lstrip('./')}"
    return f"tests/{value.removeprefix('tests/').replace('.', '/')}.py"


def to_safe_name(test_file: str) -> str:
    return normalize_test_file(test_file).replace("/", "__").replace(".", "_")


def test_requires_gpu(test_file: str) -> bool:
    try:
        contents = Path(normalize_test_file(test_file)).read_text(encoding="utf-8")
    except OSError:
        return True
    return GPU_DISABLED_MARKER.search(contents) is None


def quote_url_value(value: str) -> str:
    return urllib.parse.quote(value, safe="")


def build_server_info() -> dict[str, str]:
    os_info = Device("os")
    cpu_model = Device("cpu").model
    platform_name = (
            os.environ.get("GPU_PLATFORM")
            or cpu_model
    )
    return {
        "platform": platform_name,
        "arch": os_info.arch,
        "system": os_info.name,
    }


def query_gpu_inventory() -> list[dict[str, object]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free,memory.total,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    gpus: list[dict[str, object]] = []
    for line in result.stdout.splitlines():
        if not line.strip():
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 12:
            raise ValueError(f"Unexpected nvidia-smi output line: {line}")
        gpus.append(
            {
                "index": int(fields[0]),
                "uuid": fields[1],
                "util": int(fields[2]),
                "memUsed": int(fields[3]),
                "memFree": int(fields[4]),
                "memTotal": int(fields[5]),
                "driver": fields[6],
                "name": fields[7],
                "serial": fields[8],
                "displayActive": fields[9].lower() == "enabled",
                "displayMode": fields[10].lower() == "enabled",
                "temperature": int(fields[11]),
            }
        )
    return gpus


def build_get_request(*, runner_name: str, run_id: str, test_name: str, count: str) -> dict[str, object]:
    return {
        "server": build_server_info(runner_name),
        "job": {
            "jobId": int(run_id),
            "count": int(count),
            "test": test_name,
            "exclusive": True,
            "timestamp": now_ms(),
        },
        "gpu": query_gpu_inventory(),
    }


def build_job_request(*, runner_name: str, run_id: str, test_name: str) -> dict[str, object]:
    return {
        "server": build_server_info(runner_name),
        "job": {
            "jobId": int(run_id),
            "test": test_name,
        },
    }


def extract_gpu_ids(response: object | None) -> str:
    if not isinstance(response, dict):
        return ""
    gpu_ids = response.get("gpuIds")
    return gpu_ids.strip() if isinstance(gpu_ids, str) else ""


def format_info_url(base_url: str, platform_name: str) -> str:
    query = urllib.parse.urlencode({"platform": platform_name, "plain": "true"})
    return f"{normalize_base_url(base_url)}/info?{query}"
