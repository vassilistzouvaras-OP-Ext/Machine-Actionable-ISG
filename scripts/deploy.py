#!/usr/bin/env python3
"""
Generic Open WebUI + Open WebUI Pipelines deployer.

This script creates a named local deployment that can coexist with other
Open WebUI/Pipelines containers. It mounts user-provided pipeline files or
directories and source directories read-only, starts Docker Compose, copies
the selected pipeline *.py files into the Pipelines runtime volume, and
restarts the Pipelines container.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import platform
import re
import secrets
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEPLOYMENTS_DIR = PROJECT_ROOT / ".deployments"
DEFAULT_OPENWEBUI_PORT = 8080
DEFAULT_PIPELINES_PORT = 9090
PIPELINES_INTERNAL_PORT = 9099
OPENWEBUI_INTERNAL_PORT = 8080
# The CUDA image is pinned to a Pascal-compatible PyTorch build and has been
# verified locally on a GTX 1080 (sm_61).
MIN_CUDA_COMPUTE_CAPABILITY = 6.1


class DeployError(RuntimeError):
    """A user-facing deployment error."""


@dataclass(frozen=True)
class PipelineSelection:
    """Resolved pipeline input and the Python files that should be deployed."""

    source_path: Path
    mount_dir: Path
    entry_files: Tuple[Path, ...]
    files_to_copy: Tuple[Path, ...]


@dataclass(frozen=True)
class GpuInfo:
    """GPU inventory entry from nvidia-smi."""

    index: int
    name: str
    free_memory_mb: int
    total_memory_mb: int
    compute_capability: float


def run(
    command: Sequence[str],
    *,
    cwd: Optional[Path] = None,
    check: bool = True,
    capture: bool = True,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        list(command),
        cwd=str(cwd) if cwd else None,
        check=check,
        capture_output=capture,
        text=True,
    )


def normalize_name(raw_name: str) -> str:
    name = raw_name.strip().lower()
    name = re.sub(r"[^a-z0-9-]+", "-", name)
    name = re.sub(r"-{2,}", "-", name).strip("-")
    if not name:
        raise DeployError("Instance name must contain at least one letter or number.")
    if not re.match(r"^[a-z0-9][a-z0-9-]*$", name):
        raise DeployError("Instance name must use letters, numbers, and dashes only.")
    return name


def ensure_directory(path_value: str, label: str) -> Path:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise DeployError(f"{label} does not exist: {path}")
    if not path.is_dir():
        raise DeployError(f"{label} must be a directory: {path}")
    return path


def discover_pipeline_files(selections: Iterable[PipelineSelection]) -> List[Path]:
    files: List[Path] = []
    for selection in selections:
        files.extend(selection.entry_files)
    return files


def _resolve_local_module(module_name: str, *, current_dir: Path, mount_dir: Path) -> Optional[Path]:
    module_path = Path(*[part for part in module_name.split(".") if part])
    if not module_path.parts:
        return None

    candidates = [
        mount_dir / module_path.with_suffix(".py"),
        current_dir / module_path.with_suffix(".py"),
        mount_dir / module_path / "__init__.py",
        current_dir / module_path / "__init__.py",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def _resolve_relative_module(module_name: str, level: int, *, current_file: Path, mount_dir: Path) -> Optional[Path]:
    base_dir = current_file.parent
    for _ in range(max(level - 1, 0)):
        base_dir = base_dir.parent
    module_path = Path(*[part for part in module_name.split(".") if part]) if module_name else Path()
    candidates = [
        (base_dir / module_path).with_suffix(".py"),
        base_dir / module_path / "__init__.py",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(mount_dir)
            except ValueError:
                continue
            return resolved
    return None


def _discover_python_dependencies(entry_file: Path, mount_dir: Path) -> List[Path]:
    discovered: List[Path] = []
    pending = [entry_file.resolve()]
    seen: set[Path] = set()
    resolved_mount_dir = mount_dir.resolve()

    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        discovered.append(current)

        try:
            source = current.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DeployError(f"Could not read pipeline file as UTF-8: {current}") from exc

        try:
            tree = ast.parse(source, filename=str(current))
        except SyntaxError as exc:
            raise DeployError(f"Could not parse pipeline file {current}: {exc}") from exc

        referenced: set[Path] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    candidate = _resolve_local_module(alias.name, current_dir=current.parent, mount_dir=mount_dir)
                    if candidate is not None:
                        referenced.add(candidate)
            elif isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                if node.level:
                    candidate = _resolve_relative_module(
                        module_name,
                        node.level,
                        current_file=current,
                        mount_dir=mount_dir,
                    )
                    if candidate is not None:
                        referenced.add(candidate)
                    for alias in node.names:
                        alias_candidate = _resolve_relative_module(
                            f"{module_name}.{alias.name}" if module_name else alias.name,
                            node.level,
                            current_file=current,
                            mount_dir=mount_dir,
                        )
                        if alias_candidate is not None:
                            referenced.add(alias_candidate)
                elif module_name:
                    candidate = _resolve_local_module(module_name, current_dir=current.parent, mount_dir=mount_dir)
                    if candidate is not None:
                        referenced.add(candidate)
                    for alias in node.names:
                        alias_candidate = _resolve_local_module(
                            f"{module_name}.{alias.name}",
                            current_dir=current.parent,
                            mount_dir=mount_dir,
                        )
                        if alias_candidate is not None:
                            referenced.add(alias_candidate)

        for raw_match in re.findall(r"['\"]([^'\"]+\.py)['\"]", source):
            sibling_candidate = current.parent / Path(raw_match).name
            if sibling_candidate.exists() and sibling_candidate.is_file():
                try:
                    resolved_candidate = sibling_candidate.resolve()
                    resolved_candidate.relative_to(resolved_mount_dir)
                except ValueError:
                    continue
                referenced.add(resolved_candidate)

        for candidate in sorted(referenced):
            if candidate not in seen:
                pending.append(candidate)

    return sorted(discovered)


def resolve_pipeline_selection(path_value: str) -> PipelineSelection:
    path = Path(path_value).expanduser().resolve()
    if not path.exists():
        raise DeployError(f"Pipeline path does not exist: {path}")

    if path.is_dir():
        entry_files = tuple(sorted(child.resolve() for child in path.glob("*.py") if child.is_file()))
        if not entry_files:
            raise DeployError(f"No *.py pipeline files were found in the provided pipeline directory: {path}")
        files_to_copy: set[Path] = set()
        for entry_file in entry_files:
            files_to_copy.update(_discover_python_dependencies(entry_file, path))
        return PipelineSelection(
            source_path=path,
            mount_dir=path,
            entry_files=entry_files,
            files_to_copy=tuple(sorted(files_to_copy)),
        )

    if path.is_file():
        if path.suffix != ".py":
            raise DeployError(f"Pipeline file must be a *.py file: {path}")
        mount_dir = path.parent
        files_to_copy = tuple(_discover_python_dependencies(path, mount_dir))
        return PipelineSelection(
            source_path=path,
            mount_dir=mount_dir,
            entry_files=(path,),
            files_to_copy=files_to_copy,
        )

    raise DeployError(f"Pipeline path must be a directory or *.py file: {path}")


def infer_default_model(pipeline_files: Sequence[Path]) -> str:
    if not pipeline_files:
        return ""
    return pipeline_files[0].stem


def build_pipeline_bootstrap_spec(selections: Sequence[PipelineSelection]) -> str:
    spec = []
    for idx, selection in enumerate(selections, start=1):
        spec.append(
            {
                "root": f"/app/imported-pipelines/pipeline_{idx}",
                "files": [
                    path.relative_to(selection.mount_dir).as_posix()
                    for path in selection.files_to_copy
                ],
            }
        )
    return json.dumps(spec, separators=(",", ":"))


def port_is_available(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_available_port(preferred_port: int, unavailable: Optional[set[int]] = None) -> int:
    unavailable = unavailable or set()
    if preferred_port < 1 or preferred_port > 65535:
        raise DeployError(f"Preferred port must be between 1 and 65535: {preferred_port}")

    for offset in range(0, 65535):
        candidates = [preferred_port] if offset == 0 else [preferred_port + offset, preferred_port - offset]
        for port in candidates:
            if 1 <= port <= 65535 and port not in unavailable and port_is_available(port):
                return port
    raise DeployError(f"No available localhost port found near {preferred_port}.")


def validate_fixed_port(port: int, label: str) -> int:
    if port < 1 or port > 65535:
        raise DeployError(f"{label} must be between 1 and 65535.")
    if not port_is_available(port):
        raise DeployError(f"{label} {port} is already in use on localhost.")
    return port


def docker_is_available() -> None:
    if not shutil.which("docker"):
        raise DeployError("Docker CLI was not found in PATH.")
    try:
        run(["docker", "--version"])
        run(["docker", "compose", "version"])
        run(["docker", "info"])
    except subprocess.CalledProcessError as exc:
        raise DeployError(f"Docker is not available or the daemon is not reachable: {exc.stderr or exc.stdout}") from exc


def docker_container_exists(container_name: str) -> bool:
    result = run(
        ["docker", "ps", "-a", "--filter", f"name={container_name}", "--format", "{{.Names}}"],
        check=False,
    )
    return container_name in result.stdout.splitlines()


def docker_image_exists(image_name: str) -> bool:
    result = run(["docker", "image", "inspect", image_name], check=False)
    return result.returncode == 0


def command_succeeds(command: Sequence[str]) -> bool:
    try:
        return run(command, check=False).returncode == 0
    except OSError:
        return False


def docker_has_nvidia_runtime() -> bool:
    result = run(["docker", "info", "--format", "{{json .Runtimes}}"], check=False)
    output = (result.stdout + result.stderr).lower()
    return result.returncode == 0 and "nvidia" in output


def get_nvidia_gpus() -> List[GpuInfo]:
    result = run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.free,memory.total,compute_cap",
            "--format=csv,noheader,nounits",
        ],
        check=False,
    )
    if result.returncode != 0:
        return []

    gpus: List[GpuInfo] = []
    for line in result.stdout.splitlines():
        value = line.strip()
        if not value:
            continue
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != 5:
            continue
        try:
            gpus.append(
                GpuInfo(
                    index=int(parts[0]),
                    name=parts[1],
                    free_memory_mb=int(parts[2]),
                    total_memory_mb=int(parts[3]),
                    compute_capability=float(parts[4]),
                )
            )
        except ValueError:
            continue
    return gpus


def detect_accelerator() -> Tuple[str, str, str, Optional[GpuInfo]]:
    system = platform.system().lower()

    if command_succeeds(["nvidia-smi", "-L"]):
        gpus = get_nvidia_gpus()
        compatible_gpus = [
            gpu for gpu in gpus if gpu.compute_capability >= MIN_CUDA_COMPUTE_CAPABILITY
        ]
        if gpus and not compatible_gpus:
            capabilities = ", ".join(str(gpu.compute_capability) for gpu in gpus)
            return (
                "cpu",
                "Dockerfile.pipelines",
                (
                    "NVIDIA GPU detected, but compute capability "
                    f"{capabilities} is below the minimum supported CUDA capability "
                    f"{MIN_CUDA_COMPUTE_CAPABILITY:.1f} for the current PyTorch image; "
                    "using CPU pipelines image."
                ),
                None,
            )
        selected_gpu = max(
            compatible_gpus,
            key=lambda gpu: (gpu.free_memory_mb, gpu.total_memory_mb, -gpu.index),
        ) if compatible_gpus else None
        if docker_has_nvidia_runtime():
            if selected_gpu is None:
                return (
                    "cpu",
                    "Dockerfile.pipelines",
                    "NVIDIA GPU detected, but no compatible GPU details were available; using CPU pipelines image.",
                    None,
                )
            return (
                "cuda",
                "Dockerfile.pipelines.cuda",
                (
                    "NVIDIA GPU and Docker NVIDIA runtime detected; using CUDA pipelines image "
                    f"on GPU {selected_gpu.index} ({selected_gpu.name}) with "
                    f"{selected_gpu.free_memory_mb} MiB free VRAM out of "
                    f"{selected_gpu.total_memory_mb} MiB."
                ),
                selected_gpu,
            )
        return (
            "cpu",
            "Dockerfile.pipelines",
            "NVIDIA GPU detected, but Docker NVIDIA runtime/toolkit was not detected; using CPU pipelines image.",
            None,
        )

    if system == "darwin":
        machine = platform.machine().lower()
        apple_silicon = machine in {"arm64", "aarch64"}
        reason = (
            "Apple Silicon detected, but Docker Desktop for Mac does not expose Apple GPU/MPS "
            "to Linux containers; using CPU pipelines image."
            if apple_silicon
            else "macOS detected without Docker-accessible CUDA GPU; using CPU pipelines image."
        )
        return ("cpu", "Dockerfile.pipelines", reason, None)

    return ("cpu", "Dockerfile.pipelines", "No Docker-accessible GPU detected; using CPU pipelines image.", None)


def wait_for_container(container_name: str, timeout: int = 120) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_name],
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip() == "true":
            return
        time.sleep(2)
    raise DeployError(f"Container did not start within {timeout} seconds: {container_name}")


def yaml_quote(value: str) -> str:
    return json.dumps(value)


def env_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace("\n", "\\n")


def build_access_urls(
    *,
    openwebui_port: int,
    pipelines_port: int,
    public_port: bool,
    public_host: str = "",
) -> dict[str, str]:
    local_openwebui_url = f"http://localhost:{openwebui_port}"
    local_pipelines_url = f"http://localhost:{pipelines_port}"
    remote_host = public_host.strip() or "<host-ip-or-dns>"
    remote_openwebui_url = f"http://{remote_host}:{openwebui_port}" if public_port else ""
    remote_pipelines_url = f"http://{remote_host}:{pipelines_port}" if public_port else ""
    return {
        "bind_host": "0.0.0.0" if public_port else "127.0.0.1",
        "public_host": remote_host if public_port else "",
        "openwebui_url": local_openwebui_url,
        "pipelines_url": local_pipelines_url,
        "remote_openwebui_url": remote_openwebui_url,
        "remote_pipelines_url": remote_pipelines_url,
    }


def generate_compose(
    *,
    name: str,
    openwebui_port: int,
    pipelines_port: int,
    pipeline_selections: Sequence[PipelineSelection],
    source_dirs: Sequence[Path],
    accelerator: str,
    pipelines_dockerfile: str,
    selected_gpu: Optional[GpuInfo],
    public_port: bool = False,
) -> str:
    openwebui_container = f"{name}-open-webui"
    pipelines_container = f"{name}-pipelines"
    network_name = f"{name}-net"
    openwebui_volume = f"{name}-openwebui-data"
    # Bind-mounted (not a Docker-managed named volume) so the ingested chunks
    # and FAISS/embedding caches built under /app/pipelines live as plain
    # host files. That makes them inspectable and copyable (rsync/scp) to
    # another Docker host, so embeddings only need to be computed once
    # instead of being recomputed from scratch on every new machine.
    pipelines_data_dir = DEPLOYMENTS_DIR / name / "pipelines_data"
    host_prefix = "" if public_port else "127.0.0.1:"

    pipeline_mounts = [
        f"{selection.mount_dir}:/app/imported-pipelines/pipeline_{idx}:ro"
        for idx, selection in enumerate(pipeline_selections, start=1)
    ]
    source_mounts = [
        f"{path}:/app/sources/source_{idx}:ro"
        for idx, path in enumerate(source_dirs, start=1)
    ]
    source_container_paths = ":".join(
        f"/app/sources/source_{idx}" for idx, _ in enumerate(source_dirs, start=1)
    )

    pipeline_volume_lines = [
        f"      - {yaml_quote(f'{pipelines_data_dir}:/app/pipelines')}",
        *[f"      - {yaml_quote(mount)}" for mount in pipeline_mounts],
        *[f"      - {yaml_quote(mount)}" for mount in source_mounts],
    ]
    gpu_lines = (
        [
            "    runtime: nvidia",
            "    environment:",
            "      - PIPELINES_API_KEY=${PIPELINES_API_KEY}",
            "      - OPENAI_API_KEY=${UPSTREAM_OPENAI_API_KEY}",
            "      - OPENAI_BASE_URL=${UPSTREAM_OPENAI_BASE_URL}",
            "      - OPENAI_API_BASE_URL=${UPSTREAM_OPENAI_BASE_URL}",
            "      - DEFAULT_MODEL=${DEFAULT_MODEL}",
            "      - PIPELINES_ACCELERATOR=${PIPELINES_ACCELERATOR:-cuda}",
            "      - PIPELINE_BOOTSTRAP_SPEC=${PIPELINE_BOOTSTRAP_SPEC}",
            f"      - SOURCE_DIRS={source_container_paths}",
            f"      - PIPELINE_SOURCE_DIRS={source_container_paths}",
            f"      - NVIDIA_VISIBLE_DEVICES=${{NVIDIA_VISIBLE_DEVICES:-{selected_gpu.index if selected_gpu is not None else 'all'}}}",
            "      - NVIDIA_DRIVER_CAPABILITIES=compute,utility",
        ]
        if accelerator == "cuda"
        else [
            "    environment:",
            "      - PIPELINES_API_KEY=${PIPELINES_API_KEY}",
            "      - OPENAI_API_KEY=${UPSTREAM_OPENAI_API_KEY}",
            "      - OPENAI_BASE_URL=${UPSTREAM_OPENAI_BASE_URL}",
            "      - OPENAI_API_BASE_URL=${UPSTREAM_OPENAI_BASE_URL}",
            "      - DEFAULT_MODEL=${DEFAULT_MODEL}",
            f"      - PIPELINES_ACCELERATOR={accelerator}",
            "      - PIPELINE_BOOTSTRAP_SPEC=${PIPELINE_BOOTSTRAP_SPEC}",
            f"      - SOURCE_DIRS={source_container_paths}",
            f"      - PIPELINE_SOURCE_DIRS={source_container_paths}",
        ]
    )
    image_name = f"generic-openwebui-pipelines:{accelerator}"

    return "\n".join(
        [
            "services:",
            "  open-webui:",
            "    image: ghcr.io/open-webui/open-webui:main",
            f"    container_name: {openwebui_container}",
            "    ports:",
            f"      - {yaml_quote(f'{host_prefix}{openwebui_port}:{OPENWEBUI_INTERNAL_PORT}')}",
            "    env_file:",
            "      - .env",
            "    environment:",
            f"      - OPENAI_API_BASE_URL=http://{pipelines_container}:{PIPELINES_INTERNAL_PORT}/v1",
            "      - OPENAI_API_KEY=${PIPELINES_API_KEY}",
            "      - ENABLE_OPENAI_API=true",
            "      - ENABLE_OLLAMA_API=false",
            "      - WEBUI_AUTH=true",
            "      - ENABLE_SIGNUP=true",
            "      - DEFAULT_USER_ROLE=user",
            "      - WEBUI_SECRET_KEY=${WEBUI_SECRET_KEY}",
            "      - WEBUI_ADMIN_EMAIL=${WEBUI_ADMIN_EMAIL}",
            "      - WEBUI_ADMIN_PASSWORD=${WEBUI_ADMIN_PASSWORD}",
            "      - DEFAULT_MODELS=${DEFAULT_MODEL}",
            "    volumes:",
            f"      - {yaml_quote(f'{openwebui_volume}:/app/backend/data')}",
            "    restart: unless-stopped",
            "    depends_on:",
            "      - pipelines",
            "    networks:",
            f"      - {network_name}",
            "",
            "  pipelines:",
            "    build:",
            f"      context: {yaml_quote(str(PROJECT_ROOT / 'docker_images'))}",
            f"      dockerfile: {pipelines_dockerfile}",
            f"    image: {image_name}",
            f"    container_name: {pipelines_container}",
            "    ports:",
            f"      - {yaml_quote(f'{host_prefix}{pipelines_port}:{PIPELINES_INTERNAL_PORT}')}",
            "    env_file:",
            "      - .env",
            *gpu_lines,
            "    volumes:",
            *pipeline_volume_lines,
            "    restart: unless-stopped",
            "    networks:",
            f"      - {network_name}",
            "",
            "volumes:",
            f"  {openwebui_volume}:",
            f"    name: {openwebui_volume}",
            "",
            "networks:",
            f"  {network_name}:",
            f"    name: {network_name}",
            "    driver: bridge",
            "",
        ]
    )


def write_deployment_files(
    *,
    deployment_dir: Path,
    compose_content: str,
    pipelines_key: str,
    webui_secret: str,
    admin_email: str,
    admin_password: str,
    upstream_openai_base_url: str,
    upstream_openai_api_key: str,
    upstream_openai_model: str,
    default_model: str,
    pipeline_bootstrap_spec: str,
    access_urls: dict[str, str],
    openwebui_port: int,
    pipelines_port: int,
    accelerator: str,
    accelerator_message: str,
    pipelines_dockerfile: str,
    selected_gpu: Optional[GpuInfo],
    openwebui_container: str,
    pipelines_container: str,
    source_dirs: Sequence[Path],
    public_port: bool,
) -> None:
    deployment_dir.mkdir(parents=True, exist_ok=True)
    (deployment_dir / "pipelines_data").mkdir(parents=True, exist_ok=True)
    (deployment_dir / "docker-compose.yaml").write_text(compose_content, encoding="utf-8")
    legacy_compose_path = deployment_dir / "docker compose.yaml"
    if legacy_compose_path.exists():
        legacy_compose_path.unlink()
    (deployment_dir / ".env").write_text(
        "\n".join(
            [
                f"PIPELINES_API_KEY={env_value(pipelines_key)}",
                f"WEBUI_SECRET_KEY={env_value(webui_secret)}",
                f"WEBUI_ADMIN_EMAIL={env_value(admin_email)}",
                f"WEBUI_ADMIN_PASSWORD={env_value(admin_password)}",
                f"UPSTREAM_OPENAI_BASE_URL={env_value(upstream_openai_base_url)}",
                f"UPSTREAM_OPENAI_API_KEY={env_value(upstream_openai_api_key)}",
                f"OPENAI_MODEL={env_value(upstream_openai_model)}",
                f"DEFAULT_MODEL={env_value(default_model)}",
                f"PIPELINE_BOOTSTRAP_SPEC={env_value(pipeline_bootstrap_spec)}",
                f"OPENWEBUI_PORT={openwebui_port}",
                f"PIPELINES_PORT={pipelines_port}",
                f"BIND_HOST={env_value(access_urls['bind_host'])}",
                f"PUBLIC_HOST={env_value(access_urls['public_host'])}",
                f"OPENWEBUI_URL={env_value(access_urls['openwebui_url'])}",
                f"PIPELINES_URL={env_value(access_urls['pipelines_url'])}",
                f"REMOTE_OPENWEBUI_URL={env_value(access_urls['remote_openwebui_url'])}",
                f"REMOTE_PIPELINES_URL={env_value(access_urls['remote_pipelines_url'])}",
                f"PIPELINES_ACCELERATOR={env_value(accelerator)}",
                f"PIPELINES_DOCKERFILE={env_value(pipelines_dockerfile)}",
                f"NVIDIA_VISIBLE_DEVICES={env_value(str(selected_gpu.index) if selected_gpu is not None else '')}",
                f"SELECTED_GPU_INDEX={env_value(str(selected_gpu.index) if selected_gpu is not None else '')}",
                f"SELECTED_GPU_NAME={env_value(selected_gpu.name if selected_gpu is not None else '')}",
                f"SELECTED_GPU_FREE_MB={env_value(str(selected_gpu.free_memory_mb) if selected_gpu is not None else '')}",
                f"SELECTED_GPU_TOTAL_MB={env_value(str(selected_gpu.total_memory_mb) if selected_gpu is not None else '')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    secrets_lines = [
        "Generic Open WebUI deployment",
        "",
        f"Port binding: {access_urls['bind_host']} ({'public' if public_port else 'local only'})",
        f"Local Open WebUI URL: {access_urls['openwebui_url']}",
        f"Local Pipelines API URL: {access_urls['pipelines_url']}",
        f"Open WebUI port: {openwebui_port}",
        f"Pipelines port: {pipelines_port}",
        f"Pipelines accelerator: {accelerator}",
        f"Pipelines Dockerfile: {pipelines_dockerfile}",
        f"Accelerator note: {accelerator_message}",
        (
            "Selected GPU: "
            f"{selected_gpu.index} ({selected_gpu.name}), "
            f"{selected_gpu.free_memory_mb} MiB free of {selected_gpu.total_memory_mb} MiB"
            if selected_gpu is not None
            else "Selected GPU: none"
        ),
        f"Open WebUI container: {openwebui_container}",
        f"Pipelines container: {pipelines_container}",
        "",
        f"Admin email: {admin_email}",
        f"Admin password: {admin_password}",
        f"Pipelines API key: {pipelines_key}",
        "",
        "Mounted source directories:",
    ]
    if public_port:
        secrets_lines.insert(5, f"Remote Open WebUI URL: {access_urls['remote_openwebui_url']}")
        secrets_lines.insert(6, f"Remote Pipelines API URL: {access_urls['remote_pipelines_url']}")
    for idx, source_dir in enumerate(source_dirs, start=1):
        secrets_lines.append(f"  {source_dir} -> /app/sources/source_{idx}")
    secrets_lines.append("")
    (deployment_dir / "secrets.txt").write_text("\n".join(secrets_lines), encoding="utf-8")
    os.chmod(deployment_dir / "secrets.txt", 0o600)
    os.chmod(deployment_dir / ".env", 0o600)


def bootstrap_pipelines(container_name: str) -> None:
    wait_for_container(container_name, timeout=300)
    copy_script = r"""
import json
import os
import shutil
from pathlib import Path

spec = json.loads(os.environ.get("PIPELINE_BOOTSTRAP_SPEC") or "[]")
destination_root = Path("/app/pipelines")
copied = []
desired_top_level_py = set()

for item in spec:
    for relative_path in item.get("files", []):
        relative = Path(relative_path)
        if len(relative.parts) == 1 and relative.suffix == ".py":
            desired_top_level_py.add(relative.name)

for existing_path in destination_root.glob("*.py"):
    if existing_path.name not in desired_top_level_py:
        existing_path.unlink()

for item in spec:
    root = Path(item["root"])
    for relative_path in item.get("files", []):
        source_path = root / relative_path
        if not source_path.is_file():
            continue
        destination_path = destination_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        copied.append(str(destination_path))

for entry in copied:
    print(entry)
"""
    result = run(["docker", "exec", container_name, "python3", "-c", copy_script], check=False)
    if result.returncode != 0:
        raise DeployError(f"Failed to copy pipeline files into {container_name}: {result.stderr}")
    run(["docker", "restart", container_name], capture=False)
    wait_for_container(container_name, timeout=300)
    copied = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not copied:
        raise DeployError("No pipeline files were copied into /app/pipelines.")


def compose_down_quiet(deployment_dir: Path, name: str) -> None:
    run(
        ["docker", "compose", "-p", name, "-f", "docker-compose.yaml", "down"],
        cwd=deployment_dir,
        check=False,
        capture=True,
    )


def compose_up_with_retry(
    *,
    deployment_dir: Path,
    name: str,
    pipeline_selections: Sequence[PipelineSelection],
    source_dirs: Sequence[Path],
    fixed_openwebui_port: Optional[int],
    fixed_pipelines_port: Optional[int],
    openwebui_port: int,
    pipelines_port: int,
    pipelines_key: str,
    webui_secret: str,
    admin_email: str,
    admin_password: str,
    upstream_openai_base_url: str,
    upstream_openai_api_key: str,
    upstream_openai_model: str,
    default_model: str,
    accelerator: str,
    accelerator_message: str,
    pipelines_dockerfile: str,
    selected_gpu: Optional[GpuInfo],
    rebuild: bool = False,
    public_port: bool = False,
    public_host: str = "",
    max_attempts: int = 10,
) -> Tuple[int, int]:
    attempted_ports: set[int] = set()
    current_openwebui_port = openwebui_port
    current_pipelines_port = pipelines_port
    pipeline_bootstrap_spec = build_pipeline_bootstrap_spec(pipeline_selections)

    for attempt in range(1, max_attempts + 1):
        openwebui_container = f"{name}-open-webui"
        pipelines_container = f"{name}-pipelines"
        access_urls = build_access_urls(
            openwebui_port=current_openwebui_port,
            pipelines_port=current_pipelines_port,
            public_port=public_port,
            public_host=public_host,
        )
        compose_content = generate_compose(
            name=name,
            openwebui_port=current_openwebui_port,
            pipelines_port=current_pipelines_port,
            pipeline_selections=pipeline_selections,
            source_dirs=source_dirs,
            accelerator=accelerator,
            pipelines_dockerfile=pipelines_dockerfile,
            selected_gpu=selected_gpu,
            public_port=public_port,
        )
        write_deployment_files(
            deployment_dir=deployment_dir,
            compose_content=compose_content,
            pipelines_key=pipelines_key,
            webui_secret=webui_secret,
            admin_email=admin_email,
            admin_password=admin_password,
            upstream_openai_base_url=upstream_openai_base_url,
            upstream_openai_api_key=upstream_openai_api_key,
            upstream_openai_model=upstream_openai_model,
            default_model=default_model,
            pipeline_bootstrap_spec=pipeline_bootstrap_spec,
            access_urls=access_urls,
            openwebui_port=current_openwebui_port,
            pipelines_port=current_pipelines_port,
            accelerator=accelerator,
            accelerator_message=accelerator_message,
            pipelines_dockerfile=pipelines_dockerfile,
            selected_gpu=selected_gpu,
            openwebui_container=openwebui_container,
            pipelines_container=pipelines_container,
            source_dirs=source_dirs,
            public_port=public_port,
        )

        image_name = f"generic-openwebui-pipelines:{accelerator}"
        compose_command = ["docker", "compose", "-p", name, "-f", "docker-compose.yaml", "up", "-d"]
        should_build = rebuild or not docker_image_exists(image_name)
        if should_build:
            compose_command.append("--build")
            if attempt == 1:
                print(f"Building pipelines image: {image_name}")
        elif attempt == 1:
            print(f"Reusing existing pipelines image: {image_name}")

        result = run(compose_command, cwd=deployment_dir, check=False, capture=True)
        output = f"{result.stdout}\n{result.stderr}"
        if result.returncode == 0:
            return current_openwebui_port, current_pipelines_port

        port_conflict = "port is already allocated" in output.lower() or "bind" in output.lower()
        fixed_openwebui = fixed_openwebui_port is not None
        fixed_pipelines = fixed_pipelines_port is not None

        if not port_conflict:
            raise DeployError(f"Docker Compose failed:\n{output.strip()}")

        if fixed_openwebui and fixed_pipelines:
            raise DeployError(f"Docker Compose failed because fixed ports are unavailable:\n{output.strip()}")

        compose_down_quiet(deployment_dir, name)
        attempted_ports.update({current_openwebui_port, current_pipelines_port})
        if not fixed_openwebui:
            current_openwebui_port = find_available_port(
                current_openwebui_port + 1,
                unavailable=attempted_ports,
            )
        if not fixed_pipelines:
            current_pipelines_port = find_available_port(
                current_pipelines_port + 1,
                unavailable=attempted_ports | {current_openwebui_port},
            )
        print(f"Port conflict during Docker startup; retrying with ports {current_openwebui_port}/{current_pipelines_port} (attempt {attempt + 1}).")

    raise DeployError("Docker Compose kept failing because of port conflicts.")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy a named Open WebUI + Pipelines stack from local pipeline files/directories and source directories.",
    )
    parser.add_argument("--name", required=True, help="Instance name used for containers, network, and volumes.")
    parser.add_argument(
        "--pipeline-dir",
        action="append",
        required=True,
        help="Pipeline directory or a single pipeline *.py file. Repeatable.",
    )
    parser.add_argument("--source-dir", action="append", required=True, help="Directory containing source files. Repeatable.")
    parser.add_argument("--openai-base-url", default="https://api.openai.com/v1", help="Upstream OpenAI-compatible base URL passed to pipelines.")
    parser.add_argument("--openai-api-key", default="", help="Upstream OpenAI-compatible API key passed to pipelines.")
    parser.add_argument("--openai-model", default="gpt-5.5", help="Upstream OpenAI chat model passed to pipelines.")
    parser.add_argument("--pipelines-api-key", default="", help="Pipelines API key used by Open WebUI. Generated if omitted.")
    parser.add_argument("--default-model", default="", help="Default Open WebUI model. Defaults to first pipeline filename stem.")
    parser.add_argument("--openwebui-port", type=int, help="Optional fixed localhost port for Open WebUI. Auto-selected if omitted.")
    parser.add_argument("--pipelines-port", type=int, help="Optional fixed localhost port for Pipelines. Auto-selected if omitted.")
    parser.add_argument("--admin-email", default="admin@example.local", help="Initial Open WebUI admin email for fresh deployments.")
    parser.add_argument("--admin-password", default="", help="Initial Open WebUI admin password. Generated if omitted.")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuilding the pipelines image even if a matching local image already exists.")
    parser.add_argument("--public-port", action="store_true", help="Bind to all interfaces to make the deployment publicly accessible.")
    parser.add_argument("--public-host", default="", help="Public IP address or DNS name to print in remote access URLs.")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    try:
        args = parse_args(argv)
        name = normalize_name(args.name)
        if name != args.name:
            print(f"Using normalized instance name: {name}")

        pipeline_selections = [resolve_pipeline_selection(path) for path in args.pipeline_dir]
        source_dirs = [ensure_directory(path, "Source directory") for path in args.source_dir]
        pipeline_files = discover_pipeline_files(pipeline_selections)
        if not pipeline_files:
            raise DeployError("No *.py pipeline files were selected for deployment.")

        default_model = args.default_model.strip() or infer_default_model(pipeline_files)
        if not default_model:
            raise DeployError("A default model could not be inferred. Pass --default-model.")

        docker_is_available()
        accelerator, pipelines_dockerfile, accelerator_message, selected_gpu = detect_accelerator()
        print(f"Accelerator: {accelerator}")
        print(f"  {accelerator_message}")

        openwebui_container = f"{name}-open-webui"
        pipelines_container = f"{name}-pipelines"
        for container_name in (openwebui_container, pipelines_container):
            if docker_container_exists(container_name):
                raise DeployError(
                    f"Container already exists: {container_name}. "
                    "Choose a different --name or remove the old deployment first."
                )

        openwebui_port = (
            validate_fixed_port(args.openwebui_port, "Open WebUI port")
            if args.openwebui_port is not None
            else find_available_port(DEFAULT_OPENWEBUI_PORT)
        )
        pipelines_port = (
            validate_fixed_port(args.pipelines_port, "Pipelines port")
            if args.pipelines_port is not None
            else find_available_port(DEFAULT_PIPELINES_PORT, unavailable={openwebui_port})
        )

        deployment_dir = DEPLOYMENTS_DIR / name
        pipelines_key = args.pipelines_api_key.strip() or secrets.token_hex(32)
        webui_secret = secrets.token_hex(32)
        admin_password = args.admin_password or secrets.token_urlsafe(18)

        final_openwebui_port, final_pipelines_port = compose_up_with_retry(
            deployment_dir=deployment_dir,
            name=name,
            pipeline_selections=pipeline_selections,
            source_dirs=source_dirs,
            fixed_openwebui_port=args.openwebui_port,
            fixed_pipelines_port=args.pipelines_port,
            openwebui_port=openwebui_port,
            pipelines_port=pipelines_port,
            pipelines_key=pipelines_key,
            webui_secret=webui_secret,
            admin_email=args.admin_email,
            admin_password=admin_password,
            upstream_openai_base_url=args.openai_base_url,
            upstream_openai_api_key=args.openai_api_key,
            upstream_openai_model=args.openai_model,
            default_model=default_model,
            accelerator=accelerator,
            accelerator_message=accelerator_message,
            pipelines_dockerfile=pipelines_dockerfile,
            selected_gpu=selected_gpu,
            rebuild=args.rebuild,
            public_port=args.public_port,
            public_host=args.public_host,
        )

        bootstrap_pipelines(pipelines_container)

        access_urls = build_access_urls(
            openwebui_port=final_openwebui_port,
            pipelines_port=final_pipelines_port,
            public_port=args.public_port,
            public_host=args.public_host,
        )
        print("\nDeployment ready")
        print(f"  Local Open WebUI: {access_urls['openwebui_url']}")
        print(f"  Local Pipelines API: {access_urls['pipelines_url']}")
        if args.public_port:
            print("  Port binding: 0.0.0.0 (public)")
            print(f"  Remote Open WebUI: {access_urls['remote_openwebui_url']}")
            print(f"  Remote Pipelines API: {access_urls['remote_pipelines_url']}")
        else:
            print("  Port binding: 127.0.0.1 (local only)")
        print(f"  Open WebUI container: {openwebui_container}")
        print(f"  Pipelines container: {pipelines_container}")
        print(f"  Pipelines accelerator: {accelerator}")
        if selected_gpu is not None:
            print(
                "  Selected GPU: "
                f"{selected_gpu.index} ({selected_gpu.name}), "
                f"{selected_gpu.free_memory_mb} MiB free of {selected_gpu.total_memory_mb} MiB"
            )
        print(f"  Deployment files: {deployment_dir}")
        print(f"  Credentials: {deployment_dir / 'secrets.txt'}")
        print("  Mounted pipeline inputs:")
        for selection in pipeline_selections:
            print(f"    {selection.source_path} -> {selection.mount_dir}")
        print("  Mounted sources:")
        for idx, source_dir in enumerate(source_dirs, start=1):
            print(f"    {source_dir} -> /app/sources/source_{idx}")
        return 0
    except DeployError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nDeployment cancelled.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
