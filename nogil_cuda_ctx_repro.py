from __future__ import annotations

import argparse
import threading
import time
from contextlib import nullcontext

import torch


def parse_args() -> argparse.Namespace:
    """Parse args."""
    parser = argparse.ArgumentParser(description="No-GIL CUDA context repro")
    parser.add_argument("--devices", nargs="+", required=True, help="CUDA devices to span")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--modules-per-device", type=int, default=2)
    parser.add_argument("--iters", type=int, default=500)
    parser.add_argument("--guard", action="store_true", help="use torch.cuda.device guard")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def prepare_tensors(devices: list[str], modules_per_device: int) -> list[torch.Tensor]:
    """Prepare tensors. Keep the nested traversal explicit so ordering and metadata stay aligned."""
    tensors = []
    for device in devices:
        for _ in range(modules_per_device):
            tensors.append(torch.randn(512, 512, dtype=torch.bfloat16, device=device))
    return tensors


def worker(
    worker_idx: int,
    tensors: list[torch.Tensor],
    devices: list[str],
    iters: int,
    guard: bool,
    log_every: int,
    crash_flag: threading.Event,
    print_lock: threading.Lock,
) -> None:
    """Implement worker for this module. Preserve the fallback order expected by the surrounding caller."""
    device = devices[worker_idx % len(devices)]
    other_device = devices[(devices.index(device) + 1) % len(devices)]
    null_ctx = nullcontext

    for iteration in range(iters):
        if crash_flag.is_set():
            return

        idx = (worker_idx * 31 + iteration) % len(tensors)
        tensor = tensors[idx]

        torch.cuda.set_device(other_device)
        ctx = torch.cuda.device(tensor.device) if guard else null_ctx()
        with ctx:
            if tensor is None:
                with print_lock:
                    print(f"worker {worker_idx} saw None tensor at idx {idx}")
                crash_flag.set()
                return

            if not guard and torch.cuda.current_device() != tensor.device.index:
                tensors[idx] = None
                with print_lock:
                    print(f"worker {worker_idx} cleared tensor idx={idx} due to ctx mismatch")
                crash_flag.set()
                return

            tmp = tensor + 1.0
            torch.cuda.synchronize(tensor.device)

        if iteration and iteration % log_every == 0:
            with print_lock:
                print(
                    f"worker {worker_idx} iteration {iteration} current_device={torch.cuda.current_device()} tensor_device={tensor.device}"
                )

        time.sleep(0.001)


def main() -> None:
    """Run the CLI entry point for this module."""
    args = parse_args()
    shared = prepare_tensors(args.devices, args.modules_per_device)
    crash_flag = threading.Event()
    lock = threading.Lock()
    threads = []

    print("guard mode", args.guard)
    for idx in range(args.threads):
        t = threading.Thread(
            target=worker,
            args=(idx, shared, args.devices, args.iters, args.guard, args.log_every, crash_flag, lock),
            name=f"nogil-{idx}",
        )
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    if crash_flag.is_set():
        raise SystemExit("crash simulated")
    print("completed without crash")


if __name__ == "__main__":
    main()
