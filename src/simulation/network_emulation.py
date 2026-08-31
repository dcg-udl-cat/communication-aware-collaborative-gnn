"""Calibrated single-host TCP replay for declared network profiles.

This is an application-level fallback for environments without Linux ``tc``.
It validates the timing model over a real socket but is not a physical-device
or production-network measurement.
"""

from __future__ import annotations

import socket
import statistics
import threading
import time
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class LinkProfile:
    name: str
    bandwidth_bps: float
    rtt_ms: float


LINK_PROFILES = {
    "constrained": LinkProfile("constrained", 1e6, 100.0),
    "moderate": LinkProfile("moderate", 10e6, 25.0),
    "lan": LinkProfile("lan", 100e6, 5.0),
}


@dataclass(frozen=True)
class ReplayResult:
    profile: str
    replay_bytes: int
    repeats: int
    median_s: float
    q1_s: float
    q3_s: float
    observed_rtt_median_s: float
    rtt_error_fraction: float
    rtt_error_q1_fraction: float
    rtt_error_q3_fraction: float
    rtt_within_tolerance: bool
    achieved_bandwidth_bps: float
    bandwidth_error_fraction: float
    bandwidth_error_q1_fraction: float
    bandwidth_error_q3_fraction: float
    calibrated: bool
    method: str = "application_shaped_tcp_socketpair"

    def as_dict(self) -> dict:
        return asdict(self)


def modeled_transfer_s(wire_bytes: int, profile: LinkProfile) -> float:
    if wire_bytes < 0 or profile.bandwidth_bps <= 0 or profile.rtt_ms < 0:
        raise ValueError("Invalid payload or link profile.")
    return wire_bytes * 8 / profile.bandwidth_bps + profile.rtt_ms / 1000.0


def replay_profile(
    profile: LinkProfile,
    *,
    wire_bytes: int,
    repeats: int = 3,
    minimum_transfer_s: float = 0.20,
) -> ReplayResult:
    """Verify bandwidth and record scheduler distortion of the imposed RTT."""

    if repeats < 1:
        raise ValueError("repeats must be positive.")
    replay_bytes = max(wire_bytes, int(profile.bandwidth_bps * minimum_transfer_s / 8))
    samples = [_replay_once(profile, replay_bytes) for _ in range(repeats)]
    elapsed = sorted(sample[0] for sample in samples)
    observed_rtts = sorted(sample[1] for sample in samples)
    median = statistics.median(elapsed)
    observed_rtt = statistics.median(observed_rtts)
    target_rtt = profile.rtt_ms / 1000.0
    achieved_bandwidths = sorted(
        replay_bytes * 8 / max(elapsed_s - observed_rtt_s, 1e-12)
        for elapsed_s, observed_rtt_s in samples
    )
    bandwidth_errors = sorted(
        abs(achieved - profile.bandwidth_bps) / profile.bandwidth_bps
        for achieved in achieved_bandwidths
    )
    rtt_errors = sorted(
        abs(observed_rtt_s - target_rtt) / target_rtt if target_rtt else 0.0
        for _, observed_rtt_s in samples
    )
    achieved = statistics.median(achieved_bandwidths)
    error = statistics.median(bandwidth_errors)
    rtt_error = statistics.median(rtt_errors)
    return ReplayResult(
        profile=profile.name,
        replay_bytes=replay_bytes,
        repeats=repeats,
        median_s=median,
        q1_s=_percentile(elapsed, 0.25),
        q3_s=_percentile(elapsed, 0.75),
        observed_rtt_median_s=observed_rtt,
        rtt_error_fraction=rtt_error,
        rtt_error_q1_fraction=_percentile(rtt_errors, 0.25),
        rtt_error_q3_fraction=_percentile(rtt_errors, 0.75),
        rtt_within_tolerance=rtt_error <= 0.10,
        achieved_bandwidth_bps=achieved,
        bandwidth_error_fraction=error,
        bandwidth_error_q1_fraction=_percentile(bandwidth_errors, 0.25),
        bandwidth_error_q3_fraction=_percentile(bandwidth_errors, 0.75),
        calibrated=error <= 0.10,
    )


def _replay_once(profile: LinkProfile, byte_count: int) -> tuple[float, float]:
    sender, receiver = socket.socketpair()

    def drain() -> None:
        remaining = byte_count
        while remaining:
            chunk = receiver.recv(min(65_536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)

    worker = threading.Thread(target=drain)
    worker.start()
    chunk = bytes(min(4096, byte_count))
    sent = 0
    start = time.perf_counter()
    half_rtt_s = profile.rtt_ms / 2000.0
    observed_rtt_s = 0.0
    if half_rtt_s:
        sleep_start = time.perf_counter()
        time.sleep(half_rtt_s)
        observed_rtt_s += time.perf_counter() - sleep_start
    while sent < byte_count:
        current = min(len(chunk), byte_count - sent)
        sender.sendall(chunk[:current])
        sent += current
        deadline = start + half_rtt_s + sent * 8 / profile.bandwidth_bps
        delay = deadline - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
    sender.shutdown(socket.SHUT_WR)
    worker.join()
    if half_rtt_s:
        sleep_start = time.perf_counter()
        time.sleep(half_rtt_s)
        observed_rtt_s += time.perf_counter() - sleep_start
    elapsed = time.perf_counter() - start
    sender.close()
    receiver.close()
    return elapsed, observed_rtt_s


def _percentile(sorted_values: list[float], fraction: float) -> float:
    position = fraction * (len(sorted_values) - 1)
    lower = int(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = position - lower
    return sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight
