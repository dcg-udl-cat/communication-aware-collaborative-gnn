"""Cost model: simulated bandwidth, latency, and energy for each system.

All costs are per inference over the full node set (one classification round).

Latency model
-------------
  latency = transmission_time + compute_time
  transmission_time = bytes_transmitted / bandwidth_bps
  compute_time      = flops / flops_per_second

We fix bandwidth_bps and flops_per_second as scenario parameters so results
are comparable across systems. Default values reflect a realistic edge scenario:
  bandwidth: 1 Mbit/s (constrained WiFi / LoRa-like link)
  edge_flops_per_s: 100 MFLOPs/s (MCU-class processor)
  cloud_flops_per_s: 10 GFLOPs/s (server GPU, normalized)

Bandwidth model
---------------
  bytes_transmitted = num_messages * payload_dim * bytes_per_float

Local system: 0 (no transmission)
Distributed:  one compressed embedding is sent for each remote graph message
Cloud:        each node sends raw features OR compressed embedding (configurable)

Energy proxy
------------
  energy ∝ flops + alpha * bytes_transmitted
  alpha = relative cost of communication vs computation (default 10:
  transmitting 1 byte costs ~10x more energy than 1 FLOP on edge hardware)
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class ScenarioConfig:
    """Hardware and network parameters for the cost simulation."""
    bandwidth_bps: float        = 1e6        # 1 Mbit/s edge link
    edge_flops_per_s: float     = 100e6      # 100 MFLOPs/s (edge node)
    cloud_flops_per_s: float    = 10e9       # 10 GFLOPs/s (cloud server)
    bytes_per_float: int        = 4          # float32
    energy_comm_alpha: float    = 10.0       # comm cost relative to compute


@dataclass
class CostBreakdown:
    """Per-system cost breakdown for one inference round.

    Encoder framing:
      When USE_PRETRAINED_ENCODER=True, the frozen encoder (ResNet18/DINO)
      is treated as OFFLINE PREPROCESSING: it runs once during node setup,
      not on every inference round. Its FLOPs are therefore NOT included in
      the per-round cost numbers by default.

      If you want to account for online encoding (e.g. live camera feeds
      where each frame must be encoded before transmission), set
      encoder_preprocessing_flops to the encoder's forward-pass cost.
      This adds it to the per-round latency and energy estimates.

      For the thesis: explicitly state which framing you use. The offline
      framing is appropriate when nodes pre-compute embeddings during
      a calibration phase; the online framing applies to streaming data.
    """
    system_name: str

    # Bandwidth
    bytes_per_node: float        = 0.0  # compatibility alias for payload bytes
    bytes_per_message: float     = 0.0
    num_messages: int            = 0
    total_bytes: float           = 0.0

    # Latency components (seconds)
    transmission_latency_s: float = 0.0
    compute_latency_s: float      = 0.0
    # Encoder cost: 0.0 if offline preprocessing, >0 if online (per-round)
    encoder_latency_s: float      = 0.0

    @property
    def total_latency_s(self) -> float:
        return self.transmission_latency_s + self.compute_latency_s + self.encoder_latency_s

    # FLOPs
    edge_flops: float            = 0.0
    cloud_flops: float           = 0.0
    encoder_preprocessing_flops: float = 0.0   # excluded from per-round by default

    @property
    def total_flops(self) -> float:
        """Per-round FLOPs. Encoder excluded unless encoder_latency_s > 0."""
        return self.edge_flops + self.cloud_flops

    @property
    def total_flops_including_encoder(self) -> float:
        """Full FLOPs including offline encoder preprocessing."""
        return self.edge_flops + self.cloud_flops + self.encoder_preprocessing_flops

    # Energy proxy
    energy_proxy: float          = 0.0

    def as_dict(self) -> dict:
        return {
            "system": self.system_name,
            "bytes_per_node": self.bytes_per_node,
            "bytes_per_message": self.bytes_per_message,
            "num_messages": self.num_messages,
            "total_bytes": self.total_bytes,
            "transmission_latency_s": self.transmission_latency_s,
            "compute_latency_s": self.compute_latency_s,
            "encoder_latency_s": self.encoder_latency_s,
            "total_latency_s": self.total_latency_s,
            "edge_flops": self.edge_flops,
            "cloud_flops": self.cloud_flops,
            "encoder_preprocessing_flops": self.encoder_preprocessing_flops,
            "total_flops": self.total_flops,
            "total_flops_including_encoder": self.total_flops_including_encoder,
            "energy_proxy": self.energy_proxy,
        }


def compute_costs(
    system_name: str,
    num_nodes: int,
    embedding_dim: int,
    model_flops: float,
    is_cloud: bool,
    transmit_full_features: bool = False,
    full_feature_dim: int = 128,
    num_messages: int | None = None,
    cfg: ScenarioConfig | None = None,
) -> CostBreakdown:
    """Compute all cost metrics for one system configuration.

    Args:
        system_name:          "local", "distributed", or "cloud"
        num_nodes:            Number of edge nodes
        embedding_dim:        Compressed embedding size sent per node (0 for local)
        model_flops:          FLOPs for a full-graph inference round
        is_cloud:             If True, inference runs on cloud hardware
        transmit_full_features: Cloud mode: send full features instead of embedding
        full_feature_dim:     Feature dim when transmitting full features
        num_messages:         Number of network payloads. Defaults to num_nodes
                              for backward compatibility and cloud uploads.
        cfg:                  Hardware scenario config (uses defaults if None)
    """
    cfg = cfg or ScenarioConfig()

    # Bytes transmitted
    transmitted_dim = full_feature_dim if transmit_full_features else embedding_dim
    payload_bytes = transmitted_dim * cfg.bytes_per_float
    effective_messages = num_nodes if num_messages is None else num_messages
    if effective_messages < 0:
        raise ValueError("num_messages must be non-negative.")
    total_bytes = effective_messages * payload_bytes

    # Latency
    transmission_latency = total_bytes * 8 / cfg.bandwidth_bps   # bytes → bits
    flops_per_s = cfg.cloud_flops_per_s if is_cloud else cfg.edge_flops_per_s
    compute_latency = model_flops / flops_per_s

    # FLOPs split
    edge_flops  = 0.0 if is_cloud else model_flops
    cloud_flops = model_flops if is_cloud else 0.0

    # Energy proxy: compute + alpha * comm
    energy = model_flops + cfg.energy_comm_alpha * total_bytes

    return CostBreakdown(
        system_name=system_name,
        bytes_per_node=payload_bytes,
        bytes_per_message=payload_bytes,
        num_messages=effective_messages,
        total_bytes=total_bytes,
        transmission_latency_s=transmission_latency,
        compute_latency_s=compute_latency,
        edge_flops=edge_flops,
        cloud_flops=cloud_flops,
        energy_proxy=energy,
    )
