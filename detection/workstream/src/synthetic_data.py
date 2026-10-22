"""Synthetic network traffic generator for development, testing, and smoke runs.

Generates realistic network flow records across multiple attack classes and benign
traffic with 15+ numeric and categorical features.

NOTE: This generator produces purely SYNTHETIC REFERENCE DATA.
It must never be labelled, presented, or evaluated as actual ACI IoT 2023 data.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd


CLASSES = [
    "Benign",
    "DDoS_UDP",
    "PortScan",
    "Mirai_Botnet",
    "Infiltration",
    "BruteForce",
]


def generate_synthetic_network_data(
    n_rows: int = 5000,
    seed: int = 42,
    target_col: str = "label",
) -> pd.DataFrame:
    """Generate deterministic synthetic network traffic data with >= 5 classes and >= 15 features."""
    rng = np.random.default_rng(seed)

    # Distribute classes with realistic class imbalance
    class_probs = [0.45, 0.15, 0.15, 0.10, 0.08, 0.07]
    labels = rng.choice(CLASSES, size=n_rows, p=class_probs)

    base_time = datetime(2026, 10, 20, 8, 0, 0, tzinfo=timezone.utc)
    timestamps = [
        (base_time + timedelta(seconds=int(rng.integers(0, 86400)))).isoformat()
        for _ in range(n_rows)
    ]
    flow_ids = [f"FLW-20261020-{i+1:06d}" for i in range(n_rows)]
    device_ids = [f"IOT-DEV-{rng.integers(1, 10):03d}" for _ in range(n_rows)]
    src_ips = [f"192.168.1.{rng.integers(10, 150)}" for _ in range(n_rows)]
    dst_ips = [f"10.0.0.{rng.integers(1, 25)}" for _ in range(n_rows)]

    # Feature generation per class
    src_ports = np.zeros(n_rows, dtype=np.int32)
    dst_ports = np.zeros(n_rows, dtype=np.int32)
    durations = np.zeros(n_rows, dtype=np.float32)
    src_bytes = np.zeros(n_rows, dtype=np.int64)
    dst_bytes = np.zeros(n_rows, dtype=np.int64)
    src_pkts = np.zeros(n_rows, dtype=np.int32)
    dst_pkts = np.zeros(n_rows, dtype=np.int32)
    packet_rate = np.zeros(n_rows, dtype=np.float32)
    byte_rate = np.zeros(n_rows, dtype=np.float32)
    tcp_flags_syn = np.zeros(n_rows, dtype=np.int32)
    tcp_flags_ack = np.zeros(n_rows, dtype=np.int32)
    tcp_flags_fin = np.zeros(n_rows, dtype=np.int32)
    tcp_flags_rst = np.zeros(n_rows, dtype=np.int32)
    flow_iat_mean = np.zeros(n_rows, dtype=np.float32)
    flow_iat_std = np.zeros(n_rows, dtype=np.float32)
    header_length = np.zeros(n_rows, dtype=np.int32)
    protos = []
    services = []

    for i, label in enumerate(labels):
        if label == "Benign":
            proto = rng.choice(["tcp", "udp"], p=[0.7, 0.3])
            service = rng.choice(["http", "dns", "ssh", "mqtt"], p=[0.4, 0.3, 0.1, 0.2])
            sp = int(rng.integers(1024, 65535))
            dp = int(rng.choice([80, 443, 53, 22, 1883]))
            dur = float(rng.uniform(0.1, 15.0))
            spk = int(rng.integers(2, 60))
            dpk = int(rng.integers(2, 80))
            sb = int(spk * rng.integers(40, 1400))
            db = int(dpk * rng.integers(40, 1400))
            syn = int(1 if proto == "tcp" else 0)
            ack = int(spk if proto == "tcp" else 0)
            fin = int(1 if proto == "tcp" and rng.random() > 0.3 else 0)
            rst = int(1 if rng.random() > 0.95 else 0)
            iat_m = float(dur / max(1, spk + dpk))
            iat_s = float(iat_m * rng.uniform(0.1, 0.5))

        elif label == "DDoS_UDP":
            proto = "udp"
            service = "other"
            sp = int(rng.integers(1024, 65535))
            dp = int(rng.choice([80, 443, 8080, 53]))
            dur = float(rng.uniform(0.01, 2.0))
            spk = int(rng.integers(500, 3000))
            dpk = int(rng.integers(0, 5))
            sb = int(spk * rng.integers(500, 1400))
            db = int(dpk * 64)
            syn = 0
            ack = 0
            fin = 0
            rst = 0
            iat_m = float(dur / max(1, spk))
            iat_s = float(iat_m * 0.1)

        elif label == "PortScan":
            proto = "tcp"
            service = "other"
            sp = int(rng.integers(40000, 65535))
            dp = int(rng.integers(1, 1024))
            dur = float(rng.uniform(0.001, 0.1))
            spk = int(rng.integers(1, 3))
            dpk = int(rng.integers(0, 2))
            sb = int(spk * 60)
            db = int(dpk * 40)
            syn = spk
            ack = 0
            fin = 0
            rst = int(1 if dpk > 0 else 0)
            iat_m = float(dur / max(1, spk))
            iat_s = 0.001

        elif label == "Mirai_Botnet":
            proto = rng.choice(["tcp", "udp"], p=[0.8, 0.2])
            service = rng.choice(["telnet", "http", "other"], p=[0.6, 0.3, 0.1])
            sp = int(rng.integers(1024, 65535))
            dp = int(rng.choice([23, 2323, 80, 5555]))
            dur = float(rng.uniform(0.5, 5.0))
            spk = int(rng.integers(20, 150))
            dpk = int(rng.integers(5, 30))
            sb = int(spk * rng.integers(80, 300))
            db = int(dpk * 64)
            syn = int(1 if proto == "tcp" else 0)
            ack = int(spk // 2 if proto == "tcp" else 0)
            fin = 0
            rst = int(1 if rng.random() > 0.5 else 0)
            iat_m = float(dur / max(1, spk + dpk))
            iat_s = float(iat_m * 0.3)

        elif label == "Infiltration":
            proto = "tcp"
            service = rng.choice(["ssh", "http"], p=[0.5, 0.5])
            sp = int(rng.integers(1024, 65535))
            dp = int(rng.choice([22, 443, 8443]))
            dur = float(rng.uniform(5.0, 45.0))
            spk = int(rng.integers(50, 400))
            dpk = int(rng.integers(80, 600))
            sb = int(spk * rng.integers(100, 800))
            db = int(dpk * rng.integers(200, 1200))
            syn = 1
            ack = spk
            fin = 1
            rst = 0
            iat_m = float(dur / max(1, spk + dpk))
            iat_s = float(iat_m * 0.8)

        else:  # BruteForce
            proto = "tcp"
            service = rng.choice(["ssh", "http"], p=[0.7, 0.3])
            sp = int(rng.integers(1024, 65535))
            dp = int(rng.choice([22, 80, 443]))
            dur = float(rng.uniform(0.5, 8.0))
            spk = int(rng.integers(10, 80))
            dpk = int(rng.integers(10, 80))
            sb = int(spk * rng.integers(60, 200))
            db = int(dpk * rng.integers(60, 200))
            syn = 1
            ack = spk
            fin = 1
            rst = int(1 if rng.random() > 0.4 else 0)
            iat_m = float(dur / max(1, spk + dpk))
            iat_s = float(iat_m * 0.4)

        src_ports[i] = sp
        dst_ports[i] = dp
        durations[i] = dur
        src_bytes[i] = sb
        dst_bytes[i] = db
        src_pkts[i] = spk
        dst_pkts[i] = dpk
        total_pkts = spk + dpk
        total_bytes = sb + db
        packet_rate[i] = float(total_pkts / max(dur, 1e-4))
        byte_rate[i] = float(total_bytes / max(dur, 1e-4))
        tcp_flags_syn[i] = syn
        tcp_flags_ack[i] = ack
        tcp_flags_fin[i] = fin
        tcp_flags_rst[i] = rst
        flow_iat_mean[i] = iat_m
        flow_iat_std[i] = iat_s
        header_length[i] = int(total_pkts * 32)
        protos.append(proto)
        services.append(service)

    data = {
        "flow_id": flow_ids,
        "timestamp": timestamps,
        "device_id": device_ids,
        "src_ip": src_ips,
        "dst_ip": dst_ips,
        "src_port": src_ports,
        "dst_port": dst_ports,
        "proto": protos,
        "service": services,
        "duration": durations,
        "src_bytes": src_bytes,
        "dst_bytes": dst_bytes,
        "src_pkts": src_pkts,
        "dst_pkts": dst_pkts,
        "packet_rate": packet_rate,
        "byte_rate": byte_rate,
        "tcp_flags_syn": tcp_flags_syn,
        "tcp_flags_ack": tcp_flags_ack,
        "tcp_flags_fin": tcp_flags_fin,
        "tcp_flags_rst": tcp_flags_rst,
        "flow_iat_mean": flow_iat_mean,
        "flow_iat_std": flow_iat_std,
        "header_length": header_length,
        target_col: labels,
    }

    df = pd.DataFrame(data)
    return df


def save_synthetic_data(
    output_path: str | Path,
    n_rows: int = 5000,
    seed: int = 42,
    target_col: str = "label",
) -> Path:
    """Generate and write synthetic reference CSV to disk."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate_synthetic_network_data(n_rows=n_rows, seed=seed, target_col=target_col)
    df.to_csv(out, index=False)
    print(f"[SYNTHETIC] Saved {len(df)} synthetic rows ({df[target_col].nunique()} classes, {len(df.columns)} columns) to {out}")
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic reference network traffic dataset.")
    parser.add_argument("--output", default="data/reference_synthetic.csv", help="Output CSV path")
    parser.add_argument("--n-rows", type=int, default=5000, help="Number of rows to generate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--target-col", default="label", help="Target column name")
    args = parser.parse_args()

    save_synthetic_data(args.output, args.n_rows, args.seed, args.target_col)
