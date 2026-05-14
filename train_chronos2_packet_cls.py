#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Chronos-2 packet-level Facebook video stalling recognition.

Default run:
  conda run -n zwq python train_chronos2_packet_cls.py

Quick checks:
  conda run -n zwq python train_chronos2_packet_cls.py --dry-run-data
  conda run -n zwq python train_chronos2_packet_cls.py --epochs 1 --debug-max-samples-per-pair 128 --max-packets 128

Sweep:
  conda run -n zwq python train_chronos2_packet_cls.py --sweep
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import logging
import math
import os
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


# Manual split pairs. Empty means auto-scan TRAIN/VAL/TEST data folders and
# match every traffic CSV to one label CSV under LABEL_DIR.
TRAIN_PAIRS: List[Tuple[str, str]] = []
VAL_PAIRS: List[Tuple[str, str]] = []
TEST_PAIRS: List[Tuple[str, str]] = []
# Example manual configuration:
# TRAIN_PAIRS = [
#     ("./train_csv_data/N3_Facebook_129_0909_100526.csv", "./labels/N3_Facebook_129_20250909_100526.csv"),
# ]


@dataclass
class CFG:
    TRAIN_DATA_DIR: str = "./train_csv_data"
    VAL_DATA_DIR: str = "./val_csv_data"
    TEST_DATA_DIR: str = "./test_csv_data"
    LABEL_DIR: str = "./labels"
    MODEL_PATH: str = "./chronos_2_weights"

    INPUT_WINDOW: float = 30.0
    STRIDE: float = 10.0
    MAX_PACKETS: int = 512
    MIN_PACKETS: int = 5
    PACKET_SELECT_MODE: str = "recent"  # "recent" or "uniform"

    BATCH_SIZE: int = 4
    EPOCHS: int = 20
    TRAIN_MODE: str = "head"  # "head", "partial", "full"
    UNFREEZE_LAST_N: int = 2
    LR_HEAD: float = 1e-3
    LR_BACKBONE: float = 1e-5
    WEIGHT_DECAY: float = 1e-4
    DROPOUT: float = 0.2
    GRAD_CLIP_NORM: float = 1.0
    MIXED_PRECISION: bool = True
    NUM_WORKERS: int = 0

    USE_VAL_THRESHOLD_SEARCH: bool = True
    FIXED_THRESHOLD: float = 0.5
    THRESHOLDS: Tuple[float, ...] = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2).tolist())

    SAVE_PATH: str = "./best_chronos2_packet_cls.pt"
    TEST_RESULT_SAVE_PATH: str = "./test_window_predictions.csv"
    SWEEP_RESULT_SAVE_PATH: str = "./sweep_results.csv"

    TIMEZONE: str = "Asia/Shanghai"
    SEED: int = 42
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    DEBUG_MAX_SAMPLES_PER_PAIR: Optional[int] = None
    DEBUG_MAX_TRAFFIC_ROWS: Optional[int] = None
    FORCE_TINY_ENCODER: bool = False  # debug only; real runs should be False

    FEATURE_NAMES: Tuple[str, ...] = (
        "log_packet_length",
        "inter_arrival_time",
        "relative_time_to_window_start",
        "relative_time_to_window_end",
        "direction",
        "is_tcp",
        "is_udp",
    )

    TRAFFIC_TIME_COLS: Tuple[str, ...] = ("frame.time_epoch", "timestamp", "time")
    TRAFFIC_LEN_COLS: Tuple[str, ...] = ("frame.len", "length", "_ws.col.Length")
    PROTO_COLS: Tuple[str, ...] = ("_ws.col.Protocol", "protocol", "proto", "ip.proto")
    TCP_SRC_COLS: Tuple[str, ...] = ("tcp.srcport",)
    TCP_DST_COLS: Tuple[str, ...] = ("tcp.dstport",)
    UDP_SRC_COLS: Tuple[str, ...] = ("udp.srcport",)
    UDP_DST_COLS: Tuple[str, ...] = ("udp.dstport",)

    LABEL_TIME_COL: str = "时间"
    LABEL_VALUE_COL: str = "预测结果(流畅/卡顿;0/1)"


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv_auto(path: str, nrows: Optional[int] = None) -> pd.DataFrame:
    last_exc: Optional[Exception] = None
    for enc in ("utf-8", "utf-8-sig", "gbk", "gb18030"):
        try:
            return pd.read_csv(path, encoding=enc, nrows=nrows, low_memory=False)
        except UnicodeDecodeError as exc:
            last_exc = exc
    try:
        return pd.read_csv(path, nrows=nrows, low_memory=False)
    except Exception as exc:
        raise RuntimeError(f"Failed to read {path}; last decode error={last_exc}") from exc


def norm_col(s: str) -> str:
    return re.sub(r"[\s_().:/;\\-]+", "", str(s).lower())


def find_traffic_col(columns: Sequence[str], candidates: Sequence[str], role: str, required: bool = True) -> Optional[str]:
    col_list = list(map(str, columns))
    for cand in candidates:
        if cand in col_list:
            return cand
    norm_map = {norm_col(c): c for c in col_list}
    for cand in candidates:
        if norm_col(cand) in norm_map:
            return norm_map[norm_col(cand)]
    if required:
        logging.error("Cannot find traffic column for %s.", role)
        logging.error("Current CSV columns: %s", col_list)
        logging.error("Modify CFG candidate list for %s if your tshark column name differs.", role)
        raise KeyError(f"Missing traffic column: {role}")
    return None


def require_label_cols(df: pd.DataFrame, path: str, cfg: CFG) -> None:
    missing = [c for c in (cfg.LABEL_TIME_COL, cfg.LABEL_VALUE_COL) if c not in df.columns]
    if missing:
        logging.error("Label file %s is missing required columns: %s", path, missing)
        logging.error("Current label columns: %s", list(df.columns))
        logging.error("Please check label column names. Required exactly: %r and %r", cfg.LABEL_TIME_COL, cfg.LABEL_VALUE_COL)
        raise KeyError(f"Missing label columns: {missing}")


def parse_time_seconds(values: pd.Series, timezone: str) -> np.ndarray:
    if pd.api.types.is_numeric_dtype(values):
        arr = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            raise ValueError("Numeric time column contains no finite values.")
        med = float(np.nanmedian(finite))
        if med > 1e17:
            arr /= 1e9
        elif med > 1e14:
            arr /= 1e6
        elif med > 1e11:
            arr /= 1e3
        return arr

    dt = pd.to_datetime(values, errors="coerce")
    if getattr(dt.dt, "tz", None) is None:
        dt = dt.dt.tz_localize(ZoneInfo(timezone), ambiguous="NaT", nonexistent="NaT")
    else:
        dt = dt.dt.tz_convert("UTC")
    raw = dt.dt.tz_convert("UTC").astype("int64").to_numpy(dtype=np.float64)
    finite = raw[np.isfinite(raw) & (raw > 0)]
    if len(finite) == 0:
        return np.full(len(values), np.nan, dtype=np.float64)
    med = float(np.nanmedian(finite))
    if med > 1e17:
        return raw / 1e9
    if med > 1e14:
        return raw / 1e6
    if med > 1e11:
        return raw / 1e3
    return raw


def to_local_str(sec: float, tz: str) -> str:
    return str(pd.to_datetime(sec, unit="s", utc=True).tz_convert(tz))


def parse_labels(values: pd.Series) -> np.ndarray:
    y = pd.to_numeric(values, errors="coerce")
    if y.isna().any():
        bad = values[y.isna()].head(10).tolist()
        raise ValueError(f"Label column must contain 0/1. Bad examples: {bad}")
    return (y.to_numpy(dtype=np.float32) > 0).astype(np.int64)


def numeric_port(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=np.float64)


def standardize_traffic(path: str, cfg: CFG) -> Tuple[pd.DataFrame, Dict[str, int]]:
    df = read_csv_auto(path, nrows=cfg.DEBUG_MAX_TRAFFIC_ROWS)
    columns = df.columns
    time_col = find_traffic_col(columns, cfg.TRAFFIC_TIME_COLS, "time")
    len_col = find_traffic_col(columns, cfg.TRAFFIC_LEN_COLS, "packet length")
    proto_col = find_traffic_col(columns, cfg.PROTO_COLS, "protocol", required=False)
    tcp_src_col = find_traffic_col(columns, cfg.TCP_SRC_COLS, "tcp.srcport")
    tcp_dst_col = find_traffic_col(columns, cfg.TCP_DST_COLS, "tcp.dstport")
    udp_src_col = find_traffic_col(columns, cfg.UDP_SRC_COLS, "udp.srcport")
    udp_dst_col = find_traffic_col(columns, cfg.UDP_DST_COLS, "udp.dstport")

    out = pd.DataFrame()
    out["time"] = parse_time_seconds(df[time_col], cfg.TIMEZONE)
    out["length"] = pd.to_numeric(df[len_col], errors="coerce").fillna(0).clip(lower=0).to_numpy(dtype=np.float32)
    out["proto"] = df[proto_col].astype(str) if proto_col else ""
    out["tcp_srcport"] = numeric_port(df[tcp_src_col])
    out["tcp_dstport"] = numeric_port(df[tcp_dst_col])
    out["udp_srcport"] = numeric_port(df[udp_src_col])
    out["udp_dstport"] = numeric_port(df[udp_dst_col])

    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    diag = {
        "raw_packets": int(len(df)),
        "tcp_src_nonnull": int(pd.notna(df[tcp_src_col]).sum()),
        "tcp_dst_nonnull": int(pd.notna(df[tcp_dst_col]).sum()),
        "udp_src_nonnull": int(pd.notna(df[udp_src_col]).sum()),
        "udp_dst_nonnull": int(pd.notna(df[udp_dst_col]).sum()),
    }
    return out, diag


def standardize_labels(path: str, cfg: CFG) -> pd.DataFrame:
    df = read_csv_auto(path)
    require_label_cols(df, path, cfg)
    out = pd.DataFrame()
    out["time"] = parse_time_seconds(df[cfg.LABEL_TIME_COL], cfg.TIMEZONE)
    out["label"] = parse_labels(df[cfg.LABEL_VALUE_COL])
    out = out.replace([np.inf, -np.inf], np.nan).dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    return out


def filter_port443_packets(traffic: pd.DataFrame, diag: Dict[str, int], traffic_path: str) -> pd.DataFrame:
    tcp_src_443 = traffic["tcp_srcport"] == 443
    tcp_dst_443 = traffic["tcp_dstport"] == 443
    udp_src_443 = traffic["udp_srcport"] == 443
    udp_dst_443 = traffic["udp_dstport"] == 443
    src443 = tcp_src_443 | udp_src_443
    dst443 = tcp_dst_443 | udp_dst_443
    keep = src443 ^ dst443
    out = traffic.loc[keep].copy()
    out["direction"] = np.where(dst443.loc[keep].to_numpy(), 1.0, -1.0).astype(np.float32)
    out["is_tcp"] = ((tcp_src_443 | tcp_dst_443).loc[keep]).to_numpy(dtype=np.float32)
    out["is_udp"] = ((udp_src_443 | udp_dst_443).loc[keep]).to_numpy(dtype=np.float32)
    logging.info("[%s] port 443 filter: before=%d after=%d", Path(traffic_path).name, len(traffic), len(out))
    if out.empty:
        logging.error("Port 443 filtering produced no packets for %s", traffic_path)
        logging.error("Original packet rows: %d", diag["raw_packets"])
        logging.error("Non-empty port counts: tcp.srcport=%d tcp.dstport=%d udp.srcport=%d udp.dstport=%d",
                      diag["tcp_src_nonnull"], diag["tcp_dst_nonnull"], diag["udp_src_nonnull"], diag["udp_dst_nonnull"])
        logging.error("Possible causes: wrong port columns, non-numeric port format, or this file has no 443 packets.")
        raise RuntimeError("No packets after port 443 filtering.")
    return out.sort_values("time").reset_index(drop=True)


def build_packet_feature_arrays(traffic443: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    t = traffic443["time"].to_numpy(dtype=np.float64)
    length = traffic443["length"].to_numpy(dtype=np.float32)
    log_len = np.log1p(length).astype(np.float32)
    iat = np.diff(t, prepend=t[0]).clip(min=0, max=60).astype(np.float32)
    feat = np.stack(
        [
            log_len,
            iat,
            np.zeros_like(log_len),
            np.zeros_like(log_len),
            traffic443["direction"].to_numpy(dtype=np.float32),
            traffic443["is_tcp"].to_numpy(dtype=np.float32),
            traffic443["is_udp"].to_numpy(dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    return t, feat


def window_labels(window_start: np.ndarray, window_end: np.ndarray, label_t: np.ndarray, label_y: np.ndarray) -> np.ndarray:
    prefix = np.concatenate([[0], np.cumsum(label_y.astype(np.int64))])
    left = np.searchsorted(label_t, window_start, side="left")
    right = np.searchsorted(label_t, window_end, side="right")
    return ((prefix[right] - prefix[left]) > 0).astype(np.int64)


def build_samples_for_pair(traffic_path: str, label_path: str, pair_id: int, cfg: CFG) -> Dict[str, Any]:
    traffic, diag = standardize_traffic(traffic_path, cfg)
    labels = standardize_labels(label_path, cfg)
    if traffic.empty or labels.empty:
        raise RuntimeError(f"Empty traffic or label file: {traffic_path}, {label_path}")

    logging.info("[%s] traffic time range: %s -> %s", Path(traffic_path).name,
                 to_local_str(float(traffic["time"].iloc[0]), cfg.TIMEZONE),
                 to_local_str(float(traffic["time"].iloc[-1]), cfg.TIMEZONE))
    logging.info("[%s] label time range:   %s -> %s", Path(label_path).name,
                 to_local_str(float(labels["time"].iloc[0]), cfg.TIMEZONE),
                 to_local_str(float(labels["time"].iloc[-1]), cfg.TIMEZONE))

    traffic443 = filter_port443_packets(traffic, diag, traffic_path)
    pkt_t, base_feat = build_packet_feature_arrays(traffic443)
    label_t = labels["time"].to_numpy(dtype=np.float64)
    label_y = labels["label"].to_numpy(dtype=np.int64)

    valid_start = max(float(label_t[0]), float(pkt_t[0]))
    valid_end = min(float(label_t[-1]), float(pkt_t[-1]))
    if valid_end - valid_start < cfg.INPUT_WINDOW:
        logging.error("Traffic and label ranges do not overlap enough for %s / %s", traffic_path, label_path)
        logging.error("traffic443 range: %s -> %s", to_local_str(float(pkt_t[0]), cfg.TIMEZONE), to_local_str(float(pkt_t[-1]), cfg.TIMEZONE))
        logging.error("label range:      %s -> %s", to_local_str(float(label_t[0]), cfg.TIMEZONE), to_local_str(float(label_t[-1]), cfg.TIMEZONE))
        logging.error("INPUT_WINDOW=%s STRIDE=%s port443_packets=%d", cfg.INPUT_WINDOW, cfg.STRIDE, len(pkt_t))
        raise RuntimeError("Insufficient traffic-label overlap.")

    starts = np.arange(valid_start, valid_end - cfg.INPUT_WINDOW + 1e-6, cfg.STRIDE, dtype=np.float64)
    ends = starts + cfg.INPUT_WINDOW
    y = window_labels(starts, ends, label_t, label_y)

    X_list: List[np.ndarray] = []
    M_list: List[np.ndarray] = []
    keep_starts: List[float] = []
    keep_ends: List[float] = []
    keep_y: List[int] = []

    for i in tqdm(range(len(starts)), desc=f"windows {Path(traffic_path).name}", leave=False):
        ws = float(starts[i])
        we = float(ends[i])
        left = np.searchsorted(pkt_t, ws, side="left")
        right = np.searchsorted(pkt_t, we, side="right")
        n = right - left
        if n < cfg.MIN_PACKETS:
            continue
        if n > cfg.MAX_PACKETS:
            if cfg.PACKET_SELECT_MODE == "uniform":
                idx = np.linspace(left, right - 1, cfg.MAX_PACKETS).round().astype(np.int64)
            else:
                idx = np.arange(right - cfg.MAX_PACKETS, right, dtype=np.int64)
        else:
            idx = np.arange(left, right, dtype=np.int64)
        n2 = len(idx)
        feat = base_feat[idx].copy()
        feat[:, 2] = (pkt_t[idx] - ws).astype(np.float32)
        feat[:, 3] = (we - pkt_t[idx]).astype(np.float32)
        sample = np.zeros((cfg.MAX_PACKETS, len(cfg.FEATURE_NAMES)), dtype=np.float32)
        mask = np.zeros(cfg.MAX_PACKETS, dtype=np.float32)
        sample[cfg.MAX_PACKETS - n2:] = feat
        mask[cfg.MAX_PACKETS - n2:] = 1.0
        X_list.append(sample)
        M_list.append(mask)
        keep_starts.append(ws)
        keep_ends.append(we)
        keep_y.append(int(y[i]))
        if cfg.DEBUG_MAX_SAMPLES_PER_PAIR and len(X_list) >= cfg.DEBUG_MAX_SAMPLES_PER_PAIR:
            break

    if not X_list:
        logging.error("Constructed 0 samples.")
        logging.error("traffic range: %s -> %s", to_local_str(float(traffic["time"].iloc[0]), cfg.TIMEZONE), to_local_str(float(traffic["time"].iloc[-1]), cfg.TIMEZONE))
        logging.error("label range:   %s -> %s", to_local_str(float(label_t[0]), cfg.TIMEZONE), to_local_str(float(label_t[-1]), cfg.TIMEZONE))
        logging.error("INPUT_WINDOW=%s STRIDE=%s port443_packets=%d", cfg.INPUT_WINDOW, cfg.STRIDE, len(pkt_t))
        logging.error("Possible causes: time unit error, timezone error, no traffic-label overlap, too strict port 443 filtering, or MIN_PACKETS too high.")
        raise RuntimeError("No samples constructed.")

    out = {
        "X": np.stack(X_list).astype(np.float32),
        "mask": np.stack(M_list).astype(np.float32),
        "y": np.asarray(keep_y, dtype=np.int64),
        "window_start": np.asarray(keep_starts, dtype=np.float64),
        "window_end": np.asarray(keep_ends, dtype=np.float64),
        "pair_id": np.full(len(keep_y), pair_id, dtype=np.int64),
        "traffic_file": np.asarray([Path(traffic_path).name] * len(keep_y), dtype=object),
        "label_file": np.asarray([Path(label_path).name] * len(keep_y), dtype=object),
    }
    logging.info("[%s] samples=%d positive_ratio=%.4f", Path(traffic_path).name, len(out["y"]), float(out["y"].mean()))
    return out


def split_pair(data: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    n = len(data["y"])
    a, b = int(n * 0.6), int(n * 0.8)
    slices = {"train": slice(0, a), "val": slice(a, b), "test": slice(b, n)}
    return {name: {k: v[sl] for k, v in data.items()} for name, sl in slices.items()}


def concat_splits(parts: List[Dict[str, Dict[str, Any]]]) -> Dict[str, Dict[str, Any]]:
    keys = ["X", "mask", "y", "window_start", "window_end", "pair_id", "traffic_file", "label_file"]
    merged: Dict[str, Dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        merged[split] = {}
        for k in keys:
            arrays = [p[split][k] for p in parts if len(p[split]["y"]) > 0]
            if not arrays:
                raise RuntimeError(f"No {split} samples after split.")
            merged[split][k] = np.concatenate(arrays, axis=0)
    return merged


def fit_transform_scaler(splits: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], StandardScaler]:
    scaler = StandardScaler()
    train_X = splits["train"]["X"]
    train_mask = splits["train"]["mask"].astype(bool)
    scaler.fit(train_X[train_mask])
    for split in ("train", "val", "test"):
        X = splits[split]["X"].copy()
        m = splits[split]["mask"].astype(bool)
        X[m] = scaler.transform(X[m]).astype(np.float32)
        X[~m] = 0.0
        splits[split]["X"] = X.astype(np.float32)
    return splits, scaler


def expected_label_name(traffic_name: str) -> Optional[str]:
    pat = re.compile(r"^(?P<prefix>.+)_(?P<mmdd>\d{4})_(?P<hms>\d{6})\.csv$")
    m = pat.match(traffic_name)
    if not m:
        return None
    return f"{m.group('prefix')}_2025{m.group('mmdd')}_{m.group('hms')}.csv"


def scan_pairs_for_split(
    split_name: str,
    data_dir: str,
    label_dir: str,
    manual_pairs: Sequence[Tuple[str, str]],
) -> List[Tuple[str, str]]:
    if manual_pairs:
        logging.info("Using manually configured %s pairs=%d", split_name, len(manual_pairs))
        return list(manual_pairs)

    traffic_root = Path(data_dir)
    label_root = Path(label_dir)
    if not traffic_root.exists():
        raise FileNotFoundError(
            f"{split_name} data directory does not exist: {data_dir}. "
            "Set CFG.TRAIN_DATA_DIR / CFG.VAL_DATA_DIR / CFG.TEST_DATA_DIR or pass CLI args."
        )
    if not label_root.exists():
        raise FileNotFoundError(f"Label directory does not exist: {label_dir}")

    traffic_files = sorted(traffic_root.glob("*.csv"))
    label_files = sorted(label_root.glob("*.csv"))
    logging.info("[%s] scanned traffic files: %d from %s", split_name, len(traffic_files), data_dir)
    logging.info("[%s] scanned label files: %d from %s", split_name, len(label_files), label_dir)
    label_map = {p.name: p for p in label_files}
    pairs: List[Tuple[str, str]] = []
    for tf in traffic_files:
        expected = expected_label_name(tf.name)
        if expected is None:
            logging.warning("Skip unmatched traffic filename format: %s", tf.name)
            continue
        lf = label_map.get(expected)
        if lf is None:
            logging.warning("No label match for %s; expected %s", tf.name, expected)
            continue
        pairs.append((str(tf), str(lf)))
    logging.info("[%s] matched traffic-label pairs: %d", split_name, len(pairs))
    for i, (t, l) in enumerate(pairs):
        logging.info("[%s] pair[%d]: %s <-> %s", split_name, i, t, l)
    if not pairs:
        raise RuntimeError(
            f"No {split_name} traffic-label pairs matched. "
            "Check folder paths, filenames, or fill TRAIN_PAIRS/VAL_PAIRS/TEST_PAIRS manually."
        )
    return pairs


def scan_split_pairs(cfg: CFG) -> Dict[str, List[Tuple[str, str]]]:
    return {
        "train": scan_pairs_for_split("train", cfg.TRAIN_DATA_DIR, cfg.LABEL_DIR, TRAIN_PAIRS),
        "val": scan_pairs_for_split("val", cfg.VAL_DATA_DIR, cfg.LABEL_DIR, VAL_PAIRS),
        "test": scan_pairs_for_split("test", cfg.TEST_DATA_DIR, cfg.LABEL_DIR, TEST_PAIRS),
    }


def log_split_stats(splits: Dict[str, Dict[str, Any]]) -> None:
    for split in ("train", "val", "test"):
        y = splits[split]["y"]
        logging.info("%s samples=%d positive_ratio=%.4f", split, len(y), float(y.mean()) if len(y) else 0.0)


class PacketDataset(Dataset):
    def __init__(self, split: Dict[str, Any]) -> None:
        self.x = torch.from_numpy(split["X"].astype(np.float32))
        self.mask = torch.from_numpy(split["mask"].astype(np.float32))
        self.y = torch.from_numpy(split["y"].astype(np.float32))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x": self.x[idx], "mask": self.mask[idx], "y": self.y[idx]}


class TinyEncoder(nn.Module):
    def __init__(self, feature_dim: int, max_len: int, hidden: int = 128) -> None:
        super().__init__()
        self.input = nn.Linear(feature_dim, hidden)
        self.pos = nn.Parameter(torch.zeros(1, max_len, hidden))
        layer = nn.TransformerEncoderLayer(hidden, 4, hidden * 4, batch_first=True, activation="gelu", norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, 2)
        self.hidden_size = hidden

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.input(x) + self.pos[:, : x.size(1)]
        return self.encoder(h, src_key_padding_mask=(mask <= 0))


class Chronos2Encoder(nn.Module):
    def __init__(self, cfg: CFG, feature_dim: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.backend = "chronos2"
        self.model: nn.Module
        self.hidden_size = 0

        if cfg.FORCE_TINY_ENCODER:
            self.model = TinyEncoder(feature_dim, cfg.MAX_PACKETS)
            self.hidden_size = self.model.hidden_size
            self.backend = "tiny_debug_encoder"
            logging.warning("FORCE_TINY_ENCODER=True. This is for debugging only, not a Chronos-2 experiment.")
            return

        if not Path(cfg.MODEL_PATH).exists():
            raise FileNotFoundError(
                f"MODEL_PATH does not exist: {cfg.MODEL_PATH}. "
                "Check that chronos_2_weights contains config.json and model.safetensors."
            )
        try:
            from chronos import Chronos2Pipeline

            pipe = Chronos2Pipeline.from_pretrained(cfg.MODEL_PATH, device_map=cfg.DEVICE)
            if not hasattr(pipe, "model"):
                raise RuntimeError("Chronos2Pipeline has no .model attribute.")
            self.model = pipe.model
            model_cfg = getattr(self.model, "config", None)
            chronos_cfg = getattr(self.model, "chronos_config", None)
            self.hidden_size = int(
                getattr(model_cfg, "d_model", None)
                or getattr(model_cfg, "hidden_size", None)
                or getattr(chronos_cfg, "d_model", None)
                or 0
            )
            if self.hidden_size <= 0:
                raise RuntimeError("Cannot infer Chronos-2 hidden size from config.")
            logging.info("Loaded Chronos-2 from local MODEL_PATH=%s hidden_size=%d", cfg.MODEL_PATH, self.hidden_size)
        except Exception as exc:
            logging.exception("Failed to load Chronos-2 from local MODEL_PATH=%s", cfg.MODEL_PATH)
            logging.error("Check MODEL_PATH contains valid Chronos-2 files such as config.json and model.safetensors.")
            raise RuntimeError("Chronos-2 local loading failed.") from exc

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.backend == "tiny_debug_encoder":
            return self.model(x, mask)
        if not hasattr(self.model, "encode"):
            raise RuntimeError("Loaded Chronos-2 model has no encode method; inspect chronos-forecasting API.")
        bsz, seq_len, feat_dim = x.shape
        context = x.transpose(1, 2).reshape(bsz * feat_dim, seq_len).to(dtype=torch.float32)
        context_mask = mask.unsqueeze(1).expand(bsz, feat_dim, seq_len).reshape(bsz * feat_dim, seq_len).to(dtype=torch.float32)
        group_ids = torch.arange(bsz, device=x.device, dtype=torch.long).repeat_interleave(feat_dim)
        enc_out, *_ = self.model.encode(
            context=context,
            context_mask=context_mask,
            group_ids=group_ids,
            num_output_patches=1,
        )
        if torch.is_tensor(enc_out):
            h = enc_out
        elif hasattr(enc_out, "last_hidden_state") and enc_out.last_hidden_state is not None:
            h = enc_out.last_hidden_state
        elif isinstance(enc_out, (tuple, list)) and len(enc_out) > 0:
            h = enc_out[0]
        else:
            raise RuntimeError(f"Cannot extract Chronos encoder hidden states from {type(enc_out)}")
        h = h.reshape(bsz, feat_dim, h.shape[-2], h.shape[-1])
        return h.mean(dim=(1, 2))


class Chronos2PacketClassifier(nn.Module):
    def __init__(self, cfg: CFG, feature_dim: int) -> None:
        super().__init__()
        self.encoder = Chronos2Encoder(cfg, feature_dim)
        h = self.encoder.hidden_size
        self.head = nn.Sequential(
            nn.LayerNorm(h),
            nn.Linear(h, h),
            nn.GELU(),
            nn.Dropout(cfg.DROPOUT),
            nn.Linear(h, 1),
        )
        self.apply_train_mode(cfg.TRAIN_MODE, cfg.UNFREEZE_LAST_N)

    def apply_train_mode(self, mode: str, unfreeze_last_n: int) -> None:
        if mode not in {"head", "partial", "full"}:
            raise ValueError("TRAIN_MODE must be head, partial, or full.")
        for p in self.encoder.parameters():
            p.requires_grad = mode == "full"
        for p in self.head.parameters():
            p.requires_grad = True
        if mode == "partial":
            modules = list(self.encoder.model.children())
            for m in modules[-max(1, unfreeze_last_n):]:
                for p in m.parameters():
                    p.requires_grad = True
        if mode == "full":
            logging.warning("TRAIN_MODE=full may overfit when data is small.")

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x, mask)
        if h.dim() == 3:
            m = mask.unsqueeze(-1)
            h = (h * m).sum(1) / m.sum(1).clamp_min(1.0)
        return self.head(h).squeeze(-1)


def make_loaders(splits: Dict[str, Dict[str, Any]], cfg: CFG) -> Dict[str, DataLoader]:
    return {
        split: DataLoader(
            PacketDataset(splits[split]),
            batch_size=cfg.BATCH_SIZE,
            shuffle=(split == "train"),
            num_workers=cfg.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        for split in ("train", "val", "test")
    }


def metrics_from_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> Dict[str, Any]:
    pred = (probs >= threshold).astype(np.int64)
    p, r, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
    auc: Optional[float]
    try:
        auc = float(roc_auc_score(y_true, probs)) if len(np.unique(y_true)) > 1 else None
    except ValueError:
        auc = None
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(p),
        "recall": float(r),
        "f1": float(f1),
        "auc": auc,
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1]).tolist(),
    }


def threshold_search(y_true: np.ndarray, probs: np.ndarray, cfg: CFG) -> Tuple[float, Dict[str, Any]]:
    if not cfg.USE_VAL_THRESHOLD_SEARCH:
        t = float(cfg.FIXED_THRESHOLD)
        return t, metrics_from_probs(y_true, probs, t)
    best_t = float(cfg.THRESHOLDS[0])
    best_m = metrics_from_probs(y_true, probs, best_t)
    for t in cfg.THRESHOLDS:
        m = metrics_from_probs(y_true, probs, float(t))
        if m["f1"] > best_m["f1"]:
            best_t, best_m = float(t), m
    return best_t, best_m


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, cfg: CFG) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, probs = [], []
    for batch in tqdm(loader, desc="predict", leave=False):
        x = batch["x"].to(cfg.DEVICE)
        mask = batch["mask"].to(cfg.DEVICE)
        logits = model(x, mask)
        ys.append(batch["y"].numpy())
        probs.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(ys).astype(np.int64), np.concatenate(probs).astype(np.float64)


def train_one_epoch(model: nn.Module, loader: DataLoader, opt: torch.optim.Optimizer, loss_fn: nn.Module, cfg: CFG, scaler: torch.amp.GradScaler) -> float:
    model.train()
    losses: List[float] = []
    use_amp = cfg.MIXED_PRECISION and cfg.DEVICE.startswith("cuda")
    for batch in tqdm(loader, desc="train", leave=False):
        x = batch["x"].to(cfg.DEVICE)
        mask = batch["mask"].to(cfg.DEVICE)
        y = batch["y"].to(cfg.DEVICE)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(x, mask)
            loss = loss_fn(logits, y)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        scaler.step(opt)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


def make_optimizer(model: Chronos2PacketClassifier, cfg: CFG) -> torch.optim.Optimizer:
    head, backbone = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (head if name.startswith("head") else backbone).append(p)
    groups = [{"params": head, "lr": cfg.LR_HEAD}]
    if backbone:
        groups.append({"params": backbone, "lr": cfg.LR_BACKBONE})
    return torch.optim.AdamW(groups, weight_decay=cfg.WEIGHT_DECAY)


def save_test_predictions(path: str, test_split: Dict[str, Any], y_true: np.ndarray, probs: np.ndarray, threshold: float) -> None:
    pred = (probs >= threshold).astype(np.int64)
    df = pd.DataFrame({
        "window_start": test_split["window_start"],
        "window_end": test_split["window_end"],
        "true_label": y_true,
        "pred_label": pred,
        "pred_prob": probs,
        "threshold": threshold,
        "pair_id": test_split["pair_id"],
        "traffic_file": test_split["traffic_file"],
        "label_file": test_split["label_file"],
    })
    df.to_csv(path, index=False, encoding="utf-8-sig")
    logging.info("Saved test window predictions to %s", path)


def save_checkpoint(path: str, model: nn.Module, cfg: CFG, scaler: StandardScaler, best_val_f1: float, best_threshold: float) -> None:
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(cfg),
        "best_val_f1": float(best_val_f1),
        "best_threshold": float(best_threshold),
        "feature_names": list(cfg.FEATURE_NAMES),
        "scaler": scaler,
    }, path)
    logging.info("Saved best model to %s", path)


def concat_split_samples(items: List[Dict[str, Any]], split_name: str) -> Dict[str, Any]:
    if not items:
        raise RuntimeError(f"No samples were built for split={split_name}")
    keys = ["X", "mask", "y", "window_start", "window_end", "pair_id", "traffic_file", "label_file"]
    return {k: np.concatenate([item[k] for item in items], axis=0) for k in keys}


def prepare_data(cfg: CFG, split_pairs: Dict[str, Sequence[Tuple[str, str]]]) -> Tuple[Dict[str, Dict[str, Any]], StandardScaler]:
    splits: Dict[str, Dict[str, Any]] = {}
    global_pair_id = 0
    for split in ("train", "val", "test"):
        built_items: List[Dict[str, Any]] = []
        for traffic_path, label_path in split_pairs[split]:
            logging.info("Loading split=%s pair_id=%d traffic=%s label=%s", split, global_pair_id, traffic_path, label_path)
            built_items.append(build_samples_for_pair(traffic_path, label_path, global_pair_id, cfg))
            global_pair_id += 1
        splits[split] = concat_split_samples(built_items, split)
    log_split_stats(splits)
    return fit_transform_scaler(splits)


def run_experiment(cfg: CFG, split_pairs: Dict[str, Sequence[Tuple[str, str]]], dry_run_data: bool = False) -> Dict[str, Any]:
    set_seed(cfg.SEED)
    logging.info("Device=%s", cfg.DEVICE)
    logging.info("USE_VAL_THRESHOLD_SEARCH=%s FIXED_THRESHOLD=%.3f", cfg.USE_VAL_THRESHOLD_SEARCH, cfg.FIXED_THRESHOLD)
    splits, scaler_obj = prepare_data(cfg, split_pairs)
    if dry_run_data:
        logging.info("Dry-run data completed.")
        return {"status": "dry_run_data"}

    loaders = make_loaders(splits, cfg)
    y_train = splits["train"]["y"]
    pos = int(y_train.sum())
    neg = int(len(y_train) - pos)
    pos_weight = float(neg / pos) if pos > 0 else 1.0
    if pos == 0:
        logging.warning("Train split has no positive samples; pos_weight forced to 1.0.")
    logging.info("BCE pos_weight=%.4f neg=%d pos=%d", pos_weight, neg, pos)

    model = Chronos2PacketClassifier(cfg, feature_dim=len(cfg.FEATURE_NAMES)).to(cfg.DEVICE)
    logging.info("Encoder backend=%s", model.encoder.backend)
    logging.info("Parameters trainable=%d total=%d",
                 sum(p.numel() for p in model.parameters() if p.requires_grad),
                 sum(p.numel() for p in model.parameters()))

    opt = make_optimizer(model, cfg)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=cfg.DEVICE))
    amp_scaler = torch.amp.GradScaler("cuda", enabled=cfg.MIXED_PRECISION and cfg.DEVICE.startswith("cuda"))

    best_val_f1 = -1.0
    best_threshold = float(cfg.FIXED_THRESHOLD)
    for epoch in range(1, cfg.EPOCHS + 1):
        try:
            train_loss = train_one_epoch(model, loaders["train"], opt, loss_fn, cfg, amp_scaler)
        except torch.cuda.OutOfMemoryError:
            logging.error("CUDA OOM. Try BATCH_SIZE=1/2, MAX_PACKETS=256, smaller local Chronos-2 weights if available, MIXED_PRECISION=True, TRAIN_MODE=head.")
            raise
        y_val, p_val = predict(model, loaders["val"], cfg)
        epoch_threshold, val_m = threshold_search(y_val, p_val, cfg)
        logging.info(
            "epoch=%03d train_loss=%.5f val_acc=%.4f val_p=%.4f val_r=%.4f val_f1=%.4f val_auc=%s threshold=%.2f val_cm=%s",
            epoch, train_loss, val_m["accuracy"], val_m["precision"], val_m["recall"], val_m["f1"],
            "nan" if val_m["auc"] is None else f"{val_m['auc']:.4f}", epoch_threshold, val_m["confusion_matrix"],
        )
        if val_m["f1"] > best_val_f1:
            best_val_f1 = float(val_m["f1"])
            best_threshold = float(epoch_threshold)
            save_checkpoint(cfg.SAVE_PATH, model, cfg, scaler_obj, best_val_f1, best_threshold)

    ckpt = torch.load(cfg.SAVE_PATH, map_location=cfg.DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    used_threshold = float(ckpt["best_threshold"] if cfg.USE_VAL_THRESHOLD_SEARCH else cfg.FIXED_THRESHOLD)
    logging.info("Current test threshold=%.4f", used_threshold)
    y_test, p_test = predict(model, loaders["test"], cfg)
    test_m = metrics_from_probs(y_test, p_test, used_threshold)
    test_05 = metrics_from_probs(y_test, p_test, 0.5)
    logging.info("Final test metrics: %s", json.dumps(test_m, ensure_ascii=False))
    logging.info("Test metrics with threshold=0.5 for comparison: %s", json.dumps(test_05, ensure_ascii=False))
    save_test_predictions(cfg.TEST_RESULT_SAVE_PATH, splits["test"], y_test, p_test, used_threshold)

    result = {
        "max_packets": cfg.MAX_PACKETS,
        "input_window": cfg.INPUT_WINDOW,
        "stride": cfg.STRIDE,
        "train_mode": cfg.TRAIN_MODE,
        "use_val_threshold_search": cfg.USE_VAL_THRESHOLD_SEARCH,
        "best_val_f1": best_val_f1,
        "used_threshold": used_threshold,
        "test": test_m,
        "test_threshold_0_5": test_05,
    }
    return result


def print_sweep_table(results: List[Dict[str, Any]], save_path: str) -> None:
    if not results:
        return
    rows = []
    for r in results:
        t, t05 = r["test"], r["test_threshold_0_5"]
        rows.append({
            "MAX_PACKETS": r["max_packets"],
            "INPUT_WINDOW": r["input_window"],
            "STRIDE": r["stride"],
            "TRAIN_MODE": r["train_mode"],
            "VAL_THR_SEARCH": r["use_val_threshold_search"],
            "best_val_f1": r["best_val_f1"],
            "used_thr": r["used_threshold"],
            "test_f1": t["f1"],
            "test_precision": t["precision"],
            "test_recall": t["recall"],
            "test_auc": t["auc"],
            "test_f1_thr0.5": t05["f1"],
            "test_precision_thr0.5": t05["precision"],
            "test_recall_thr0.5": t05["recall"],
        })
    df = pd.DataFrame(rows).sort_values(["test_f1", "best_val_f1"], ascending=False)
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print("\n===== Sweep Results (focus on F1) =====")
    print(df.to_string(index=False))
    logging.info("Saved sweep results to %s", save_path)


def run_sweep(base_cfg: CFG, split_pairs: Dict[str, Sequence[Tuple[str, str]]]) -> None:
    results: List[Dict[str, Any]] = []
    for max_packets in [256, 512, 1024]:
        for input_window in [10, 20, 30]:
            for stride in [5, 10]:
                for mode in ["head", "partial"]:
                    for use_thr in [True, False]:
                        cfg = copy.deepcopy(base_cfg)
                        cfg.MAX_PACKETS = max_packets
                        cfg.INPUT_WINDOW = float(input_window)
                        cfg.STRIDE = float(stride)
                        cfg.TRAIN_MODE = mode
                        cfg.USE_VAL_THRESHOLD_SEARCH = use_thr
                        suffix = f"mp{max_packets}_win{input_window}_st{stride}_{mode}_thr{int(use_thr)}"
                        cfg.SAVE_PATH = f"./best_chronos2_packet_cls_{suffix}.pt"
                        cfg.TEST_RESULT_SAVE_PATH = f"./test_window_predictions_{suffix}.csv"
                        logging.info("Sweep run: %s", suffix)
                        try:
                            results.append(run_experiment(cfg, split_pairs, dry_run_data=False))
                        except torch.cuda.OutOfMemoryError:
                            logging.error("Skipping sweep run due to OOM.")
                        except Exception as exc:
                            logging.exception("Skipping sweep run due to error: %r", exc)
                        finally:
                            gc.collect()
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        print_sweep_table(results, base_cfg.SWEEP_RESULT_SAVE_PATH)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run-data", action="store_true")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--epochs", type=int)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--max-packets", type=int)
    p.add_argument("--input-window", type=float)
    p.add_argument("--stride", type=float)
    p.add_argument("--train-mode", choices=["head", "partial", "full"])
    p.add_argument("--model-path", type=str)
    p.add_argument("--train-data-dir", type=str)
    p.add_argument("--val-data-dir", type=str)
    p.add_argument("--test-data-dir", type=str)
    p.add_argument("--label-dir", type=str)
    p.add_argument("--fixed-threshold", type=float)
    p.add_argument("--no-val-threshold-search", action="store_true")
    p.add_argument("--debug-max-samples-per-pair", type=int)
    p.add_argument("--debug-max-traffic-rows", type=int)
    p.add_argument("--force-tiny-encoder", action="store_true", help="Debug only; bypasses Chronos-2.")
    return p.parse_args()


def apply_args(cfg: CFG, args: argparse.Namespace) -> CFG:
    if args.epochs is not None:
        cfg.EPOCHS = args.epochs
    if args.batch_size is not None:
        cfg.BATCH_SIZE = args.batch_size
    if args.max_packets is not None:
        cfg.MAX_PACKETS = args.max_packets
    if args.input_window is not None:
        cfg.INPUT_WINDOW = args.input_window
    if args.stride is not None:
        cfg.STRIDE = args.stride
    if args.train_mode is not None:
        cfg.TRAIN_MODE = args.train_mode
    if args.model_path is not None:
        cfg.MODEL_PATH = args.model_path
    if args.train_data_dir is not None:
        cfg.TRAIN_DATA_DIR = args.train_data_dir
    if args.val_data_dir is not None:
        cfg.VAL_DATA_DIR = args.val_data_dir
    if args.test_data_dir is not None:
        cfg.TEST_DATA_DIR = args.test_data_dir
    if args.label_dir is not None:
        cfg.LABEL_DIR = args.label_dir
    if args.fixed_threshold is not None:
        cfg.FIXED_THRESHOLD = args.fixed_threshold
    if args.no_val_threshold_search:
        cfg.USE_VAL_THRESHOLD_SEARCH = False
    if args.debug_max_samples_per_pair is not None:
        cfg.DEBUG_MAX_SAMPLES_PER_PAIR = args.debug_max_samples_per_pair
    if args.debug_max_traffic_rows is not None:
        cfg.DEBUG_MAX_TRAFFIC_ROWS = args.debug_max_traffic_rows
    if args.force_tiny_encoder:
        cfg.FORCE_TINY_ENCODER = True
    return cfg


def main() -> None:
    setup_logging()
    args = parse_args()
    cfg = apply_args(CFG(), args)
    split_pairs = scan_split_pairs(cfg)
    if not Path(cfg.MODEL_PATH).exists() and not cfg.FORCE_TINY_ENCODER:
        raise FileNotFoundError(f"MODEL_PATH not found: {cfg.MODEL_PATH}")
    if args.sweep:
        run_sweep(cfg, split_pairs)
    else:
        run_experiment(cfg, split_pairs, dry_run_data=args.dry_run_data)


if __name__ == "__main__":
    main()
