from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
import torch.nn as nn
from scipy import sparse
from threadpoolctl import threadpool_limits


RANDOM_STATE = 42
SAMPLE_EVENTS = int(os.getenv("PRAXIS_LATENCY_EVENTS", "200"))
REPEATS_PER_EVENT = int(os.getenv("PRAXIS_LATENCY_REPEATS", "3"))
WARMUP_EVENTS = int(os.getenv("PRAXIS_LATENCY_WARMUP", "20"))
SUMMARY_BOOTSTRAPS = int(os.getenv("PRAXIS_LATENCY_BOOTSTRAPS", "2000"))
LOCAL_HOPS = 2

torch.set_num_threads(1)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass

WORKSPACE = Path(__file__).resolve().parents[1]
PROJECT_ROOT = WORKSPACE / "audit_work" / "Project" / "Jupyter Notebook"
BASE_DIR = PROJECT_ROOT / "data" / "processed" / "point_event_v2"
ALIGNMENT_DIR = BASE_DIR / "nf3_alignment_v2"
MODELING_DIR = BASE_DIR / "modeling_v4"
TRANSFORMER_DIR = BASE_DIR / "transformer_v2"
RGCN_DIR = BASE_DIR / "rgcn_leakage_safe_v2"
OUTPUT_DIR = BASE_DIR / "validity_extensions_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FEATURES_IN = ALIGNMENT_DIR / "candidate_point_events_v2_with_context_and_nf3_primary.csv"
SPLITS_IN = MODELING_DIR / "model_split_assignments_v4.csv"
SELECTED_EDGES_IN = RGCN_DIR / "selected_event_flow_edges_v2.csv"
RGCN_CONFIG_IN = RGCN_DIR / "rgcn_config_v2.json"

SAMPLES_OUT = OUTPUT_DIR / "standardized_latency_samples_v3.csv"
EVENT_MEDIANS_OUT = OUTPUT_DIR / "standardized_latency_event_medians_v3.csv"
SUMMARY_OUT = OUTPUT_DIR / "standardized_latency_summary_v3.csv"
MANIFEST_OUT = OUTPUT_DIR / "standardized_latency_manifest_v3.json"

DEVICE = torch.device("cpu")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_checkpoint(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint is not a dictionary: {path}")
    return cast(dict[str, Any], payload)


def dense_float32(value: Any) -> np.ndarray:
    if sparse.issparse(value):
        value = value.toarray()
    result = np.asarray(value, dtype=np.float32)
    if result.ndim != 2 or not np.isfinite(result).all():
        raise ValueError(f"Invalid transformed feature matrix: shape={result.shape}")
    return result


class TemporalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, feedforward_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm_attention = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout_attention = nn.Dropout(dropout)
        self.norm_feedforward = nn.LayerNorm(d_model)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        inputs: torch.Tensor,
        causal_mask: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.norm_attention(inputs)
        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        hidden = inputs + self.dropout_attention(attention_output)
        return hidden + self.feedforward(self.norm_feedforward(hidden))


class TemporalTransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        sequence_length: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        feedforward_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.sequence_length = sequence_length
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.time_projection = nn.Sequential(nn.Linear(1, d_model), nn.Tanh())
        self.position_embedding = nn.Embedding(sequence_length, d_model)
        self.layers = nn.ModuleList(
            [
                TemporalTransformerBlock(d_model, n_heads, feedforward_dim, dropout)
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(d_model)
        hidden_out = max(16, d_model // 2)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, hidden_out),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_out, 1),
        )

    def forward(
        self,
        sequence_features: torch.Tensor,
        sequence_time: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, _ = sequence_features.shape
        positions = torch.arange(sequence_length, device=sequence_features.device)
        hidden = (
            self.input_projection(sequence_features)
            + self.time_projection(sequence_time)
            + self.position_embedding(positions).unsqueeze(0).expand(batch_size, -1, -1)
        )
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=sequence_features.device,
            ),
            diagonal=1,
        )
        for layer in self.layers:
            hidden = cast(TemporalTransformerBlock, layer)(
                hidden, causal_mask=causal_mask, padding_mask=padding_mask
            )
        return self.classifier(self.final_norm(hidden)[:, -1, :]).squeeze(-1)


class RelationalGraphConvolution(nn.Module):
    def __init__(self, hidden_dim: int, relation_names: Sequence[str], dropout: float) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_dim, hidden_dim)
        self.relation_names = tuple(relation_names)
        self.relation_linears = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, hidden_dim, bias=False) for name in relation_names}
        )
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_state: torch.Tensor,
        relation_tensors: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        output = self.self_linear(node_state)
        for relation_name in self.relation_names:
            source, destination, weight = relation_tensors[relation_name]
            if source.numel() == 0:
                continue
            messages = self.relation_linears[relation_name](
                node_state.index_select(0, source)
            ) * weight.unsqueeze(1)
            aggregate = torch.zeros_like(output)
            aggregate.index_add_(0, destination, messages)
            degree = torch.zeros(
                node_state.shape[0], dtype=node_state.dtype, device=node_state.device
            )
            degree.index_add_(0, destination, weight.abs())
            output = output + aggregate / degree.clamp_min(1.0).unsqueeze(1)
        return self.layer_norm(self.dropout(output))


class RGCNEventClassifier(nn.Module):
    def __init__(
        self,
        event_input_dim: int,
        flow_input_dim: int,
        hidden_dim: int,
        relation_names: Sequence[str],
        layer_count: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.event_encoder = nn.Sequential(
            nn.Linear(event_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.flow_encoder = nn.Sequential(
            nn.Linear(flow_input_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.node_type_embedding = nn.Embedding(2, hidden_dim)
        self.layers = nn.ModuleList(
            [
                RelationalGraphConvolution(hidden_dim, relation_names, dropout)
                for _ in range(layer_count)
            ]
        )
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        event_features: torch.Tensor,
        flow_features: torch.Tensor,
        relation_tensors: Mapping[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]],
    ) -> torch.Tensor:
        event_state = self.event_encoder(event_features)
        flow_state = self.flow_encoder(flow_features)
        node_state = torch.cat([event_state, flow_state], dim=0)
        type_ids = torch.cat(
            [
                torch.zeros(event_state.shape[0], dtype=torch.long),
                torch.ones(flow_state.shape[0], dtype=torch.long),
            ]
        )
        node_state = node_state + self.node_type_embedding(type_ids)
        for layer in self.layers:
            residual = node_state
            node_state = self.activation(
                cast(RelationalGraphConvolution, layer)(node_state, relation_tensors)
            )
            node_state = self.dropout(node_state + residual)
        return self.classifier(node_state[: event_state.shape[0]]).squeeze(1)


@dataclass(frozen=True)
class RelationEdges:
    source: np.ndarray
    destination: np.ndarray
    weight: np.ndarray


@dataclass(frozen=True)
class GraphBundle:
    name: str
    relations: dict[str, RelationEdges]


@dataclass(frozen=True)
class LocalPackage:
    event_x: torch.Tensor
    flow_x: torch.Tensor
    relations: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
    target_local_event: int
    local_event_count: int
    local_flow_count: int
    local_edge_count: int


features = pd.read_csv(FEATURES_IN, low_memory=False)
splits = pd.read_csv(SPLITS_IN, low_memory=False)
model_data = features.merge(
    splits[["event_id", "model_target", "chronological_split"]],
    on="event_id",
    how="inner",
    validate="one_to_one",
)
track_radians = np.deg2rad(
    pd.to_numeric(model_data["track_degrees"], errors="coerce")
)
model_data["track_sin"] = np.sin(track_radians)
model_data["track_cos"] = np.cos(track_radians)
model_data["aircraft_event_time_utc"] = pd.to_datetime(
    model_data["aircraft_event_time_utc"], utc=True
)
model_data["event_id"] = model_data["event_id"].astype("string")
model_data["aircraft_key"] = model_data["aircraft_key"].astype("string")
model_data["episode_id"] = model_data["episode_id"].astype("string")
model_data["chronological_split"] = model_data["chronological_split"].astype("string")
model_data["model_target"] = pd.to_numeric(
    model_data["model_target"], errors="raise"
).astype("int8")
model_data = model_data.sort_values(
    ["aircraft_event_time_utc", "event_id"], kind="mergesort"
).reset_index(drop=True)
model_data["global_position"] = np.arange(len(model_data), dtype=np.int64)

test = model_data.loc[model_data["chronological_split"].eq("test")].reset_index(drop=True)
rng = np.random.default_rng(RANDOM_STATE + 1800)
sample_parts: list[np.ndarray] = []
for label in [0, 1]:
    positions = np.flatnonzero(test["model_target"].to_numpy(dtype=np.int8) == label)
    take = min(len(positions), SAMPLE_EVENTS // 2)
    sample_parts.append(rng.choice(positions, size=take, replace=False))
selected_test_rows = np.sort(np.concatenate(sample_parts))
if len(selected_test_rows) < SAMPLE_EVENTS:
    remaining = np.setdiff1d(np.arange(len(test)), selected_test_rows)
    add = min(len(remaining), SAMPLE_EVENTS - len(selected_test_rows))
    selected_test_rows = np.sort(
        np.concatenate([selected_test_rows, rng.choice(remaining, add, replace=False)])
    )
sample = test.iloc[selected_test_rows].copy().reset_index(drop=True)

# Random forests.
rf_models: dict[str, Any] = {}
rf_raw_columns: dict[str, list[str]] = {}
rf_transformed: dict[str, Any] = {}
for name, feature_set in [
    ("physical_random_forest", "physical"),
    ("cyber_random_forest", "cyber"),
    ("fused_random_forest", "fused"),
]:
    model = joblib.load(
        MODELING_DIR / "models" / f"chronological_primary__{feature_set}__random_forest.joblib"
    )
    model.named_steps["model"].set_params(n_jobs=1)
    columns: list[str] = []
    for transformer_name, _, transformer_columns in model.named_steps["preprocess"].transformers_:
        if transformer_name != "remainder":
            columns.extend([str(value) for value in list(transformer_columns)])
    rf_models[name] = model
    rf_raw_columns[name] = columns
    rf_transformed[name] = model.named_steps["preprocess"].transform(test[columns])

# Transformer.
transformer_checkpoint = load_checkpoint(
    TRANSFORMER_DIR / "models" / "fused_temporal_transformer.pt"
)
transformer_preprocessor = joblib.load(
    TRANSFORMER_DIR / "preprocessing" / "fused_preprocessor.joblib"
)
transformer_raw_columns = [str(x) for x in transformer_preprocessor.feature_names_in_]
for column in transformer_checkpoint["raw_categorical_columns"]:
    model_data[str(column)] = model_data[str(column)].astype("string")
transformer_matrix = dense_float32(
    transformer_preprocessor.transform(model_data[transformer_raw_columns])
)
architecture = cast(dict[str, Any], transformer_checkpoint["architecture"])
sequence_length = int(architecture["sequence_length"])
gap_mean = float(transformer_checkpoint["time_gap_mean_log1p"])
gap_std = float(transformer_checkpoint["time_gap_std_log1p"])
event_epoch_s = model_data["aircraft_event_time_utc"].astype("int64").to_numpy() / 1e9
site_gap_s = np.diff(event_epoch_s, prepend=event_epoch_s[0])
site_gap_s[0] = 0.0
standardized_gap = ((np.log1p(np.maximum(site_gap_s, 0.0)) - gap_mean) / gap_std).astype(
    np.float32
)
transformer_model = TemporalTransformerClassifier(
    input_dim=int(architecture["input_dim"]),
    sequence_length=sequence_length,
    d_model=int(architecture["d_model"]),
    n_heads=int(architecture["n_heads"]),
    n_layers=int(architecture["n_layers"]),
    feedforward_dim=int(architecture["feedforward_dim"]),
    dropout=float(architecture["dropout"]),
)
transformer_model.load_state_dict(
    cast(dict[str, torch.Tensor], transformer_checkpoint["state_dict"])
)
transformer_model.eval()


def transformer_inputs_from_matrix(global_position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = max(0, global_position - sequence_length + 1)
    available = global_position - start + 1
    left = sequence_length - available
    x = np.zeros((1, sequence_length, transformer_matrix.shape[1]), dtype=np.float32)
    t = np.zeros((1, sequence_length, 1), dtype=np.float32)
    mask = np.ones((1, sequence_length), dtype=np.bool_)
    x[0, left:] = transformer_matrix[start : global_position + 1]
    t[0, left:, 0] = standardized_gap[start : global_position + 1]
    mask[0, left:] = False
    return torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(mask)


def transformer_inputs_from_raw(global_position: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    start = max(0, global_position - sequence_length + 1)
    positions = np.arange(start, global_position + 1, dtype=np.int64)
    transformed = dense_float32(
        transformer_preprocessor.transform(
            model_data.iloc[positions][transformer_raw_columns]
        )
    )
    available = len(positions)
    left = sequence_length - available
    x = np.zeros((1, sequence_length, transformed.shape[1]), dtype=np.float32)
    t = np.zeros((1, sequence_length, 1), dtype=np.float32)
    mask = np.ones((1, sequence_length), dtype=np.bool_)
    x[0, left:] = transformed
    raw_gaps = np.empty(available, dtype=np.float64)
    for i, position in enumerate(positions):
        raw_gaps[i] = 0.0 if position == 0 else max(
            0.0, event_epoch_s[position] - event_epoch_s[position - 1]
        )
    t[0, left:, 0] = ((np.log1p(raw_gaps) - gap_mean) / gap_std).astype(np.float32)
    mask[0, left:] = False
    return torch.from_numpy(x), torch.from_numpy(t), torch.from_numpy(mask)


# R-GCN feature banks and graphs.
selected_edges = pd.read_csv(SELECTED_EDGES_IN, low_memory=False)
selected_edges["event_id"] = selected_edges["event_id"].astype("string")
selected_edges["flow_id"] = selected_edges["flow_id"].astype("string")
selected_edges["flow_start_utc"] = pd.to_datetime(
    selected_edges["flow_start_utc"], utc=True, format="mixed"
)
selected_edges["flow_end_utc"] = pd.to_datetime(
    selected_edges["flow_end_utc"], utc=True, format="mixed"
)
selected_edges["relation_type"] = (
    selected_edges["relation_type"].astype("string").str.strip().str.lower()
)
flow_table = (
    selected_edges.sort_values(["flow_id", "flow_start_utc"], kind="mergesort")
    .drop_duplicates("flow_id", keep="first")
    .reset_index(drop=True)
)
flow_table["flow_duration_s"] = (
    flow_table["flow_end_utc"] - flow_table["flow_start_utc"]
).dt.total_seconds().clip(lower=0.0)
flow_table["protocol"] = flow_table["protocol"].astype("string").fillna("unknown")

event_preprocessor = joblib.load(
    RGCN_DIR / "preprocessing" / "fused_event_preprocessor.joblib"
)
flow_preprocessor = joblib.load(RGCN_DIR / "preprocessing" / "flow_preprocessor.joblib")
event_raw_columns = [str(x) for x in event_preprocessor.feature_names_in_]
flow_raw_columns = [str(x) for x in flow_preprocessor.feature_names_in_]
event_matrix = dense_float32(event_preprocessor.transform(model_data[event_raw_columns]))
flow_matrix = dense_float32(flow_preprocessor.transform(flow_table[flow_raw_columns]))

event_count = len(model_data)
event_node_index = {
    event_id: index for index, event_id in enumerate(model_data["event_id"].astype(str))
}
flow_node_index = {
    flow_id: event_count + index for index, flow_id in enumerate(flow_table["flow_id"].astype(str))
}
rgcn_config = json.loads(RGCN_CONFIG_IN.read_text(encoding="utf-8"))
lag_scale = float(rgcn_config["graph_policy"]["lag_weight_scale_s"])


def prior_edges(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    positions = pd.Series(np.arange(len(series), dtype=np.int64), index=series.index)
    previous = positions.groupby(series.astype("string"), sort=False).shift(1)
    valid = previous.notna()
    return (
        previous.loc[valid].astype("int64").to_numpy(),
        positions.loc[valid].astype("int64").to_numpy(),
    )


def add_relation(
    storage: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]],
    name: str,
    source: np.ndarray,
    destination: np.ndarray,
    weight: np.ndarray,
) -> None:
    if len(source):
        storage.setdefault(name, []).append(
            (
                np.asarray(source, dtype=np.int64),
                np.asarray(destination, dtype=np.int64),
                np.asarray(weight, dtype=np.float32),
            )
        )


def build_graph(variant: str) -> GraphBundle:
    storage: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}
    add_relation(
        storage,
        "site_previous_event",
        np.arange(event_count - 1, dtype=np.int64),
        np.arange(1, event_count, dtype=np.int64),
        np.ones(event_count - 1, dtype=np.float32),
    )
    source, destination = prior_edges(model_data["aircraft_key"])
    add_relation(
        storage,
        "aircraft_previous_event",
        source,
        destination,
        np.ones(len(source), dtype=np.float32),
    )
    source, destination = prior_edges(model_data["episode_id"])
    add_relation(
        storage,
        "episode_previous_event",
        source,
        destination,
        np.ones(len(source), dtype=np.float32),
    )
    chosen = selected_edges.copy()
    chosen["event_node"] = chosen["event_id"].astype(str).map(event_node_index)
    chosen["flow_node"] = chosen["flow_id"].astype(str).map(flow_node_index)
    lag = pd.to_numeric(chosen["absolute_flow_start_lag_s"], errors="raise").to_numpy()
    overlap = pd.to_numeric(chosen["overlap_seconds"], errors="raise").to_numpy()
    proximity = np.exp(-lag / max(lag_scale, 1e-6))
    overlap_scale = max(float(np.nanmax(overlap)), 1.0)
    overlap_weight = 0.5 + 0.5 * np.log1p(overlap) / np.log1p(overlap_scale)
    if variant == "full_lag_relation":
        mapping = {
            "pre_event": "flow_pre_event_to_event",
            "overlaps_event_time": "flow_overlaps_event_to_event",
            "post_event": "flow_post_event_to_event",
        }
        chosen["graph_relation"] = chosen["relation_type"].map(mapping).fillna(
            "flow_other_to_event"
        )
        chosen["graph_weight"] = np.clip(proximity * overlap_weight, 0.01, 1.0)
    elif variant == "lag_relation_ablation":
        chosen["graph_relation"] = "flow_to_event"
        chosen["graph_weight"] = 1.0
    else:
        raise ValueError(variant)
    for relation_name, group in chosen.groupby("graph_relation", sort=True):
        add_relation(
            storage,
            str(relation_name),
            group["flow_node"].to_numpy(dtype=np.int64),
            group["event_node"].to_numpy(dtype=np.int64),
            group["graph_weight"].to_numpy(dtype=np.float32),
        )
    relations = {
        name: RelationEdges(
            np.concatenate([part[0] for part in parts]),
            np.concatenate([part[1] for part in parts]),
            np.concatenate([part[2] for part in parts]),
        )
        for name, parts in storage.items()
    }
    return GraphBundle(variant, relations)


graphs = {
    "full_rgcn": build_graph("full_lag_relation"),
    "ablated_rgcn": build_graph("lag_relation_ablation"),
}
incoming: dict[str, dict[int, list[tuple[str, int, float]]]] = {}
for model_name, bundle in graphs.items():
    adjacency: dict[int, list[tuple[str, int, float]]] = {}
    for relation_name, edges in bundle.relations.items():
        for source, destination, weight in zip(edges.source, edges.destination, edges.weight):
            adjacency.setdefault(int(destination), []).append(
                (relation_name, int(source), float(weight))
            )
    incoming[model_name] = adjacency

rgcn_models: dict[str, RGCNEventClassifier] = {}
for model_name, checkpoint_name in [
    ("full_rgcn", "full_lag_relation_rgcn.pt"),
    ("ablated_rgcn", "lag_relation_ablation_rgcn.pt"),
]:
    checkpoint = load_checkpoint(RGCN_DIR / "models" / checkpoint_name)
    relation_names = [str(x) for x in checkpoint["relation_names"]]
    if set(relation_names) != set(graphs[model_name].relations):
        raise AssertionError(f"Graph/checkpoint relation mismatch for {model_name}")
    model = RGCNEventClassifier(
        event_input_dim=int(checkpoint["event_input_dim"]),
        flow_input_dim=int(checkpoint["flow_input_dim"]),
        hidden_dim=int(checkpoint["hidden_dim"]),
        relation_names=relation_names,
        layer_count=int(checkpoint["layer_count"]),
        dropout=float(checkpoint["dropout"]),
    )
    model.load_state_dict(cast(dict[str, torch.Tensor], checkpoint["model_state_dict"]))
    model.eval()
    rgcn_models[model_name] = model


def local_topology(model_name: str, target: int) -> tuple[list[int], list[int], dict[str, list[tuple[int, int, float]]]]:
    adjacency = incoming[model_name]
    selected: set[int] = {target}
    frontier: set[int] = {target}
    for _ in range(LOCAL_HOPS):
        next_frontier: set[int] = set()
        for destination in frontier:
            for _, source, _ in adjacency.get(destination, []):
                if source not in selected:
                    selected.add(source)
                    next_frontier.add(source)
        frontier = next_frontier
        if not frontier:
            break
    local_events = sorted(node for node in selected if node < event_count)
    local_flows = sorted(node for node in selected if node >= event_count)
    ordered = local_events + local_flows
    remap = {node: index for index, node in enumerate(ordered)}
    relation_parts: dict[str, list[tuple[int, int, float]]] = {
        name: [] for name in graphs[model_name].relations
    }
    for destination in ordered:
        for relation, source, weight in adjacency.get(destination, []):
            if source in remap:
                relation_parts[relation].append(
                    (remap[source], remap[destination], weight)
                )
    return local_events, local_flows, relation_parts


def package_from_topology(
    model_name: str,
    target: int,
    topology: tuple[list[int], list[int], dict[str, list[tuple[int, int, float]]]],
    raw_preprocessing: bool,
) -> LocalPackage:
    local_events, local_flows, relation_parts = topology
    if raw_preprocessing:
        event_x_np = dense_float32(
            event_preprocessor.transform(model_data.iloc[local_events][event_raw_columns])
        )
        flow_positions = [node - event_count for node in local_flows]
        if flow_positions:
            flow_x_np = dense_float32(
                flow_preprocessor.transform(flow_table.iloc[flow_positions][flow_raw_columns])
            )
        else:
            flow_x_np = np.empty((0, flow_matrix.shape[1]), dtype=np.float32)
    else:
        event_x_np = event_matrix[np.asarray(local_events, dtype=np.int64)]
        flow_positions = np.asarray(
            [node - event_count for node in local_flows], dtype=np.int64
        )
        flow_x_np = (
            flow_matrix[flow_positions]
            if len(flow_positions)
            else np.empty((0, flow_matrix.shape[1]), dtype=np.float32)
        )
    relation_tensors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    edge_count = 0
    for relation_name in rgcn_models[model_name].layers[0].relation_names:
        parts = relation_parts.get(relation_name, [])
        edge_count += len(parts)
        relation_tensors[relation_name] = (
            torch.tensor([part[0] for part in parts], dtype=torch.long),
            torch.tensor([part[1] for part in parts], dtype=torch.long),
            torch.tensor([part[2] for part in parts], dtype=torch.float32),
        )
    return LocalPackage(
        event_x=torch.from_numpy(np.asarray(event_x_np, dtype=np.float32)),
        flow_x=torch.from_numpy(np.asarray(flow_x_np, dtype=np.float32)),
        relations=relation_tensors,
        target_local_event=local_events.index(target),
        local_event_count=len(local_events),
        local_flow_count=len(local_flows),
        local_edge_count=edge_count,
    )


def score_transformer(inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> float:
    with torch.inference_mode():
        return float(torch.sigmoid(transformer_model(*inputs)[0]).item())


def score_rgcn(model_name: str, package: LocalPackage) -> float:
    with torch.inference_mode():
        logits = rgcn_models[model_name](package.event_x, package.flow_x, package.relations)
        return float(torch.sigmoid(logits[package.target_local_event]).item())


# Cache every model-ready object before timing.
ready_transformer = {
    int(row.global_position): transformer_inputs_from_matrix(int(row.global_position))
    for row in sample.itertuples()
}
topologies: dict[str, dict[int, Any]] = {name: {} for name in graphs}
ready_rgcn: dict[str, dict[int, LocalPackage]] = {name: {} for name in graphs}
for model_name in graphs:
    for row in sample.itertuples():
        position = int(row.global_position)
        topology = local_topology(model_name, position)
        topologies[model_name][position] = topology
        ready_rgcn[model_name][position] = package_from_topology(
            model_name, position, topology, raw_preprocessing=False
        )


def make_scorers(sample_row: int) -> list[tuple[str, str, Any, dict[str, int]]]:
    row = sample.iloc[int(sample_row)]
    global_position = int(row["global_position"])
    scorers: list[tuple[str, str, Any, dict[str, int]]] = []
    for model_name in [
        "physical_random_forest",
        "cyber_random_forest",
        "fused_random_forest",
    ]:
        model = rf_models[model_name]
        test_row = int(selected_test_rows[int(sample_row)])
        raw_columns = rf_raw_columns[model_name]
        transformed_bank = rf_transformed[model_name]
        ready_row = transformed_bank[test_row : test_row + 1]
        scorers.extend(
            [
                (
                    model_name,
                    "model_ready_inference",
                    lambda m=model, x=ready_row: float(
                        m.named_steps["model"].predict_proba(x)[0, 1]
                    ),
                    {},
                ),
                (
                    model_name,
                    "in_memory_raw_to_score",
                    lambda m=model, cols=raw_columns, gp=global_position: float(
                        m.predict_proba(model_data.iloc[[gp]][cols])[0, 1]
                    ),
                    {},
                ),
            ]
        )
    scorers.extend(
        [
            (
                "temporal_transformer",
                "model_ready_inference",
                lambda gp=global_position: score_transformer(ready_transformer[gp]),
                {},
            ),
            (
                "temporal_transformer",
                "in_memory_raw_to_score",
                lambda gp=global_position: score_transformer(
                    transformer_inputs_from_raw(gp)
                ),
                {},
            ),
        ]
    )
    for model_name in ["full_rgcn", "ablated_rgcn"]:
        package = ready_rgcn[model_name][global_position]
        metadata = {
            "local_event_nodes": package.local_event_count,
            "local_flow_nodes": package.local_flow_count,
            "local_edges": package.local_edge_count,
        }
        scorers.extend(
            [
                (
                    model_name,
                    "model_ready_inference",
                    lambda mn=model_name, p=package: score_rgcn(mn, p),
                    metadata,
                ),
                (
                    model_name,
                    "in_memory_raw_to_score",
                    lambda mn=model_name, gp=global_position: score_rgcn(
                        mn,
                        package_from_topology(
                            mn, gp, local_topology(mn, gp), raw_preprocessing=True
                        ),
                    ),
                    metadata,
                ),
            ]
        )
    return scorers


# Warm all combinations before measurement.
with threadpool_limits(limits=1):
    for sample_row in range(min(WARMUP_EVENTS, len(sample))):
        for _, _, scorer, _ in make_scorers(sample_row):
            scorer()

sample_rows: list[dict[str, Any]] = []
with threadpool_limits(limits=1):
    for sample_row in range(len(sample)):
        row = sample.iloc[sample_row]
        scorers = make_scorers(sample_row)
        order = rng.permutation(len(scorers))
        for scorer_position in order:
            model_name, boundary, scorer, metadata = scorers[int(scorer_position)]
            for repeat in range(REPEATS_PER_EVENT):
                started = time.perf_counter_ns()
                probability = scorer()
                elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000.0
                sample_rows.append(
                    {
                        "event_id": str(row["event_id"]),
                        "model_target": int(row["model_target"]),
                        "global_position": int(row["global_position"]),
                        "model": model_name,
                        "measurement_boundary": boundary,
                        "repeat": repeat,
                        "latency_ms": elapsed_ms,
                        "predicted_probability": probability,
                        **metadata,
                    }
                )
        if (sample_row + 1) % 25 == 0:
            print(f"Measured {sample_row + 1}/{len(sample)} held-out events")

latency_samples = pd.DataFrame(sample_rows)
latency_samples.to_csv(SAMPLES_OUT, index=False)
event_medians = (
    latency_samples.groupby(
        ["event_id", "model_target", "global_position", "model", "measurement_boundary"],
        as_index=False,
    )
    .agg(
        latency_ms=("latency_ms", "median"),
        predicted_probability=("predicted_probability", "median"),
        local_event_nodes=("local_event_nodes", "first"),
        local_flow_nodes=("local_flow_nodes", "first"),
        local_edges=("local_edges", "first"),
    )
)
event_medians.to_csv(EVENT_MEDIANS_OUT, index=False)

# Verify scoring output against the frozen event-level prediction banks.
frozen_predictions: dict[str, dict[str, float]] = {}
baseline_predictions = pd.read_csv(
    MODELING_DIR / "baseline_test_predictions_v4.csv", low_memory=False
)
for model_name, feature_set in [
    ("physical_random_forest", "physical"),
    ("cyber_random_forest", "cyber"),
    ("fused_random_forest", "fused"),
]:
    selected = baseline_predictions.loc[
        baseline_predictions["split_strategy"].astype(str).eq("chronological_primary")
        & baseline_predictions["feature_set"].astype(str).eq(feature_set)
        & baseline_predictions["model"].astype(str).eq("random_forest")
    ]
    frozen_predictions[model_name] = dict(
        zip(selected["event_id"].astype(str), selected["predicted_probability"].astype(float))
    )
transformer_predictions = pd.read_csv(
    TRANSFORMER_DIR / "temporal_transformer_predictions_v2.csv", low_memory=False
)
transformer_predictions = transformer_predictions.loc[
    transformer_predictions["split"].astype(str).eq("test")
]
frozen_predictions["temporal_transformer"] = dict(
    zip(
        transformer_predictions["event_id"].astype(str),
        transformer_predictions["predicted_probability"].astype(float),
    )
)
rgcn_predictions = pd.read_csv(RGCN_DIR / "rgcn_predictions_v2.csv", low_memory=False)
for model_name, variant in [
    ("full_rgcn", "full_lag_relation"),
    ("ablated_rgcn", "lag_relation_ablation"),
]:
    selected = rgcn_predictions.loc[
        rgcn_predictions["split"].astype(str).eq("test")
        & rgcn_predictions["variant"].astype(str).eq(variant)
    ]
    frozen_predictions[model_name] = dict(
        zip(selected["event_id"].astype(str), selected["probability"].astype(float))
    )

reproduction_errors: dict[str, float] = {}
for model_name in frozen_predictions:
    observed = event_medians.loc[
        event_medians["model"].eq(model_name)
        & event_medians["measurement_boundary"].eq("model_ready_inference")
    ]
    error = max(
        abs(float(row.predicted_probability) - frozen_predictions[model_name][str(row.event_id)])
        for row in observed.itertuples()
    )
    reproduction_errors[model_name] = float(error)
    tolerance = 1e-12 if "random_forest" in model_name else 1e-5
    if error > tolerance:
        raise AssertionError(
            f"{model_name} local scoring did not reproduce frozen predictions: {error:.8g}"
        )

summary_rng = np.random.default_rng(RANDOM_STATE + 1818)
summary_rows: list[dict[str, Any]] = []
for (model_name, boundary), group in event_medians.groupby(
    ["model", "measurement_boundary"], sort=True
):
    values = group["latency_ms"].to_numpy(dtype=float)
    p95_samples = np.empty(SUMMARY_BOOTSTRAPS, dtype=float)
    for repeat in range(SUMMARY_BOOTSTRAPS):
        p95_samples[repeat] = np.quantile(
            summary_rng.choice(values, size=len(values), replace=True), 0.95
        )
    summary_rows.append(
        {
            "model": model_name,
            "measurement_boundary": boundary,
            "events": len(group),
            "repeats_per_event": REPEATS_PER_EVENT,
            "latency_mean_ms": float(np.mean(values)),
            "latency_p50_ms": float(np.quantile(values, 0.50)),
            "latency_p95_ms": float(np.quantile(values, 0.95)),
            "latency_p95_bootstrap_ci_lower_ms": float(np.quantile(p95_samples, 0.025)),
            "latency_p95_bootstrap_ci_upper_ms": float(np.quantile(p95_samples, 0.975)),
            "latency_p99_ms": float(np.quantile(values, 0.99)),
            "latency_max_ms": float(np.max(values)),
            "events_over_500ms": int(np.sum(values > 500.0)),
            "reproduced_probability_max_abs_error": reproduction_errors[model_name],
            "median_local_event_nodes": float(group["local_event_nodes"].median())
            if group["local_event_nodes"].notna().any()
            else np.nan,
            "median_local_flow_nodes": float(group["local_flow_nodes"].median())
            if group["local_flow_nodes"].notna().any()
            else np.nan,
            "median_local_edges": float(group["local_edges"].median())
            if group["local_edges"].notna().any()
            else np.nan,
        }
    )
summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_OUT, index=False)

manifest = {
    "analysis_version": "standardized_model_latency_v3",
    "created_utc": datetime.now(timezone.utc).isoformat(),
    "hardware_and_runtime": {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit_learn": sklearn.__version__,
        "torch": torch.__version__,
        "device": "cpu",
        "torch_threads": torch.get_num_threads(),
        "threadpool_limit": 1,
    },
    "sample": {
        "held_out_events": len(sample),
        "negative_events": int((sample["model_target"] == 0).sum()),
        "positive_events": int((sample["model_target"] == 1).sum()),
        "sampling": "fixed-seed class-balanced sample from the common chronological test set",
        "repeats_per_event": REPEATS_PER_EVENT,
        "warmup_events": WARMUP_EVENTS,
        "event_level_summary": "median of repeated measurements per event",
        "p95_confidence_interval": f"percentile bootstrap over event medians ({SUMMARY_BOOTSTRAPS} replicates)",
    },
    "measurement_boundaries": {
        "model_ready_inference": (
            "Preprocessing and structural assembly completed before timing; measures only the "
            "frozen estimator or neural forward pass on a single event representation."
        ),
        "in_memory_raw_to_score": (
            "Begins with the required raw rows already resident in memory and includes feature "
            "preprocessing, sequence or local-subgraph assembly, tensor conversion, and the "
            "frozen model forward pass. Excludes disk I/O, model deserialization, and network I/O."
        ),
    },
    "models": [
        "physical_random_forest",
        "cyber_random_forest",
        "fused_random_forest",
        "temporal_transformer",
        "full_rgcn",
        "ablated_rgcn",
    ],
    "rgcn_local_hops": LOCAL_HOPS,
    "prediction_reproduction_max_abs_error": reproduction_errors,
    "inputs": {
        str(path.relative_to(PROJECT_ROOT)): sha256_file(path)
        for path in [FEATURES_IN, SPLITS_IN, SELECTED_EDGES_IN, RGCN_CONFIG_IN]
    },
    "interpretation_boundary": (
        "These measurements compare steady-state CPU scoring under matched software and thread "
        "conditions. They are not deployment service-level measurements."
    ),
}
MANIFEST_OUT.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

print("Standardized latency benchmark completed.")
print(summary.to_string(index=False))
