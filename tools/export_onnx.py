"""Export a safe NumPy artifact or optional reference actor to ONNX profile v1."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from racing_event.model_artifact import load_participant_artifact
from racing_event.observation import TrainingContract
from racing_event.onnx_artifact import FORMAT, load_onnx_artifact
from racing_event.trajectory import engine_code_hash


def export(source: Path, output: Path, *, reference: bool = False) -> Path:
    import onnx
    from onnx import helper as h, numpy_helper as nh

    if output.exists() or output.is_symlink():
        raise ValueError("ONNX output must be a new directory")
    if reference:
        from racing_event.ppo import FixedMLPActorCritic
        policy, _ = FixedMLPActorCritic.load_artifact(source / "policy.pt", source / "policy.json")
        contract, normalizer = TrainingContract.load(source / "training_contract.json")
        state = policy.state_dict()
        layers = [(state[f"{name}.weight"].numpy(), state[f"{name}.bias"].numpy(), "tanh")
                  for name in ("trunk.0", "trunk.2", "policy_mean")]
        mean, scale, clip = (None, None, None) if normalizer is None else (normalizer.mean, normalizer.std, normalizer.clip)
    else:
        artifact = load_participant_artifact(source)
        if artifact.manifest["environment_code_hash"] != engine_code_hash():
            raise ValueError("source artifact belongs to a different engine")
        contract, normalizer = artifact.contract, artifact.normalizer
        layers = [(layer.weights, layer.bias, layer.activation) for layer in artifact.layers]
        mean, scale, clip = ((normalizer.mean, normalizer.std, normalizer.clip) if normalizer is not None
                             else (artifact._affine_mean, artifact._affine_scale, artifact._affine_clip))
    nodes, initializers = [], []

    def constant(name, value):
        initializers.append(nh.from_array(np.asarray(value), name))
        return name

    value = "observations"
    if mean is not None:
        # Match the reference normalizer's float64 arithmetic, then cast once.
        dtype = np.float64 if normalizer is not None else np.float32
        if normalizer is not None:
            nodes.append(h.make_node("Cast", [value], ["obs64"], to=11))
            value = "obs64"
        nodes += [h.make_node("Sub", [value, constant("mean", np.asarray(mean, dtype=dtype))], ["centered"]),
                  h.make_node("Div", ["centered", constant("scale", np.asarray(scale, dtype=dtype))], ["scaled"])]
        bounds = np.asarray(clip, dtype=dtype)
        low, high = constant("clip_min", -bounds), constant("clip_max", bounds)
        if bounds.ndim == 0:
            nodes.append(h.make_node("Clip", ["scaled", low, high], ["clipped"]))
        else:
            # ONNX Clip accepts scalar bounds only; affine preprocessing permits
            # independently bounded fields, preserved through broadcast Min/Max.
            nodes += [h.make_node("Max", ["scaled", low], ["lower_bounded"]),
                      h.make_node("Min", ["lower_bounded", high], ["clipped"])]
        value = "clipped"
        if normalizer is not None:
            nodes.append(h.make_node("Cast", [value], ["normalized"], to=1))
            value = "normalized"
    for index, (weights, bias, activation) in enumerate(layers):
        dense = f"dense_{index}"
        final = index == len(layers) - 1
        nodes.append(h.make_node("Gemm", [value, constant(f"w{index}", weights), constant(f"b{index}", bias)], [dense], transB=1))
        value = "actions" if final else f"features_{index}"
        nodes.append(h.make_node({"tanh": "Tanh", "relu": "Relu", "identity": "Identity"}[activation], [dense], [value]))
    graph = h.make_graph(nodes, "participant_policy", [h.make_tensor_value_info("observations", 1, [1, contract.observation.size])],
                         [h.make_tensor_value_info("actions", 1, [1, 2])], initializer=initializers)
    model = h.make_model(graph, opset_imports=[h.make_opsetid("", 17)], ir_version=10)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".onnx-", dir=output.parent))
    try:
        (temporary / "model.onnx").write_bytes(model.SerializeToString())
        contract.save(temporary / "training_contract.json")
        manifest = {"format": FORMAT, "format_version": 1, "profile": "static_vector_cpu_opset17_v1", "environment_code_hash": engine_code_hash()}
        for name, key in (("model.onnx", "model_sha256"), ("training_contract.json", "training_contract_sha256")):
            manifest[key] = hashlib.sha256((temporary / name).read_bytes()).hexdigest()
        (temporary / "manifest.json").write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
        load_onnx_artifact(temporary)
        os.rename(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", action="store_true", help="source is optional Torch reference actor, not NumPy artifact")
    args = parser.parse_args()
    try:
        print(export(args.source, args.output, reference=args.reference))
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"ONNX export failed: {exc}\n")


if __name__ == "__main__":
    main()
