# ==============================================================================
#  NanoWakeWord: Lightweight, Intelligent Wake Word Detection
#  Copyright 2025 Arcosoph. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#
#  Project: https://github.com/arcosoph/nanowakeword
# ==============================================================================

"""                                               (✿◕‿◕✿)
Knowledge Distillation for Nanowakeword.

Trains a tiny "student" model to mimic the output of a larger "teacher" model.
The student is a stripped-down DNN - always architecture type "dnn" regardless
of what the teacher was - because the goal is maximum speed, not accuracy parity.

Loss = alpha * KL(student_soft || teacher_soft)   ← soft label matching
     + (1 - alpha) * BCE(student_logit, hard_label) ← ground truth anchoring

Temperature scaling softens the teacher's probability distribution so the
student learns from the full output distribution, not just the argmax.
"""

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from nanowakeword.modules.model import Model
from nanowakeword.utils.logger import print_info, print_error


def _build_student(teacher: Model, input_shape: tuple, dist_cfg: dict) -> Model:
    """
    Builds a tiny student Model with the same input/output interface as the teacher
    but a much smaller DNN backbone.

    Default targets ~10x compression vs a typical teacher:
      layer_size=8, n_blocks=1, embedding_dim=8  →  ~12K params / ~50KB ONNX
    Override via config distillation.student_* keys.
    """
    student_layer_size  = dist_cfg.get("student_layer_size", 8)
    student_n_blocks    = dist_cfg.get("student_n_blocks", 1)
    student_embedding   = dist_cfg.get("student_embedding_dim", 8)
    dropout_prob        = dist_cfg.get("student_dropout_prob", 0.1)

    # Build a minimal config for the student - always DNN
    student_config = {
        "activation_function": "relu",
        "embedding_dim": student_embedding,
    }

    student = Model(
        config=student_config,
        model_name=teacher.model_name + "_lite",
        n_classes=1,
        input_shape=input_shape,
        model_type="dnn",
        layer_dim=student_layer_size,
        n_blocks=student_n_blocks,
        dropout_prob=dropout_prob,
    )

    return student


def _count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def distill_model(
    teacher: Model,
    X_train,
    config,
    input_shape: tuple,
) -> Model:
    """
    Runs the full distillation pipeline and returns a trained student Model.

    Args:
        teacher:      The fully-trained teacher Model (nn.Module).
        X_train:      The training DataLoader (same one used for teacher training).
        config:       The full training config proxy/dict.
        input_shape:  Tuple (frames, features), e.g. (16, 96).

    Returns:
        A trained student Model ready for ONNX export.
    """
    import itertools

    dist_cfg     = config.get("distillation", {})
    steps        = dist_cfg.get("steps", 8000)
    temperature  = float(dist_cfg.get("temperature", 4.0))
    alpha        = float(dist_cfg.get("alpha", 0.7))   # weight on soft loss
    lr           = float(dist_cfg.get("learning_rate", 5e-4))
    log_interval = dist_cfg.get("log_interval", 500)

    device = teacher.device

    # Build student
    student = _build_student(teacher, input_shape, dist_cfg)
    student.to(device)

    teacher_params = _count_params(teacher)
    student_params = _count_params(student)
    compression    = teacher_params / max(student_params, 1)

    print_info(f"[Distillation] Teacher params : {teacher_params:,}")
    print_info(f"[Distillation] Student params : {student_params:,}  ({compression:.1f}x smaller)")
    print_info(f"[Distillation] Steps          : {steps}")
    print_info(f"[Distillation] Temperature    : {temperature}")
    print_info(f"[Distillation] Alpha (soft)   : {alpha}")

    # Optimizer
    all_params = list(student.parameters()) + list(student.classifier.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps
    )

    # Freeze teacher
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)

    student.train()
    student.classifier.train()

    # Training loop 
    data_iter   = iter(itertools.cycle(X_train))
    ema_loss    = None
    ema_alpha   = 0.02
    best_loss   = float("inf")
    best_state  = None

    pbar = tqdm(range(steps), desc="Distilling", unit="step")

    for step in pbar:
        features, labels, _ = next(data_iter)
        features = features.to(device)
        labels   = labels.to(device).float().view(-1)

        # Teacher soft targets (no grad)
        with torch.no_grad():
            teacher_logits = teacher(features).view(-1)
            # Temperature-scaled soft probabilities
            teacher_soft = torch.sigmoid(teacher_logits / temperature)

        # Student forward
        student_logits = student(features).view(-1)
        student_soft   = torch.sigmoid(student_logits / temperature)

        # Soft loss: binary KL divergence
        # KL(teacher || student) in binary form:
        #   teacher * log(teacher/student) + (1-teacher) * log((1-teacher)/(1-student))
        eps = 1e-7
        soft_loss = -(
            teacher_soft * torch.log(student_soft + eps)
            + (1 - teacher_soft) * torch.log(1 - student_soft + eps)
        ).mean()
        # Scale by T² as per Hinton et al. to preserve gradient magnitude
        soft_loss = soft_loss * (temperature ** 2)

        # Hard loss: standard BCE on ground-truth labels 
        hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)

        # Combined loss 
        loss = alpha * soft_loss + (1.0 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # Tracking
        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else ema_alpha * loss_val + (1 - ema_alpha) * ema_loss

        if ema_loss < best_loss:
            best_loss  = ema_loss
            best_state = copy.deepcopy(student.state_dict())

        if step % log_interval == 0:
            pbar.set_postfix({"ema_loss": f"{ema_loss:.4f}", "lr": f"{scheduler.get_last_lr()[0]:.2e}"})

    pbar.close()

    # Load best checkpoint
    if best_state is not None:
        student.load_state_dict(best_state)
        print_info(f"[Distillation] Best EMA loss: {best_loss:.4f}")

    student.eval()
    print_info("[Distillation] Student model ready.")

    return student


def distill_from_onnx(
    onnx_path: str,
    X_train,
    config,
    input_shape: tuple,
    output_dir: str,
    model_name: str,
) -> str:
    """
    Standalone distillation from an already-exported ONNX teacher.
    Useful for post-training lite model generation without re-training.

    Loads the ONNX teacher, runs distillation using the feature files
    specified in config's feature_manifest, and exports a _lite.onnx.

    Returns the path to the exported lite ONNX file.
    """
    import onnxruntime as ort
    from nanowakeword._export.onnx import export_onnx_model

    dist_cfg    = config.get("distillation", {})
    steps       = dist_cfg.get("steps", 8000)
    temperature = float(dist_cfg.get("temperature", 4.0))
    alpha       = float(dist_cfg.get("alpha", 0.7))
    lr          = float(dist_cfg.get("learning_rate", 5e-4))
    log_interval = dist_cfg.get("log_interval", 500)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # ONNX teacher wrapper
    sess_opts = ort.SessionOptions()
    sess_opts.inter_op_num_threads = 1
    sess_opts.intra_op_num_threads = 1
    teacher_sess = ort.InferenceSession(
        onnx_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )

    def onnx_teacher_logit(features_np: np.ndarray) -> torch.Tensor:
        """Run ONNX teacher and return logits (inverse sigmoid of output prob)."""
        out = teacher_sess.run(None, {"input": features_np})[0]  # [B, 1, 1]
        prob = torch.tensor(out.reshape(-1), dtype=torch.float32)
        # Convert probability back to logit space for temperature scaling
        prob = prob.clamp(1e-7, 1 - 1e-7)
        return torch.log(prob / (1 - prob))

    # Build student
    # We need a dummy teacher Model just to reuse _build_student
    dummy_teacher_cfg = {"activation_function": "relu", "embedding_dim": 8}
    dummy_teacher = Model(
        config=dummy_teacher_cfg,
        model_name=model_name,
        n_classes=1,
        input_shape=input_shape,
        model_type="dnn",
        layer_dim=8,
        n_blocks=1,
    )
    dummy_teacher.model_name = model_name

    student = _build_student(dummy_teacher, input_shape, dist_cfg)
    student.to(device)

    print_info(f"[Distillation] Student params: {_count_params(student):,}")
    print_info(f"[Distillation] Steps: {steps}, Temperature: {temperature}, Alpha: {alpha}")

    # Optimizer
    import itertools
    all_params = list(student.parameters()) + list(student.classifier.parameters())
    optimizer  = torch.optim.AdamW(all_params, lr=lr, weight_decay=1e-3)
    scheduler  = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, total_steps=steps
    )

    student.train()
    student.classifier.train()

    data_iter  = iter(itertools.cycle(X_train))
    ema_loss   = None
    ema_alpha  = 0.02
    best_loss  = float("inf")
    best_state = None

    pbar = tqdm(range(steps), desc="Distilling (from ONNX)", unit="step")

    for step in pbar:
        features, labels, _ = next(data_iter)
        features_np = features.numpy().astype(np.float32)
        labels      = labels.to(device).float().view(-1)

        # Teacher soft targets via ONNX
        with torch.no_grad():
            teacher_logits = onnx_teacher_logit(features_np).to(device)
            teacher_soft   = torch.sigmoid(teacher_logits / temperature)

        # Student forward
        features_t     = features.to(device)
        student_logits = student(features_t).view(-1)
        student_soft   = torch.sigmoid(student_logits / temperature)

        eps       = 1e-7
        soft_loss = -(
            teacher_soft * torch.log(student_soft + eps)
            + (1 - teacher_soft) * torch.log(1 - student_soft + eps)
        ).mean() * (temperature ** 2)

        hard_loss = F.binary_cross_entropy_with_logits(student_logits, labels)
        loss      = alpha * soft_loss + (1.0 - alpha) * hard_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(all_params, max_norm=1.0)
        optimizer.step()
        scheduler.step()

        loss_val = loss.item()
        ema_loss = loss_val if ema_loss is None else ema_alpha * loss_val + (1 - ema_alpha) * ema_loss

        if ema_loss < best_loss:
            best_loss  = ema_loss
            best_state = copy.deepcopy(student.state_dict())

        if step % log_interval == 0:
            pbar.set_postfix({"ema_loss": f"{ema_loss:.4f}"})

    pbar.close()

    if best_state is not None:
        student.load_state_dict(best_state)

    student.eval()

    # Export
    lite_name = model_name + "_lite"
    export_onnx_model(
        model=student,
        input_shape=input_shape,
        config=config,
        model_name=lite_name,
        output_dir=output_dir,
    )

    lite_path = f"{output_dir}/{lite_name}.onnx"
    print_info(f"[Distillation] Lite model exported to: {lite_path}")
    return lite_path
