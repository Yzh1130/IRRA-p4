import argparse
import json
import os
import os.path as op
import sys
import tempfile
from types import SimpleNamespace

import torch


ROOT_DIR = op.dirname(op.dirname(op.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from datasets import build_dataloader  # noqa: E402
from datasets.build import collate  # noqa: E402
from datasets.rgbtpedes import RGBTPEDES  # noqa: E402
from model import build_model  # noqa: E402


def _load_config_file(path):
    if not path:
        return {}
    if not op.isfile(path):
        raise RuntimeError(f"config_file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _default_args():
    return {
        "dataset_name": "RGBT-PEDES",
        "root_dir": "datasets/TextReID",
        "training": True,
        "num_workers": 0,
        "batch_size": 2,
        "test_batch_size": 2,
        "num_instance": 2,
        "sampler": "random",
        "distributed": False,
        "img_size": (384, 128),
        "img_aug": False,
        "MLM": False,
        "text_length": 77,
        "val_dataset": "test",
        "use_modality_text_prompt": False,
        "mtp_color_drop_mode": "delete",
        "use_synth_caption_bank": False,
        "synth_caption_bank_file": "",
        "synth_caption_keep_per_image": 2,
        "synth_caption_train_mode": "append",
        "use_synth_loss_weighting": False,
        "synth_loss_weight": 1.0,
        "use_offline_hardneg": False,
        "offline_hardneg_index_file": "",
        "offline_hn_num_neg": 1,
        "offline_hn_topk_train": 20,
        "offline_hn_sample_mode": "random",
        "use_mga_pseudo": True,
        "mga_pseudo_json": "",
        "mga_min_weight": 0.0,
        "mga_pseudo_sample_prob": 1.0,
        "mga_attr_sample_prob": 1.0,
        "mga_debug_print_samples": 2,
        "pretrain_choice": "ViT-B/16",
        "temperature": 0.02,
        "stride_size": 16,
        "loss_names": "",
        "cmt_depth": 4,
        "vocab_size": 49408,
        "use_global_uncertainty": False,
        "uncertainty_dist_type": "diag_gaussian",
        "logvar_min": -6.0,
        "logvar_max": 4.0,
        "use_precision_fusion": None,
        "uncertainty_loss_type": "w2",
        "uncertainty_eps": 1e-6,
        "use_distribution_sdm_aux": False,
        "use_mum": False,
        "use_rgbt_add_fusion": True,
        "use_rgbt_moe_adapter": False,
        "use_noise_token_masking": False,
        "use_informative_token_masking": False,
        "token_mask_score_source": "none",
        "use_proto_calib": False,
        "proto_version": "v1",
        "proto_momentum": 0.9,
        "use_image_local_filter": False,
        "use_img_local_filter": False,
        "use_attr_aux": False,
        "lambda_attr_aux": 0.0,
        "attr_feature_source": "fused_det",
        "use_uspc_slot_local": False,
        "lambda_uspc_slot_local": 0.0,
        "use_mga_attr_align": False,
        "use_mga_negative_loss": False,
        "lambda_mga_pseudo_sdm": 0.0,
        "lambda_mga_rgb_color": 0.0,
        "lambda_mga_t_shape": 0.0,
        "lambda_mga_negative": 0.0,
        "mga_neg_margin": 0.2,
        "use_rde_robust": False,
        "rde_warmup_epochs": 10,
        "rde_enable_pseudo_during_warmup": False,
        "rde_conf_source": "hybrid",
        "rde_clean_threshold": 0.70,
        "rde_uncertain_threshold": 0.40,
        "rde_uncertain_weight": 0.5,
        "rde_noisy_weight": 0.1,
        "lambda_rde_consensus": 0.0,
        "use_confidence_hard_negative": False,
        "conf_hn_min_weight": 0.5,
    }


def _make_temp_mga_json(root_dir):
    dataset = RGBTPEDES(root=root_dir, verbose=False)
    first_by_image_id = {}
    for sample in dataset.train:
        if len(sample) != 5:
            continue
        pid, image_id, rgb_path, t_path, caption = sample
        if image_id not in first_by_image_id:
            first_by_image_id[image_id] = (pid, image_id, rgb_path, t_path, caption)
        if len(first_by_image_id) >= 2:
            break
    if not first_by_image_id:
        raise RuntimeError("No RGBT-PEDES train samples found for temp MGA JSONL.")

    handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".mga.jsonl",
        delete=False,
    )
    with handle:
        for pid, image_id, rgb_path, t_path, caption in first_by_image_id.values():
            record = {
                "pid": int(pid),
                "image_id": int(image_id),
                "rgb_path": rgb_path,
                "thermal_path": t_path,
                "caption": caption,
                "pseudo_texts": [
                    {
                        "text": f"MGA full pseudo for image {image_id}.",
                        "kind": "full",
                        "weight": 0.95,
                        "source": "smoke",
                    }
                ],
                "rgb_color_texts": [
                    {
                        "text": f"RGB color pseudo for image {image_id}.",
                        "weight": 0.90,
                        "source": "smoke",
                    }
                ],
                "thermal_shape_texts": [
                    {
                        "text": f"Thermal shape pseudo for image {image_id}.",
                        "weight": 0.85,
                        "source": "smoke",
                    }
                ],
                "negative_texts": [
                    {
                        "text": f"Absent attribute pseudo for image {image_id}.",
                        "weight": 0.80,
                        "source": "smoke",
                    }
                ],
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return handle.name


def main():
    parser = argparse.ArgumentParser(description="Smoke test MGA dataloader fields.")
    parser.add_argument("--config_file", default="", help="optional JSON config file")
    parser.add_argument(
        "--mga_pseudo_json",
        default="",
        help="MGA pseudo JSONL; when empty, a temporary 2-row JSONL is created",
    )
    parser.add_argument("--num_batches", type=int, default=1)
    parser.add_argument("--root_dir", default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--forward_smoke",
        default=False,
        action="store_true",
        help="also build IRRA and verify MGA forward outputs",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="device used by --forward_smoke",
    )
    parser.add_argument(
        "--mga_loss_smoke",
        default=False,
        action="store_true",
        help="also compute MGA auxiliary loss terms from the forward output",
    )
    parser.add_argument("--lambda_mga_pseudo_sdm", type=float, default=0.0)
    parser.add_argument("--lambda_mga_rgb_color", type=float, default=0.0)
    parser.add_argument("--lambda_mga_t_shape", type=float, default=0.0)
    parser.add_argument("--lambda_mga_negative", type=float, default=0.0)
    parser.add_argument("--mga_neg_margin", type=float, default=0.2)
    parser.add_argument("--use_mga_attr_align", default=False, action="store_true")
    parser.add_argument("--use_mga_negative_loss", default=False, action="store_true")
    parser.add_argument("--use_rde_robust", default=False, action="store_true")
    parser.add_argument("--rde_warmup_epochs", type=int, default=10)
    parser.add_argument(
        "--rde_enable_pseudo_during_warmup",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--rde_conf_source",
        default="hybrid",
        choices=["mga_weight", "online_sim", "hybrid"],
    )
    parser.add_argument("--rde_clean_threshold", type=float, default=0.70)
    parser.add_argument("--rde_uncertain_threshold", type=float, default=0.40)
    parser.add_argument("--rde_uncertain_weight", type=float, default=0.5)
    parser.add_argument("--rde_noisy_weight", type=float, default=0.1)
    parser.add_argument("--lambda_rde_consensus", type=float, default=0.0)
    parser.add_argument(
        "--use_confidence_hard_negative",
        default=False,
        action="store_true",
    )
    parser.add_argument("--conf_hn_min_weight", type=float, default=0.5)
    parser.add_argument("--smoke_epoch", type=int, default=1)
    args = parser.parse_args()

    config = _default_args()
    config.update(_load_config_file(args.config_file))
    if args.root_dir is not None:
        config["root_dir"] = args.root_dir
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    config["lambda_mga_pseudo_sdm"] = float(args.lambda_mga_pseudo_sdm)
    config["lambda_mga_rgb_color"] = float(args.lambda_mga_rgb_color)
    config["lambda_mga_t_shape"] = float(args.lambda_mga_t_shape)
    config["lambda_mga_negative"] = float(args.lambda_mga_negative)
    config["mga_neg_margin"] = float(args.mga_neg_margin)
    config["use_rde_robust"] = bool(args.use_rde_robust)
    config["rde_warmup_epochs"] = int(args.rde_warmup_epochs)
    config["rde_enable_pseudo_during_warmup"] = bool(
        args.rde_enable_pseudo_during_warmup
    )
    config["rde_conf_source"] = str(args.rde_conf_source)
    config["rde_clean_threshold"] = float(args.rde_clean_threshold)
    config["rde_uncertain_threshold"] = float(args.rde_uncertain_threshold)
    config["rde_uncertain_weight"] = float(args.rde_uncertain_weight)
    config["rde_noisy_weight"] = float(args.rde_noisy_weight)
    config["lambda_rde_consensus"] = float(args.lambda_rde_consensus)
    config["use_confidence_hard_negative"] = bool(args.use_confidence_hard_negative)
    config["conf_hn_min_weight"] = float(args.conf_hn_min_weight)
    if (
        args.use_mga_attr_align
        or args.lambda_mga_rgb_color > 0.0
        or args.lambda_mga_t_shape > 0.0
    ):
        config["use_mga_attr_align"] = True
    if args.use_mga_negative_loss or args.lambda_mga_negative > 0.0:
        config["use_mga_negative_loss"] = True

    temp_json = None
    if args.mga_pseudo_json:
        config["mga_pseudo_json"] = args.mga_pseudo_json
    else:
        temp_json = _make_temp_mga_json(config["root_dir"])
        config["mga_pseudo_json"] = temp_json

    try:
        train_loader, _, _, _ = build_dataloader(SimpleNamespace(**config))
        expected_keys = [
            "mga_has_pseudo",
            "mga_pseudo_text",
            "mga_pseudo_weight",
            "mga_pseudo_kind",
            "mga_pseudo_ids",
            "mga_has_rgb_color",
            "mga_rgb_color_text",
            "mga_rgb_color_weight",
            "mga_rgb_color_ids",
            "mga_has_t_shape",
            "mga_t_shape_text",
            "mga_t_shape_weight",
            "mga_t_shape_ids",
            "mga_has_negative",
            "mga_negative_text",
            "mga_negative_weight",
            "mga_negative_ids",
        ]

        smoke_dataset = train_loader.dataset
        sample_count = min(int(config["batch_size"]), len(smoke_dataset))
        batch = collate([smoke_dataset[index] for index in range(sample_count)])

        for batch_idx in range(max(1, int(args.num_batches))):
            present = [key for key in expected_keys if key in batch]
            missing = [key for key in expected_keys if key not in batch]
            print(f"batch_index={batch_idx}")
            print(f"mga_pseudo_json={config['mga_pseudo_json']}")
            print(f"present_mga_keys={present}")
            print(f"missing_mga_keys={missing}")
            if "mga_has_pseudo" in batch:
                print(f"mga_has_pseudo={batch['mga_has_pseudo'].tolist()}")
                print(f"mga_pseudo_text={batch['mga_pseudo_text']}")
                print(f"mga_rgb_color_text={batch['mga_rgb_color_text']}")
                print(f"mga_t_shape_text={batch['mga_t_shape_text']}")
                print(f"mga_negative_text={batch['mga_negative_text']}")
            if missing:
                raise RuntimeError(f"Missing MGA batch keys: {missing}")
            if args.forward_smoke or args.mga_loss_smoke:
                device = args.device
                if device == "auto":
                    device = "cuda" if torch.cuda.is_available() else "cpu"
                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("--device cuda requested but CUDA is unavailable.")

                forward_config = dict(config)
                forward_config["use_mga_pseudo"] = True
                if args.mga_loss_smoke and not str(
                    forward_config.get("loss_names", "")
                ).strip():
                    forward_config["loss_names"] = "sdm"
                model = build_model(
                    SimpleNamespace(**forward_config),
                    num_classes=1,
                )
                if device == "cpu":
                    model = model.float()
                model.to(device)
                model.eval()
                batch_for_forward = {
                    key: value.to(device) if torch.is_tensor(value) else value
                    for key, value in batch.items()
                }
                with torch.no_grad():
                    ret = model(batch_for_forward)

                required_ret_keys = [
                    "mga_pseudo_text_feat",
                    "mga_pseudo_valid",
                    "mga_pseudo_weight",
                    "text_det_feat",
                ]
                missing_ret = [key for key in required_ret_keys if key not in ret]
                print(f"forward_device={device}")
                print(f"present_forward_keys={[key for key in required_ret_keys if key in ret]}")
                print(f"missing_forward_keys={missing_ret}")
                if missing_ret:
                    raise RuntimeError(f"Missing MGA forward keys: {missing_ret}")
                pseudo_shape = tuple(ret["mga_pseudo_text_feat"].shape)
                text_shape = tuple(ret["text_det_feat"].shape)
                print(f"mga_pseudo_text_feat_shape={pseudo_shape}")
                print(f"text_det_feat_shape={text_shape}")
                if pseudo_shape != text_shape:
                    raise RuntimeError(
                        "MGA pseudo feature shape mismatch: "
                        f"{pseudo_shape} vs {text_shape}."
                    )
                if args.mga_loss_smoke:
                    from processor.processor import (  # noqa: E402
                        _add_mga_aux_to_total_loss,
                        _build_total_loss,
                        _compute_mga_aux_terms,
                    )

                    loss_args = SimpleNamespace(**forward_config)
                    mga_terms = _compute_mga_aux_terms(
                        loss_args,
                        ret,
                        batch_for_forward,
                        epoch=int(args.smoke_epoch),
                    )
                    ret_with_mga = dict(ret)
                    ret_with_mga.update(mga_terms)
                    base_total = _build_total_loss(
                        loss_args,
                        ret,
                        epoch=int(args.smoke_epoch),
                    )
                    total = _add_mga_aux_to_total_loss(
                        loss_args,
                        base_total,
                        ret_with_mga,
                    )
                    loss_keys = [
                        "loss_mga_pseudo_sdm",
                        "loss_mga_rgb_color",
                        "loss_mga_t_shape",
                        "loss_mga_negative",
                        "loss_rde_consensus",
                    ]
                    ratio_keys = [
                        "mga_valid_pseudo_ratio",
                        "mga_valid_rgb_color_ratio",
                        "mga_valid_t_shape_ratio",
                        "rde_clean_ratio",
                        "rde_uncertain_ratio",
                        "rde_noisy_ratio",
                        "rde_mean_conf",
                        "conf_hn_active_ratio",
                    ]
                    print(f"present_mga_loss_keys={[key for key in loss_keys if key in mga_terms]}")
                    for key in loss_keys + ratio_keys:
                        if key in mga_terms:
                            value = mga_terms[key]
                            if torch.is_tensor(value):
                                value = value.detach().float().mean().item()
                            print(f"{key}={float(value):.6f}")
                    print(f"base_total_loss={base_total.detach().float().item():.6f}")
                    print(f"total_loss_with_mga={total.detach().float().item():.6f}")
                    print(
                        "mga_weighted_total_increment="
                        f"{(total - base_total).detach().float().item():.6f}"
                    )
            break
    finally:
        if temp_json is not None and op.isfile(temp_json):
            os.remove(temp_json)


if __name__ == "__main__":
    main()
