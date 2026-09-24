#!/usr/bin/env python3
"""Run out-of-fold U-Net inference for one cross-validation fold."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import SimpleITK as sitk
import torch
from picai_baseline.unet.training_setup.default_hyperparam import get_default_hyperparams
from picai_baseline.unet.training_setup.neural_network_selector import neural_network_for_run
from picai_baseline.unet.training_setup.preprocess_utils import z_score_norm
from picai_prep.data_utils import atomic_image_write
from picai_prep.preprocessing import Sample, PreprocessingSettings, crop_or_pad, resample_img
from report_guided_annotation import extract_lesion_candidates
from scipy.ndimage import gaussian_filter

IMAGE_DIRS = [
    "images/transverse-t2-prostate-mri",
    "images/transverse-adc-prostate-mri",
    "images/transverse-hbv-prostate-mri",
]
IMG_SPEC = {
    "image_shape": [20, 256, 256],
    "spacing": [3.0, 0.5, 0.5],
    "num_channels": 3,
    "num_classes": 2,
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", type=int, required=True, choices=range(5))
    parser.add_argument("--gc-cases-dir", type=Path, required=True)
    parser.add_argument("--splits-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--combined-dir",
        type=Path,
        default=None,
        help="Optional pooled OOF output directory (one map per case).",
    )
    parser.add_argument("--model-type", type=str, default="unet")
    return parser.parse_args()


def load_subject_list(splits_dir: Path, fold: int) -> list[str]:
    split_file = splits_dir / f"ds-config-valid-fold-{fold}.json"
    with split_file.open() as fp:
        return json.load(fp)["subject_list"]


def load_model(weights_dir: Path, fold: int, model_type: str, device: str):
    checkpoint_path = weights_dir / f"{model_type}_F{fold}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    args = get_default_hyperparams({"model_type": model_type, **IMG_SPEC})
    model = neural_network_for_run(args=args, device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded {checkpoint_path}")
    return model


def case_image_paths(case_dir: Path) -> list[Path]:
    paths = []
    for rel_dir in IMAGE_DIRS:
        matches = sorted((case_dir / rel_dir).glob("*.mha"))
        if not matches:
            raise FileNotFoundError(f"No .mha files in {case_dir / rel_dir}")
        paths.append(matches[0])
    return paths


def predict_case(model, image_paths: list[Path], device: str) -> sitk.Image:
    sample = Sample(
        scans=[sitk.ReadImage(str(path)) for path in image_paths],
        settings=PreprocessingSettings(
            matrix_size=IMG_SPEC["image_shape"],
            spacing=IMG_SPEC["spacing"],
        ),
    )
    sample.preprocess()
    cropped_img = [sitk.GetArrayFromImage(x) for x in sample.scans]
    preproc_img = [
        z_score_norm(np.expand_dims(x, axis=0), percentile=99.5) for x in cropped_img
    ]
    preproc_img = torch.from_numpy(np.expand_dims(np.vstack(preproc_img), axis=0))
    img_for_pred = [preproc_img.to(device), torch.flip(preproc_img, [4]).to(device)]

    with torch.no_grad():
        preds = [
            torch.sigmoid(model(x))[:, 1, ...].detach().cpu().numpy()
            for x in img_for_pred
        ]
    preds[1] = np.flip(preds[1], [3])
    ensemble_output = np.mean(
        [gaussian_filter(x, sigma=1.5) for x in preds],
        axis=0,
    )[0].astype("float32")

    sitk_img = [sitk.ReadImage(str(path)) for path in image_paths]
    resamp_img = [
        sitk.GetArrayFromImage(resample_img(x, out_spacing=IMG_SPEC["spacing"]))
        for x in sitk_img
    ]

    cspca_det_map_sitk = sitk.GetImageFromArray(
        crop_or_pad(ensemble_output, size=resamp_img[0].shape)
    )
    cspca_det_map_sitk.SetSpacing(list(reversed(IMG_SPEC["spacing"])))
    cspca_det_map_sitk = resample_img(
        cspca_det_map_sitk,
        out_spacing=list(reversed(sitk_img[0].GetSpacing())),
    )

    cspca_det_map_npy = extract_lesion_candidates(
        sitk.GetArrayFromImage(cspca_det_map_sitk),
        threshold="dynamic",
    )[0]
    max_val = float(np.max(cspca_det_map_npy))
    if max_val > 0:
        cspca_det_map_npy[cspca_det_map_npy < (max_val / 5)] = 0

    cspca_det_map_sitk = sitk.GetImageFromArray(
        crop_or_pad(cspca_det_map_npy, size=sitk.GetArrayFromImage(sitk_img[0]).shape)
    )
    cspca_det_map_sitk.CopyInformation(sitk_img[0])
    return cspca_det_map_sitk, max_val


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Fold {args.fold} on {device}")

    fold_output_dir = args.output_dir / f"fold_{args.fold}"
    fold_output_dir.mkdir(parents=True, exist_ok=True)
    if args.combined_dir is not None:
        args.combined_dir.mkdir(parents=True, exist_ok=True)

    subject_list = load_subject_list(args.splits_dir, args.fold)
    model = load_model(args.weights_dir, args.fold, args.model_type, device)

    processed = 0
    skipped = 0
    for case_id in subject_list:
        det_map_path = fold_output_dir / f"{case_id}_detection_map.mha"
        if det_map_path.exists():
            skipped += 1
            continue

        case_dir = args.gc_cases_dir / case_id
        if not case_dir.is_dir():
            raise FileNotFoundError(f"Missing GC case directory: {case_dir}")

        image_paths = case_image_paths(case_dir)
        det_map, case_score = predict_case(model, image_paths, device)
        atomic_image_write(det_map, det_map_path)

        score_path = fold_output_dir / f"{case_id}_case_level_likelihood.json"
        score_path.write_text(json.dumps(case_score) + "\n")

        if args.combined_dir is not None:
            combined_path = args.combined_dir / f"{case_id}_detection_map.mha"
            atomic_image_write(det_map, combined_path)

        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed}/{len(subject_list)} cases (skipped {skipped})")

    print(
        f"Done fold {args.fold}: processed={processed}, skipped={skipped}, "
        f"total={len(subject_list)}"
    )


if __name__ == "__main__":
    main()
