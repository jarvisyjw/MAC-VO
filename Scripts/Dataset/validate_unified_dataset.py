#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np
import yaml


def read_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Expected YAML mapping in {}".format(path))
    return value


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate(root: Path, check_all_images: bool = False) -> Dict[str, int]:
    root = root.expanduser().resolve()
    manifest = read_yaml(root / "manifest.yaml")
    camera = read_yaml(root / "camera.yaml")
    if manifest.get("format") != "macvo_unified_stereo" or int(manifest.get("format_version", -1)) != 1:
        raise ValueError("Unsupported manifest format/version")

    k = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or abs(k[2, 2] - 1) > 1e-6:
        raise ValueError("Invalid camera intrinsics")
    if float(camera["baseline_m"]) <= 0:
        raise ValueError("Stereo baseline must be positive")
    t_bs = np.asarray(camera["T_BS"], dtype=np.float64).reshape(4, 4)
    if not np.allclose(t_bs[3], [0, 0, 0, 1], atol=1e-7):
        raise ValueError("Invalid homogeneous T_BS bottom row")
    if not np.allclose(t_bs[:3, :3].T @ t_bs[:3, :3], np.eye(3), atol=1e-4):
        raise ValueError("T_BS rotation is not orthonormal")

    stereo = read_csv(root / "stereo.csv")
    if not stereo:
        raise ValueError("stereo.csv is empty")
    timestamps = np.asarray([int(row["timestamp_ns"]) for row in stereo], dtype=np.int64)
    if np.any(np.diff(timestamps) <= 0):
        raise ValueError("Stereo timestamps are not strictly increasing")
    tolerance = int(manifest["stereo"]["sync_tolerance_ns"])
    for row in stereo:
        if abs(int(row["delta_ns"])) > tolerance:
            raise ValueError("Stereo pair exceeds synchronization tolerance")

    indices = range(len(stereo)) if check_all_images else sorted(set([0, len(stereo) // 2, len(stereo) - 1]))
    width, height = int(camera["image_width"]), int(camera["image_height"])
    for index in indices:
        for key in ("left_path", "right_path"):
            path = root / stereo[index][key]
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                raise FileNotFoundError(path)
            if image.shape[:2] != (height, width):
                raise ValueError("{} has unexpected dimensions".format(path))

    imu_count = 0
    imu_path = root / "imu.csv"
    if imu_path.is_file():
        imu = np.loadtxt(imu_path, delimiter=",", skiprows=1, ndmin=2)
        if imu.shape[1] != 7 or not np.isfinite(imu).all():
            raise ValueError("Invalid imu.csv")
        imu_time = imu[:, 0].astype(np.int64)
        if np.any(np.diff(imu_time) <= 0):
            raise ValueError("IMU timestamps are not strictly increasing")
        if imu_time[0] > timestamps[0] or imu_time[-1] < timestamps[-1]:
            raise ValueError("IMU does not cover the complete camera time range")
        imu_count = len(imu)

    gt_count = 0
    gt_path = root / "groundtruth.csv"
    if gt_path.is_file():
        gt = np.loadtxt(gt_path, delimiter=",", skiprows=1, ndmin=2)
        if gt.shape[1] != 8 or not np.isfinite(gt).all():
            raise ValueError("Invalid groundtruth.csv")
        gt_time = gt[:, 0].astype(np.int64)
        if np.any(np.diff(gt_time) <= 0):
            raise ValueError("GT timestamps are not strictly increasing")
        quat_norm = np.linalg.norm(gt[:, 4:8], axis=1)
        if not np.allclose(quat_norm, 1.0, atol=1e-4):
            raise ValueError("GT quaternions are not normalized")
        gt_count = len(gt)

    return {"stereo_frames": len(stereo), "imu_samples": imu_count, "gt_poses": gt_count}


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--all-images", action="store_true")
    args = parser.parse_args(argv)
    stats = validate(args.root, args.all_images)
    print("VALID: {stereo_frames} stereo frames, {imu_samples} IMU samples, {gt_poses} GT poses".format(**stats))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

