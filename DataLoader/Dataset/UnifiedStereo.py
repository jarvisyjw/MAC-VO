from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import cv2
import numpy as np
import pypose as pp
import torch
import yaml

from Utility.Math import interpolate_pose
from ..Interface import IMUData, StereoData, StereoFrame, StereoInertialFrame
from ..SequenceBase import SequenceBase


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a YAML mapping in {path}")
    return data


class UnifiedStereoSequence(SequenceBase[StereoFrame]):
    """MAC-VO loader for the versioned unified stereo format."""

    @classmethod
    def name(cls) -> str:
        return "UnifiedStereo"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        cfg = self.config_dict2ns(config)
        self.root = Path(cfg.root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)

        manifest = _read_yaml(self.root / "manifest.yaml")
        if manifest.get("format") != "macvo_unified_stereo":
            raise ValueError(f"Unsupported dataset format in {self.root / 'manifest.yaml'}")
        if int(manifest.get("format_version", -1)) != 1:
            raise ValueError(f"Unsupported unified format version {manifest.get('format_version')}")

        camera = _read_yaml(self.root / "camera.yaml")
        if not bool(camera.get("rectified", False)) and not bool(getattr(cfg, "allow_unrectified", False)):
            raise ValueError("MAC-VO stereo input should be rectified; set allow_unrectified only for debugging")
        self.width = int(camera["image_width"])
        self.height = int(camera["image_height"])
        self.K = torch.tensor(camera["K"], dtype=torch.float32).reshape(1, 3, 3)
        self.baseline = torch.tensor([float(camera["baseline_m"])], dtype=torch.float32)
        t_bs = torch.tensor(camera["T_BS"], dtype=torch.float32).reshape(1, 4, 4)
        self.T_BS = pp.from_matrix(t_bs, pp.SE3_type)

        self.records: list[dict[str, str]] = []
        with (self.root / "stereo.csv").open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                self.records.append(row)
        if not self.records:
            raise ValueError(f"No frames in {self.root / 'stereo.csv'}")
        self.timestamps = np.asarray([int(row["timestamp_ns"]) for row in self.records], dtype=np.int64)
        if np.any(np.diff(self.timestamps) <= 0):
            raise ValueError("Stereo timestamps are not strictly increasing")

        self.gt_pose: pp.LieTensor | None = None
        if bool(getattr(cfg, "gt_pose", False)):
            gt_path = self.root / "groundtruth.csv"
            if not gt_path.is_file():
                raise FileNotFoundError(f"gt_pose=true but {gt_path} does not exist")
            gt = np.loadtxt(gt_path, delimiter=",", skiprows=1, ndmin=2)
            if gt.shape[1] != 8 or gt.shape[0] < 2:
                raise ValueError("groundtruth.csv must have >=2 rows and 8 columns")
            gt_time = torch.from_numpy(gt[:, 0].astype(np.int64))
            if not bool((gt_time[1:] > gt_time[:-1]).all()):
                raise ValueError("Ground-truth timestamps are not strictly increasing")
            gt_se3 = pp.SE3(torch.from_numpy(gt[:, 1:8]).double())
            interpolated, outside = interpolate_pose(
                gt_se3, gt_time, torch.from_numpy(self.timestamps)
            )
            if bool(outside.any()) and not bool(getattr(cfg, "allow_gt_extrapolation", False)):
                count = int(outside.sum().item())
                raise ValueError(
                    f"{count} camera frames are outside GT time range; clip the sequence "
                    "or set allow_gt_extrapolation=true"
                )
            self.gt_pose = interpolated.float()

        super().__init__(len(self.records))

    def _load_image(self, relative_path: str) -> torch.Tensor:
        path = self.root / relative_path
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Failed to read {path}")
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(
                f"Image {path} is {image.shape[1]}x{image.shape[0]}, "
                f"expected {self.width}x{self.height}"
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)

    def __getitem__(self, local_index: int) -> StereoFrame:
        index = self.get_index(local_index)
        row = self.records[index]
        timestamp = int(row["timestamp_ns"])
        image_left = self._load_image(row["left_path"])
        image_right = self._load_image(row["right_path"])
        return StereoFrame(
            idx=[local_index],
            time_ns=[timestamp],
            gt_pose=None if self.gt_pose is None else cast(pp.LieTensor, self.gt_pose[index:index + 1]),
            stereo=StereoData(
                T_BS=cast(pp.LieTensor, self.T_BS),
                K=self.K,
                baseline=self.baseline,
                width=self.width,
                height=self.height,
                time_ns=[timestamp],
                imageL=image_left,
                imageR=image_right,
            ),
        )

    @classmethod
    def is_valid_config(cls, config: SimpleNamespace | None) -> None:
        cls._enforce_config_spec(config, {
            "root": lambda value: isinstance(value, str),
            "gt_pose": lambda value: isinstance(value, bool),
        }, allow_excessive_cfg=True)


class UnifiedStereoInertialSequence(UnifiedStereoSequence):
    """Same sequence with IMU samples bracketing each image interval."""

    @classmethod
    def name(cls) -> str:
        return "UnifiedStereoIMU"

    def __init__(self, config: SimpleNamespace | dict[str, Any]) -> None:
        super().__init__(config)
        imu_path = self.root / "imu.csv"
        if not imu_path.is_file():
            raise FileNotFoundError(f"UnifiedStereoIMU requires {imu_path}")
        imu = np.loadtxt(imu_path, delimiter=",", skiprows=1, ndmin=2)
        if imu.shape[1] != 7 or imu.shape[0] < 2:
            raise ValueError("imu.csv must have >=2 rows and 7 columns")
        self.imu_time = imu[:, 0].astype(np.int64)
        if np.any(np.diff(self.imu_time) <= 0):
            raise ValueError("IMU timestamps are not strictly increasing")
        self.imu_acc = torch.from_numpy(imu[:, 1:4]).float()
        self.imu_gyro = torch.from_numpy(imu[:, 4:7]).float()
        manifest = _read_yaml(self.root / "manifest.yaml")
        self.gravity = float(manifest.get("imu", {}).get("gravity_m_s2", 9.81007))
        self.T_BS_imu = pp.identity_SE3(1, dtype=torch.float32)

    def _imu_for_frame(self, index: int, previous_index: int) -> IMUData:
        current = int(self.timestamps[index])
        previous = int(self.timestamps[previous_index])
        # Include one sample on either side so integration covers the image times.
        begin = max(int(np.searchsorted(self.imu_time, previous, side="right")) - 1, 0)
        end = min(int(np.searchsorted(self.imu_time, current, side="left")) + 1, len(self.imu_time))
        if end <= begin:
            end = min(begin + 1, len(self.imu_time))
        if begin >= end:
            raise ValueError(f"No IMU sample covers camera frame {index} at {current}")
        time = torch.from_numpy(self.imu_time[begin:end].copy()).reshape(1, -1, 1)
        return IMUData(
            T_BS=cast(pp.LieTensor, self.T_BS_imu),
            time_ns=time,
            gravity=[self.gravity],
            acc=self.imu_acc[begin:end].unsqueeze(0),
            gyro=self.imu_gyro[begin:end].unsqueeze(0),
        )

    def __getitem__(self, local_index: int) -> StereoInertialFrame:
        index = self.get_index(local_index)
        previous_index = self.get_index(local_index - 1) if local_index > 0 else index
        stereo = super().__getitem__(local_index)
        return StereoInertialFrame(
            idx=stereo.idx,
            time_ns=stereo.time_ns,
            gt_pose=stereo.gt_pose,
            stereo=stereo.stereo,
            imu=self._imu_for_frame(index, previous_index),
            gt_attitude=None,
        )
