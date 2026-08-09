#!/usr/bin/env python3
"""Convert a ROS1 stereo/IMU bag to the MAC-VO unified sequence format.

Designed for ROS1 Noetic/Python 3.8+. The only ROS dependency is `rosbag`;
standard Image, CompressedImage, and Imu messages are decoded directly.
"""

import argparse
import csv
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import yaml


FORMAT_VERSION = 1
TOPIC_PROFILES = {
    "fusionportable_handheld": {
        "left": "/stereo/frame_left/image_raw/compressed",
        "right": "/stereo/frame_right/image_raw/compressed",
        "imu": "/stim300/imu/data_raw",
    },
    "fusionportable_vehicle": {
        "left": "/stereo/vehicle_frame_left/image_raw/compressed",
        "right": "/stereo/vehicle_frame_right/image_raw/compressed",
        "imu": "/stim300/imu/data_raw",
    },
    "custom": {"left": None, "right": None, "imu": None},
}


def load_yaml(path: Path) -> Dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    # OpenCV YAML often begins with a non-standard `%YAML:1.0` directive.
    if text.startswith("%YAML:"):
        text = "\n".join(text.splitlines()[1:])
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError("Expected a YAML mapping in {}".format(path))
    return data


def matrix_field(data: Dict[str, Any], key: str, shape: Tuple[int, ...]) -> np.ndarray:
    value = data.get(key)
    if isinstance(value, dict):
        value = value.get("data")
    if value is None:
        raise KeyError("Missing calibration field '{}'".format(key))
    arr = np.asarray(value, dtype=np.float64)
    expected = int(np.prod(shape))
    if arr.size != expected:
        raise ValueError("{} has {} elements; expected {}".format(key, arr.size, expected))
    return arr.reshape(shape)


def quaternion_wxyz_to_rotation(q: Sequence[float]) -> np.ndarray:
    q_arr = np.asarray(q, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q_arr)
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Invalid zero/non-finite quaternion")
    w, x, y, z = q_arr / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def fusionportable_t_bs(left_calib: Dict[str, Any]) -> Tuple[np.ndarray, bool]:
    """Return MAC-VO T_BS from FusionPortable's sensor-to-body calibration.

    FusionPortable describes q/t as sensor(reference) to body_imu(target), i.e.
    p_body = T_BS p_sensor. MAC-VO's EuRoC loader and odometry-output
    conjugation use this same convention, despite a contradictory comment in
    DataLoader/Interface.py.
    """
    q_entry = left_calib.get("quaternion_sensor_body_imu")
    t_entry = left_calib.get("translation_sensor_body_imu")
    if q_entry is None or t_entry is None:
        return np.eye(4, dtype=np.float64), False
    q = q_entry.get("data") if isinstance(q_entry, dict) else q_entry
    t = t_entry.get("data") if isinstance(t_entry, dict) else t_entry
    t_bs = np.eye(4, dtype=np.float64)
    t_bs[:3, :3] = quaternion_wxyz_to_rotation(q)
    t_bs[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return t_bs, True


class StereoCalibration:
    def __init__(self, left_path: Path, right_path: Path, rectify: bool) -> None:
        self.left_raw = load_yaml(left_path)
        self.right_raw = load_yaml(right_path)
        self.width = int(self.left_raw["image_width"])
        self.height = int(self.left_raw["image_height"])
        if (self.width, self.height) != (
            int(self.right_raw["image_width"]), int(self.right_raw["image_height"])
        ):
            raise ValueError("Left/right calibration image sizes differ")

        self.k_left_raw = matrix_field(self.left_raw, "camera_matrix", (3, 3))
        self.k_right_raw = matrix_field(self.right_raw, "camera_matrix", (3, 3))
        self.d_left = matrix_field(self.left_raw, "distortion_coefficients", (1, 5))
        self.d_right = matrix_field(self.right_raw, "distortion_coefficients", (1, 5))
        self.r_left = matrix_field(self.left_raw, "rectification_matrix", (3, 3))
        self.r_right = matrix_field(self.right_raw, "rectification_matrix", (3, 3))
        self.p_left = matrix_field(self.left_raw, "projection_matrix", (3, 4))
        self.p_right = matrix_field(self.right_raw, "projection_matrix", (3, 4))
        self.rectify = rectify

        if rectify:
            self.k = self.p_left[:, :3].copy()
            self.map_left = cv2.initUndistortRectifyMap(
                self.k_left_raw, self.d_left, self.r_left, self.k,
                (self.width, self.height), cv2.CV_32FC1,
            )
            self.map_right = cv2.initUndistortRectifyMap(
                self.k_right_raw, self.d_right, self.r_right, self.p_right[:, :3],
                (self.width, self.height), cv2.CV_32FC1,
            )
        else:
            self.k = self.k_left_raw.copy()
            self.map_left = None
            self.map_right = None

        c_left_x = -self.p_left[0, 3] / self.p_left[0, 0]
        c_right_x = -self.p_right[0, 3] / self.p_right[0, 0]
        self.baseline_m = float(abs(c_right_x - c_left_x))
        if not np.isfinite(self.baseline_m) or self.baseline_m <= 0:
            raise ValueError("Derived stereo baseline is not positive")
        self.t_bs, self.has_t_bs = fusionportable_t_bs(self.left_raw)

    def process(self, image: np.ndarray, side: str) -> np.ndarray:
        if image.shape[:2] != (self.height, self.width):
            raise ValueError(
                "{} image is {}x{}, calibration expects {}x{}".format(
                    side, image.shape[1], image.shape[0], self.width, self.height
                )
            )
        if not self.rectify:
            return image
        maps = self.map_left if side == "left" else self.map_right
        assert maps is not None
        return cv2.remap(image, maps[0], maps[1], cv2.INTER_LINEAR)


def ros_time_ns(stamp: Any) -> int:
    if hasattr(stamp, "to_nsec"):
        return int(stamp.to_nsec())
    sec = getattr(stamp, "secs", getattr(stamp, "sec", None))
    nsec = getattr(stamp, "nsecs", getattr(stamp, "nanosec", None))
    if sec is None or nsec is None:
        raise TypeError("Unsupported ROS time object: {}".format(type(stamp)))
    return int(sec) * 1_000_000_000 + int(nsec)


def message_time_ns(msg: Any, bag_time: Any) -> int:
    header = getattr(msg, "header", None)
    if header is not None and hasattr(header, "stamp"):
        value = ros_time_ns(header.stamp)
        if value > 0:
            return value
    return ros_time_ns(bag_time)


def decode_image(msg: Any) -> np.ndarray:
    msg_type = getattr(msg, "_type", "")
    if msg_type.endswith("CompressedImage") or not hasattr(msg, "encoding"):
        encoded = np.frombuffer(msg.data, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_UNCHANGED)
        if image is None:
            raise ValueError("OpenCV failed to decode CompressedImage")
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        return image

    encoding = str(msg.encoding).lower()
    spec = {
        "bgr8": (np.uint8, 3), "rgb8": (np.uint8, 3),
        "bgra8": (np.uint8, 4), "rgba8": (np.uint8, 4),
        "mono8": (np.uint8, 1), "8uc1": (np.uint8, 1),
        "mono16": (np.uint16, 1), "16uc1": (np.uint16, 1),
    }.get(encoding)
    if spec is None:
        raise ValueError("Unsupported sensor_msgs/Image encoding '{}'".format(encoding))
    dtype, channels = spec
    itemsize = np.dtype(dtype).itemsize
    row_values = int(msg.step) // itemsize
    flat = np.frombuffer(msg.data, dtype=dtype)
    image = flat.reshape(int(msg.height), row_values)
    image = image[:, : int(msg.width) * channels]
    if channels > 1:
        image = image.reshape(int(msg.height), int(msg.width), channels)
    if bool(getattr(msg, "is_bigendian", False)) != (sys.byteorder == "big") and itemsize > 1:
        image = image.byteswap()
    if encoding == "rgb8":
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    elif encoding == "rgba8":
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    elif encoding == "bgra8":
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    elif channels == 1:
        if image.dtype == np.uint16:
            image = (image / 257.0).astype(np.uint8)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return np.ascontiguousarray(image)


def synchronize(
    left: List[Tuple[int, str]], right: List[Tuple[int, str]], tolerance_ns: int
) -> Tuple[List[Tuple[int, int, str, int, str]], int, int]:
    left = sorted(left, key=lambda item: item[0])
    right = sorted(right, key=lambda item: item[0])
    pairs = []
    i = j = 0
    dropped_left = dropped_right = 0
    while i < len(left) and j < len(right):
        lt, lp = left[i]
        rt, rp = right[j]
        delta = rt - lt
        if abs(delta) <= tolerance_ns:
            pairs.append((lt, lt, lp, rt, rp))
            i += 1
            j += 1
        elif delta < -tolerance_ns:
            dropped_right += 1
            j += 1
        else:
            dropped_left += 1
            i += 1
    dropped_left += len(left) - i
    dropped_right += len(right) - j
    return pairs, dropped_left, dropped_right


def copy_groundtruth(source: Path, destination: Path, time_unit: str) -> int:
    rows = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            values = stripped.replace(",", " ").split()
            if len(values) < 8:
                raise ValueError("{}:{} needs 8 TUM columns".format(source, line_number))
            numbers = [float(v) for v in values[:8]]
            ts = numbers[0]
            unit = time_unit
            if unit == "auto":
                unit = "ns" if abs(ts) > 1e12 else "s"
            timestamp_ns = int(round(ts if unit == "ns" else ts * 1e9))
            quat = np.asarray(numbers[4:8], dtype=np.float64)
            norm = float(np.linalg.norm(quat))
            if not math.isfinite(norm) or norm < 1e-12:
                raise ValueError("{}:{} has invalid quaternion".format(source, line_number))
            quat /= norm
            rows.append([timestamp_ns] + numbers[1:4] + quat.tolist())
    rows.sort(key=lambda row: row[0])
    if len(rows) < 2:
        raise ValueError("Ground truth must contain at least two poses")
    if any(a[0] >= b[0] for a, b in zip(rows, rows[1:])):
        raise ValueError("Ground-truth timestamps must be unique and increasing")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["timestamp_ns", "tx", "ty", "tz", "qx", "qy", "qz", "qw"])
        writer.writerows(rows)
    return len(rows)


def available_topics(bag: Any) -> set:
    info = bag.get_type_and_topic_info()
    topics = info.topics if hasattr(info, "topics") else info[1]
    return set(topics.keys())


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--profile", choices=sorted(TOPIC_PROFILES), default="fusionportable_handheld")
    parser.add_argument("--left-topic")
    parser.add_argument("--right-topic")
    parser.add_argument("--imu-topic")
    parser.add_argument("--no-imu", action="store_true")
    parser.add_argument("--left-calib", type=Path, required=True)
    parser.add_argument("--right-calib", type=Path, required=True)
    parser.add_argument("--no-rectify", action="store_true")
    parser.add_argument("--sync-tolerance-ms", type=float, default=1.0)
    parser.add_argument("--min-pair-ratio", type=float, default=0.95)
    parser.add_argument("--camera-time-offset-ns", type=int, default=0)
    parser.add_argument("--right-time-offset-ns", type=int, default=0)
    parser.add_argument("--imu-time-offset-ns", type=int, default=0)
    parser.add_argument("--groundtruth", type=Path)
    parser.add_argument("--groundtruth-time-unit", choices=["auto", "s", "ns"], default="auto")
    parser.add_argument("--gravity", type=float, default=9.81007)
    parser.add_argument("--png-compression", type=int, choices=range(0, 10), default=3)
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    if not args.bag.is_file():
        raise FileNotFoundError(args.bag)
    for path in (args.left_calib, args.right_calib):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.groundtruth is not None and not args.groundtruth.is_file():
        raise FileNotFoundError(args.groundtruth)
    if not 0 < args.min_pair_ratio <= 1:
        raise ValueError("--min-pair-ratio must be in (0, 1]")

    profile = TOPIC_PROFILES[args.profile]
    left_topic = args.left_topic or profile["left"]
    right_topic = args.right_topic or profile["right"]
    imu_topic = None if args.no_imu else (args.imu_topic or profile["imu"])
    if not left_topic or not right_topic:
        raise ValueError("Custom profile requires --left-topic and --right-topic")

    try:
        import rosbag  # type: ignore
    except ImportError as exc:
        raise RuntimeError("Run this converter in a sourced ROS1 environment with python3-rosbag") from exc

    output = args.output.resolve()
    incomplete = output.with_name(output.name + ".incomplete")
    if output.exists() or incomplete.exists():
        raise FileExistsError("Refusing to overwrite {} or {}".format(output, incomplete))
    incomplete.mkdir(parents=True)
    left_dir = incomplete / "camera" / "left"
    right_dir = incomplete / "camera" / "right"
    left_dir.mkdir(parents=True)
    right_dir.mkdir(parents=True)

    calibration = StereoCalibration(args.left_calib, args.right_calib, not args.no_rectify)
    if not calibration.has_t_bs:
        print("WARNING: camera calibration has no sensor-to-body extrinsic; using identity T_BS", file=sys.stderr)

    left_entries: List[Tuple[int, str]] = []
    right_entries: List[Tuple[int, str]] = []
    imu_rows: List[List[float]] = []
    seen_left, seen_right = set(), set()
    compression = [cv2.IMWRITE_PNG_COMPRESSION, args.png_compression]

    with rosbag.Bag(str(args.bag), "r") as bag:
        present = available_topics(bag)
        required = {left_topic, right_topic}
        if imu_topic:
            required.add(imu_topic)
        missing = sorted(required - present)
        if missing:
            raise KeyError("Bag is missing topics: {}\nAvailable: {}".format(missing, sorted(present)))

        topics = [left_topic, right_topic] + ([imu_topic] if imu_topic else [])
        for topic, msg, bag_time in bag.read_messages(topics=topics):
            stamp = message_time_ns(msg, bag_time)
            if topic == imu_topic:
                imu_rows.append([
                    stamp + args.imu_time_offset_ns,
                    float(msg.linear_acceleration.x), float(msg.linear_acceleration.y),
                    float(msg.linear_acceleration.z), float(msg.angular_velocity.x),
                    float(msg.angular_velocity.y), float(msg.angular_velocity.z),
                ])
                continue

            side = "left" if topic == left_topic else "right"
            offset = args.camera_time_offset_ns + (args.right_time_offset_ns if side == "right" else 0)
            aligned_stamp = stamp + offset
            seen = seen_left if side == "left" else seen_right
            if aligned_stamp in seen:
                raise ValueError("Duplicate {} timestamp {}".format(side, aligned_stamp))
            seen.add(aligned_stamp)
            image = calibration.process(decode_image(msg), side)
            rel_path = "camera/{}/{}.png".format(side, stamp)
            dst = incomplete / rel_path
            if not cv2.imwrite(str(dst), image, compression):
                raise IOError("Failed to write {}".format(dst))
            entry = (aligned_stamp, rel_path)
            (left_entries if side == "left" else right_entries).append(entry)
            total_images = len(left_entries) + len(right_entries)
            if total_images % 500 == 0:
                print("Extracted {} images...".format(total_images), flush=True)

    tolerance_ns = int(round(args.sync_tolerance_ms * 1e6))
    pairs, dropped_left, dropped_right = synchronize(left_entries, right_entries, tolerance_ns)
    if not pairs:
        raise ValueError("No stereo pairs found within {} ns".format(tolerance_ns))
    pair_ratio = len(pairs) / float(min(len(left_entries), len(right_entries)))
    if pair_ratio < args.min_pair_ratio:
        raise ValueError(
            "Stereo pair ratio {:.3f} is below {:.3f}; check topics/time offsets/tolerance".format(
                pair_ratio, args.min_pair_ratio
            )
        )

    with (incomplete / "stereo.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "timestamp_ns", "left_timestamp_ns", "right_timestamp_ns",
            "delta_ns", "left_path", "right_path",
        ])
        for timestamp, left_ts, left_path, right_ts, right_path in pairs:
            writer.writerow([timestamp, left_ts, right_ts, right_ts - left_ts, left_path, right_path])

    if imu_rows:
        imu_rows.sort(key=lambda row: row[0])
        if any(a[0] >= b[0] for a, b in zip(imu_rows, imu_rows[1:])):
            raise ValueError("IMU timestamps must be unique and increasing")
        with (incomplete / "imu.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["timestamp_ns", "ax", "ay", "az", "gx", "gy", "gz"])
            writer.writerows(imu_rows)

    gt_count = 0
    if args.groundtruth is not None:
        gt_count = copy_groundtruth(args.groundtruth, incomplete / "groundtruth.csv", args.groundtruth_time_unit)

    camera_yaml = {
        "format_version": FORMAT_VERSION,
        "camera_model": "pinhole",
        "rectified": calibration.rectify,
        "image_width": calibration.width,
        "image_height": calibration.height,
        "K": calibration.k.reshape(-1).tolist(),
        "baseline_m": calibration.baseline_m,
        "T_BS": calibration.t_bs.reshape(-1).tolist(),
        "T_BS_convention": "p_body = T_BS @ p_sensor",
        "body_frame": "body_imu",
        "sensor_frame": "left_camera",
    }
    (incomplete / "camera.yaml").write_text(yaml.safe_dump(camera_yaml, sort_keys=False), encoding="utf-8")

    manifest = {
        "format": "macvo_unified_stereo",
        "format_version": FORMAT_VERSION,
        "source": {"type": "ros1_bag", "bag": args.bag.name, "profile": args.profile},
        "topics": {"left": left_topic, "right": right_topic, "imu": imu_topic},
        "time_unit": "nanoseconds",
        "time_offsets_ns": {
            "left_camera": args.camera_time_offset_ns,
            "right_camera": args.camera_time_offset_ns + args.right_time_offset_ns,
            "imu": args.imu_time_offset_ns,
        },
        "stereo": {
            "frames": len(pairs), "left_messages": len(left_entries),
            "right_messages": len(right_entries), "dropped_left": dropped_left,
            "dropped_right": dropped_right, "sync_tolerance_ns": tolerance_ns,
            "max_abs_delta_ns": max(abs(pair[3] - pair[1]) for pair in pairs),
            "rectified": calibration.rectify,
        },
        "imu": {"present": bool(imu_rows), "samples": len(imu_rows), "gravity_m_s2": args.gravity},
        "groundtruth": {"present": gt_count > 0, "poses": gt_count, "pose": "T_world_body"},
        "calibration_sources": {"left": str(args.left_calib), "right": str(args.right_calib)},
    }
    (incomplete / "manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")

    os.replace(str(incomplete), str(output))
    print(
        "Wrote {} stereo frames, {} IMU samples, {} GT poses to {}".format(
            len(pairs), len(imu_rows), gt_count, output
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise
