# FusionPortable and custom ROS1 datasets

This data-I/O layer converts a ROS1 stereo/IMU bag into a versioned on-disk
format consumed by MAC-VO's `UnifiedStereo` loader. The command defaults to
FusionPortable v1 handheld topics, while all topics remain configurable for
other collected datasets.

## Unified sequence layout

```text
sequence_root/
├── manifest.yaml
├── camera.yaml
├── stereo.csv
├── imu.csv                  # optional
├── groundtruth.csv          # optional, TUM pose convention
└── camera/
    ├── left/*.png
    └── right/*.png
```

`stereo.csv` is the authoritative frame index. All timestamps are signed
64-bit nanoseconds. `camera.yaml` stores rectified pinhole intrinsics, the
metric baseline, and sensor-to-body `T_BS` (`p_body = T_BS @ p_sensor`). This
matches MAC-VO's EuRoC loader and odometry pose conversion; the opposite
wording currently present in `DataLoader/Interface.py` is inconsistent with
the executable code.

Ground-truth poses use `T_world_body` and the columns
`timestamp_ns,tx,ty,tz,qx,qy,qz,qw`. IMU data uses SI units and the columns
`timestamp_ns,ax,ay,az,gx,gy,gz`.

## Convert FusionPortable v1

Run inside a sourced ROS1 Noetic environment that can import `rosbag`:

```bash
python Scripts/Dataset/ros1_bag_to_unified.py \
  --bag /data/20220216_garden_day.bag \
  --output /data/unified/20220216_garden_day \
  --left-calib /calib/20220209_calib/calib/frame_cam00.yaml \
  --right-calib /calib/20220209_calib/calib/frame_cam01.yaml \
  --groundtruth /data/groundtruth/traj/20220216_garden_day.txt \
  --profile fusionportable_handheld
```

Use `--profile fusionportable_vehicle` for the original Apollo vehicle
sequence. For another dataset, use `--profile custom`, provide
`--left-topic`/`--right-topic`, and optionally provide `--imu-topic`.

FusionPortable calibration files contain camera-to-IMU time shifts, but some
released/refined sequences already compensate timing. The converter does not
silently apply this value. Inspect the sequence and pass
`--camera-time-offset-ns` only when required. Its convention is
`aligned_camera_time = ROS_header_time + offset`.

The conversion is transactional: it writes `<output>.incomplete` and renames
it only after successful extraction and consistency checks. Existing output is
never overwritten.

## Validate and run MAC-VO

```bash
python Scripts/Dataset/validate_unified_dataset.py \
  /data/unified/20220216_garden_day --all-images

python MACVO.py \
  --data Config/Sequence/FusionPortable.yaml \
  --odom Config/Experiment/MACVO/MACVO_Fast.yaml \
  --useRR
```

Edit the dataset root in `Config/Sequence/FusionPortable.yaml` first.

`type: UnifiedStereo` is the appropriate input for MAC-VO's visual odometry
path. `type: UnifiedStereoIMU` is also registered for downstream consumers
that explicitly operate on `StereoInertialFrame`.

## Dependencies

- Conversion: ROS1 `rosbag`, NumPy, OpenCV, PyYAML
- Loading: MAC-VO's existing PyTorch, PyPose, OpenCV, PyYAML environment
- Tests: Python standard-library `unittest`

