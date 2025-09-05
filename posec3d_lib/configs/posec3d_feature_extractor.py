from collections import defaultdict

# Action recognition info
transformations = [
    defaultdict(
        type="data.pose_transforms.uniform_sample_frames_from_data",
        clip_len=48,
        num_clips=10,
    ),
    defaultdict(type="data.pose_transforms.pose_decode_from_data"),
    defaultdict(
        type="data.pose_transforms.pose_compact_from_data",
        hw_ratio=1.0,
        allow_imgpad=True,
    ),
    defaultdict(
        type="data.processing.resize_from_data",
        scale=(-1, 64),
        interpolation="bilinear",
    ),
    defaultdict(type="data.processing.center_crop_from_data", crop_size=64),
    defaultdict(
        type="data.pose_transforms.generate_pose_target_from_data",
        sigma=0.6,
        use_score=True,
    ),
    defaultdict(
        type="data.formatting.format_shape_from_data", input_format="NCTHW_Heatmap"
    ),
    defaultdict(type="data.formatting.pack_action_inputs_from_data"),
]

ACTION_MODEL_PATH = "posec3d_lib/weights/posec3d_features.onnx"

# Pose estimator info
YOLO_MODEL = "yolov8n-pose.pt"
KEYPOINTS_CONF_THRESHOLD = 0.35
