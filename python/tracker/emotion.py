import os

import cv2
import numpy as np

from .config import (
    ENABLE_INDOOR_EMOTION,
    INDOOR_EMOTION_MODEL_PATH,
    INDOOR_EMOTIONS,
    MAX_EMOTION_FRAMES,
    POSE_LANDMARKER_MODEL_PATH,
)


emotion_model = None
pose = None
pad_sequences = None
mediapipe_image_class = None
mediapipe_image_format = None
sequence_buffers = {}
current_emotions = {}


def setup_indoor_emotion():
    global emotion_model, pose, pad_sequences, mediapipe_image_class, mediapipe_image_format

    if not ENABLE_INDOOR_EMOTION:
        return False

    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        import mediapipe as mp
        import tensorflow as tf
        from mediapipe.tasks import python as mp_tasks
        from mediapipe.tasks.python import vision as mp_vision
        from tensorflow.keras.preprocessing.sequence import pad_sequences as keras_pad_sequences

        emotion_model = tf.keras.models.load_model(INDOOR_EMOTION_MODEL_PATH)
        base_options = mp_tasks.BaseOptions(
            model_asset_path=str(POSE_LANDMARKER_MODEL_PATH)
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
            output_segmentation_masks=False,
        )
        pose = mp_vision.PoseLandmarker.create_from_options(options)
        pad_sequences = keras_pad_sequences
        mediapipe_image_class = mp.Image
        mediapipe_image_format = mp.ImageFormat.SRGB
        print(f"屋内感情モデルをロードしました: {INDOOR_EMOTION_MODEL_PATH}")
        print(f"姿勢推定モデルをロードしました: {POSE_LANDMARKER_MODEL_PATH}")
        return True
    except Exception as error:
        print(f"【警告】屋内感情推定を無効化します: {error}")
        return False


# def extract_egait_features(landmarks):
#     mp_indices = [24, 23, 0, 0, 11, 13, 15, 12, 14, 16, 23, 25, 27, 24, 26, 28]
#     features = []
#     for index in mp_indices:
#         landmark = landmarks[index]
#         features.extend([landmark.x, landmark.y, landmark.z])
#     return np.array(features)

def extract_egait_features(landmarks):
    """
    MediaPipeのランドマークからEmotion-Gaitデータセット(メートル単位)に準拠した
    16関節の3D座標を生成し、腰を原点にセンタリングして返す。
    """
    # 各関節の3D座標(x,y,z)を取得するヘルパー関数
    def get_pt(idx):
        return np.array([landmarks[idx].x, landmarks[idx].y, landmarks[idx].z])

    # 1. MediaPipeから必要なキーポイントを取得
    nose = get_pt(0)
    l_sh, r_sh = get_pt(11), get_pt(12)
    l_el, r_el = get_pt(13), get_pt(14)
    l_wri, r_wri = get_pt(15), get_pt(16)
    l_hip, r_hip = get_pt(23), get_pt(24)
    l_knee, r_knee = get_pt(25), get_pt(26)
    l_ank, r_ank = get_pt(27), get_pt(28)

    # 2. Emotion-Gait特有の関節を計算（中間点を取る）
    root = (l_hip + r_hip) / 2.0     # 左右の腰の中間
    neck = (l_sh + r_sh) / 2.0       # 左右の肩の中間
    spine = (root + neck) / 2.0      # RootとNeckの中間（背骨）
    head = nose                      # 鼻の位置をHeadの近似として使用

    # 3. 仕様書の並び順通りに16関節のリストを作成
    joints = [
        root, spine, neck, head,
        l_sh, l_el, l_wri,
        r_sh, r_el, r_wri,
        l_hip, l_knee, l_ank,
        r_hip, r_knee, r_ank
    ]
    
    coords = np.array(joints)

    # 4. 空間の正規化（割り算は不要、センタリングのみ）
    # Rootを厳密な原点 (0, 0, 0) に合わせる
    coords_centered = coords - root

    # 平坦化して (48,) の1次元配列にして返す
    return coords_centered.flatten()


def predict_indoor_emotion(frame, box_bounds, track_id):
    if (
        emotion_model is None
        or pose is None
        or pad_sequences is None
        or mediapipe_image_class is None
        or mediapipe_image_format is None
    ):
        return "happy"

    x_min, y_min, x_max, y_max = box_bounds
    pad = 30
    height, width, _ = frame.shape
    y1, y2 = max(0, y_min - pad), min(height, y_max + pad)
    x1, x2 = max(0, x_min - pad), min(width, x_max + pad)
    crop_img = frame[y1:y2, x1:x2]

    if crop_img.shape[0] == 0 or crop_img.shape[1] == 0:
        return current_emotions.get(track_id, "happy")

    crop_rgb = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
    mp_image = mediapipe_image_class(
        image_format=mediapipe_image_format,
        data=crop_rgb,
    )
    pose_results = pose.detect(mp_image)

    if not pose_results.pose_world_landmarks:
        return current_emotions.get(track_id, "happy")

    features = extract_egait_features(pose_results.pose_world_landmarks[0])
    sequence_buffers.setdefault(track_id, []).append(features)
    if len(sequence_buffers[track_id]) > MAX_EMOTION_FRAMES:
        sequence_buffers[track_id].pop(0)

    if len(sequence_buffers[track_id]) > 30:
        input_data = pad_sequences(
            [sequence_buffers[track_id]],
            maxlen=MAX_EMOTION_FRAMES,
            dtype="float32",
            padding="post",
            truncating="post",
        )
        prediction = emotion_model.predict(input_data, verbose=0)
        emotion_index = int(np.argmax(prediction[0]))
        current_emotions[track_id] = INDOOR_EMOTIONS[emotion_index]

    return current_emotions.get(track_id, "happy")
