import os
import re
import json
import time
import warnings
import threading
from pathlib import Path
import cv2
import numpy as np

try:
    from tf_keras.optimizers import Adam
    from tf_keras.utils import to_categorical
    from tf_keras.callbacks import LambdaCallback
except ImportError:
    from keras.optimizers import Adam
    from keras.utils import to_categorical
    from keras.callbacks import LambdaCallback

from Model_architecture.modelArch import DenseArchs
from embedding import emb
from ai_engine import face_detector, align_face_crop, assess_face_quality, prototype_matcher

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
PEOPLE_DIR = BASE_DIR / "people"
MODEL_DIR = BASE_DIR / "Model"
LABELS_PATH = MODEL_DIR / "labels.json"
MODEL_PATH = MODEL_DIR / "Face_recognition.MODEL"
MODEL_H5_PATH = MODEL_DIR / "Face_recognition.h5"

# Global training state for web UI polling
TRAINING_STATE = {
    "is_training": False,
    "progress": 0,
    "current_epoch": 0,
    "total_epochs": 30,
    "status": "Idle",
    "loss": 0.0,
    "accuracy": 0.0,
    "error": None,
    "classes": []
}

_TRAIN_LOCK = threading.Lock()


def parse_student_folder(folder_name):
    """
    Parses folder name into (roll_number, student_name).
    Supports all naming conventions: 'CSE-23_Ayush_Pandey',
    '2510001_ABHINAV_CHANDRA_PRAJAPATI', 'TEST_TT_101_Aarav_Sharma', etc.
    """
    try:
        from database_manager import db_manager
        students = db_manager.get_all_students()
        for s in students:
            roll = str(s.get("roll_number", "")).strip()
            if roll and (folder_name == roll or folder_name.startswith(f"{roll}_")):
                return roll, s.get("name", roll)
    except Exception:
        pass

    if "_" in folder_name:
        parts = folder_name.split("_", 1)
        return parts[0].strip(), parts[1].replace("_", " ").strip()
    return folder_name, folder_name


def extract_facial_feature_embeddings(img_bgr, e_model, augment=True):
    """
    Extracts high-precision 128-D FaceNet embeddings strictly from facial features.
    Performs multi-face detection, 5-point affine landmark alignment to canonical (160, 160),
    and subtle identity-preserving data augmentation.
    """
    if img_bgr is None or img_bgr.size == 0:
        return []

    faces = face_detector.detect(img_bgr)
    if faces and len(faces) > 0:
        primary = faces[0]
        box = primary["bbox"]
        lmks = primary.get("landmarks")
        aligned = align_face_crop(img_bgr, box, lmks, output_size=(160, 160))
    else:
        # Fallback to center-focused square crop if already tightly cropped
        h, w = img_bgr.shape[:2]
        side = min(h, w)
        cx, cy = w // 2, h // 2
        crop = img_bgr[max(0, cy - side // 2):cy + side // 2, max(0, cx - side // 2):cx + side // 2]
        aligned = cv2.resize(crop, (160, 160))

    if aligned is None or aligned.size == 0:
        return []

    embeddings = []
    # 1. Base aligned facial features
    prep = e_model.preprocess_face(aligned)
    if prep is not None:
        v = e_model.calculate(prep)
        v = v / max(1e-9, np.linalg.norm(v))
        embeddings.append(v)

    if augment:
        # 2. Horizontal flip (pose invariance)
        flipped = cv2.flip(aligned, 1)
        p_flip = e_model.preprocess_face(flipped)
        if p_flip is not None:
            v_flip = e_model.calculate(p_flip)
            v_flip = v_flip / max(1e-9, np.linalg.norm(v_flip))
            embeddings.append(v_flip)

        # 3. Mild brightness scaling (lighting invariance)
        hsv = cv2.cvtColor(aligned, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.08, 0, 255)
        bright = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        p_br = e_model.preprocess_face(bright)
        if p_br is not None:
            v_br = e_model.calculate(p_br)
            v_br = v_br / max(1e-9, np.linalg.norm(v_br))
            embeddings.append(v_br)

    return embeddings


def Model_Training(epochs=30, batch_size=16, learning_rate=0.001, progress_callback=None):
    """
    Trains the AI recognition model on the available facial image dataset:
    1. Extracts high-precision FaceNet embeddings from pure facial feature crops.
    2. Updates exemplar prototypes & calibrated thresholds.
    3. Trains and exports the deep classifier network (Face_recognition.h5) & labels.json.
    4. Automatically hot-reloads live classroom camera inference models.
    """
    global TRAINING_STATE

    with _TRAIN_LOCK:
        TRAINING_STATE["is_training"] = True
        TRAINING_STATE["status"] = "Scanning student image dataset..."
        TRAINING_STATE["progress"] = 5
        TRAINING_STATE["error"] = None
        TRAINING_STATE["total_epochs"] = epochs

        try:
            if not PEOPLE_DIR.exists():
                raise FileNotFoundError("People directory does not exist. Please enroll students first.")

            folders = sorted([f for f in os.listdir(PEOPLE_DIR) if (PEOPLE_DIR / f).is_dir()])
            valid_folders = []
            for f in folders:
                images = [img for img in os.listdir(PEOPLE_DIR / f) if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if len(images) > 0:
                    valid_folders.append(f)

            if len(valid_folders) == 0:
                raise ValueError("No student image data found in people/ directory.")

            n_classes = len(valid_folders)
            actual_classes = max(2, n_classes)

            # Build label mappings
            class_to_idx = {folder: idx for idx, folder in enumerate(valid_folders)}
            idx_to_label = {}
            idx_to_roll = {}
            idx_to_name = {}

            for folder, idx in class_to_idx.items():
                roll, name = parse_student_folder(folder)
                idx_to_label[str(idx)] = f"{name} ({roll})"
                idx_to_roll[str(idx)] = roll
                idx_to_name[str(idx)] = name

            labels_data = {
                "classes": valid_folders,
                "class_to_idx": class_to_idx,
                "idx_to_label": idx_to_label,
                "idx_to_roll": idx_to_roll,
                "idx_to_name": idx_to_name,
                "n_classes": n_classes
            }

            TRAINING_STATE["classes"] = list(idx_to_label.values())
            TRAINING_STATE["status"] = "Extracting facial feature embeddings (FaceNet 128-D)..."
            TRAINING_STATE["progress"] = 15

            e = emb()
            x_data = []
            y_data = []

            total_images = sum(len(os.listdir(PEOPLE_DIR / f)) for f in valid_folders)
            processed = 0

            # Store per-student vectors for prototype updating
            student_prototype_vectors = {}

            for folder in valid_folders:
                roll, name = parse_student_folder(folder)
                folder_path = PEOPLE_DIR / folder
                class_idx = class_to_idx[folder]
                img_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])

                student_prototype_vectors[roll] = {"name": name, "vectors": []}

                for img_name in img_files:
                    img_path = folder_path / img_name
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # Extract facial feature embeddings (pure facial crop with 5-point alignment)
                    embs_list = extract_facial_feature_embeddings(img, e, augment=True)
                    for vec in embs_list:
                        x_data.append(vec)
                        y_data.append(class_idx)
                        student_prototype_vectors[roll]["vectors"].append(vec)

                    processed += 1
                    if total_images > 0:
                        prog = 15 + int((processed / total_images) * 40)
                        TRAINING_STATE["progress"] = min(prog, 55)

            if len(x_data) == 0:
                raise ValueError("Could not extract any valid face embeddings from student images.")

            # Reset and update prototype matcher with clean, verified facial embeddings
            prototype_matcher.prototypes = {}
            for roll, p_data in student_prototype_vectors.items():
                if p_data["vectors"]:
                    prototype_matcher.add_or_update_student(roll, p_data["name"], p_data["vectors"])
            prototype_matcher.save_prototypes()

            X = np.array(x_data, dtype='float32')
            y_arr = np.array(y_data, dtype='int32')

            # Handle edge case where only 1 student is enrolled
            if n_classes == 1:
                dummy_emb = np.random.normal(0, 0.05, (10, 128)).astype('float32')
                X = np.vstack([X, dummy_emb])
                y_arr = np.concatenate([y_arr, np.full((10,), 1, dtype='int32')])
                idx_to_label["1"] = "Unknown"
                labels_data["idx_to_label"] = idx_to_label

            y = to_categorical(y_arr, num_classes=actual_classes)

            TRAINING_STATE["status"] = "Compiling neural architecture..."
            TRAINING_STATE["progress"] = 60

            arc = DenseArchs(actual_classes)
            face_model = arc.arch()
            optimizer = Adam(learning_rate=learning_rate)
            face_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

            TRAINING_STATE["status"] = "Training deep recognition classifier..."

            class ProgressCallback:
                def __call__(self, epoch, logs):
                    TRAINING_STATE["current_epoch"] = epoch + 1
                    TRAINING_STATE["loss"] = round(float(logs.get("loss", 0)), 4)
                    TRAINING_STATE["accuracy"] = round(float(logs.get("accuracy", 0)), 4)
                    ep_prog = 60 + int(((epoch + 1) / epochs) * 35)
                    TRAINING_STATE["progress"] = min(ep_prog, 95)
                    if progress_callback:
                        progress_callback(TRAINING_STATE)

            cb = LambdaCallback(on_epoch_end=lambda ep, logs: ProgressCallback()(ep, logs))

            face_model.fit(
                X, y,
                batch_size=min(batch_size, len(X)),
                epochs=epochs,
                shuffle=True,
                verbose=0,
                callbacks=[cb]
            )

            MODEL_DIR.mkdir(parents=True, exist_ok=True)
            if MODEL_PATH.exists() and not MODEL_PATH.is_dir():
                MODEL_PATH.unlink()
            face_model.save(str(MODEL_PATH))
            face_model.save(str(MODEL_H5_PATH))

            with open(LABELS_PATH, "w", encoding="utf-8") as f:
                json.dump(labels_data, f, indent=4)

            # Hot reload models in live video engines
            try:
                from classroom_camera import classroom_camera
                classroom_camera.load_models()
            except Exception as e:
                print(f"[ModelTrain] Camera reload notice: {e}")

            TRAINING_STATE["progress"] = 100
            TRAINING_STATE["status"] = f"Training completed successfully ({len(X)} facial samples, {n_classes} students)."
            TRAINING_STATE["is_training"] = False
            print(f"[ModelTrain] {TRAINING_STATE['status']}")
            return True, TRAINING_STATE["status"]

        except Exception as e:
            TRAINING_STATE["is_training"] = False
            TRAINING_STATE["status"] = f"Training failed: {str(e)}"
            TRAINING_STATE["error"] = str(e)
            print(f"[ModelTrain] Error: {e}")
            return False, str(e)


def update_model_on_student_data(roll_number, student_name=None, run_async=True):
    """
    Called whenever a student is registered or new photographs are added.
    Automatically updates biometric prototypes and trains the classifier model.
    """
    def _task():
        print(f"[BiometricUpdate] Auto-updating AI models for student: {roll_number} ({student_name})...")
        Model_Training(epochs=25, batch_size=16)

    if run_async:
        t = threading.Thread(target=_task, daemon=True)
        t.start()
        return t
    else:
        return _task()
