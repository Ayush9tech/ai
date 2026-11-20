import os
import json
import warnings
from pathlib import Path
import cv2
import numpy as np
try:
    from tf_keras.models import load_model
except ImportError:
    from keras.models import load_model

from FaceDetection.face_detection import face
from embedding import emb
from database_manager import db_manager
from ai_engine import face_detector, align_face_crop, prototype_matcher

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "Model"
MODEL_PATH = MODEL_DIR / "Face_recognition.MODEL"
MODEL_H5_PATH = MODEL_DIR / "Face_recognition.h5"
LABELS_PATH = MODEL_DIR / "labels.json"


class FaceRecognizer:
    """
    Handles live recognition and attendance marking for both web streaming and desktop OpenCV.
    Uses FaceNet 128-D aligned facial feature embeddings & prototype exemplar matching.
    """

    def __init__(self, confidence_threshold=0.70, consecutive_frames_needed=4):
        self.confidence_threshold = confidence_threshold
        self.consecutive_frames_needed = consecutive_frames_needed
        self.fd = face()
        self.e = emb()
        self.model = None
        self.labels_data = None
        self.session_counts = {}  # Tracks consecutive detection frames: {roll: frame_count}
        self.session_marked = set()  # Roll numbers already marked during current session
        self._load_recognition_model()

    def _load_recognition_model(self):
        prototype_matcher.load_prototypes()
        if LABELS_PATH.exists():
            try:
                with open(LABELS_PATH, "r", encoding="utf-8") as f:
                    self.labels_data = json.load(f)
            except Exception as e:
                print(f"[Recognizer] Error reading labels.json: {e}")

        model_file = None
        if MODEL_H5_PATH.exists():
            model_file = MODEL_H5_PATH
        elif MODEL_PATH.exists():
            model_file = MODEL_PATH

        if model_file:
            try:
                self.model = load_model(str(model_file), compile=False)
                print(f"[Recognizer] Loaded classifier model from {model_file.name}")
            except Exception as e:
                print(f"[Recognizer] Error loading model: {e}")
                self.model = None
        else:
            self.model = None

    def is_model_ready(self):
        return (len(prototype_matcher.prototypes) > 0) or (self.model is not None and self.labels_data is not None)

    def reset_session(self):
        self.session_counts.clear()
        self.session_marked.clear()

    @staticmethod
    def _draw_banner(img, x, y, w, h, title_text, sub_text, color_bgr):
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.50
        thick = 1
        (tw, th), base = cv2.getTextSize(title_text, font, scale, thick)
        pad = 6
        banner_w = max(w, tw + pad * 2)
        banner_h = th + pad * 2 + 2

        if y >= banner_h + 2:
            by1 = y - banner_h
            by2 = y
            ty = y - pad - 2
        else:
            by1 = y
            by2 = y + banner_h
            ty = y + banner_h - pad - 2

        cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, 2)
        cv2.rectangle(img, (x, by1), (x + banner_w, by2), color_bgr, -1)
        cv2.putText(img, title_text, (x + pad + 1, ty + 1), font, scale, (0, 0, 0), thick + 1, cv2.LINE_AA)
        cv2.putText(img, title_text, (x + pad, ty), font, scale, (255, 255, 255), thick, cv2.LINE_AA)

        if sub_text:
            (stw, sth), sbase = cv2.getTextSize(sub_text, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
            sub_w = max(w, stw + pad * 2)
            sub_h = sth + pad * 2
            sy1 = y + h
            sy2 = y + h + sub_h
            cv2.rectangle(img, (x, sy1), (x + sub_w, sy2), (10, 25, 47), -1)
            cv2.putText(img, sub_text, (x + pad, sy1 + sth + pad), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color_bgr, 1, cv2.LINE_AA)

    def process_frame(self, frame, subject="Hindi", auto_mark=True):
        """
        Processes a single BGR frame, detects faces, predicts identity using
        calibrated face embeddings, and marks attendance in the database.
        Returns: (annotated_frame, list_of_detected_faces_info)
        """
        if frame is None or frame.size == 0:
            return frame, []

        subj = subject.strip().capitalize()
        if subj not in ["Hindi", "English"]:
            subj = "Hindi"

        annotated_frame = frame.copy()
        detected = face_detector.detect(frame)
        if not detected:
            # Fallback to OpenCV cascade
            crops, coords = self.fd.detectFace(frame)
        else:
            coords = [f["bbox"] for f in detected]
            crops = []
            for f in detected:
                x, y, w, h = f["bbox"]
                lmks = f.get("landmarks")
                aligned = align_face_crop(frame, [x, y, w, h], lmks, output_size=(160, 160))
                crops.append(aligned)

        detections = []

        if not self.is_model_ready():
            for (x, y, w, h) in coords:
                self._draw_banner(annotated_frame, x, y, w, h, "Model Not Trained", "Please train model", (0, 165, 255))
            return annotated_frame, [{"status": "Model Not Trained"}]

        for i, crop in enumerate(crops):
            if crop is None or crop.size == 0:
                continue

            x, y, w, h = coords[i]
            prep = self.e.preprocess_face(crop)
            if prep is None:
                continue

            emb_feat = self.e.calculate(prep)
            emb_feat = emb_feat / max(1e-9, np.linalg.norm(emb_feat))

            matched_roll, student_name, confidence, recog_state = prototype_matcher.match_embedding(
                emb_feat, eligible_ids=None
            )

            # High-precision fallback to classifier model
            if (recog_state == "UNKNOWN" or not prototype_matcher.prototypes) and self.model is not None and self.labels_data is not None:
                emb_input = np.expand_dims(emb_feat, axis=0)
                predictions = self.model.predict(emb_input, verbose=0)[0]
                best_idx = int(np.argmax(predictions))
                mlp_conf = float(predictions[best_idx])
                str_idx = str(best_idx)
                idx_to_roll = self.labels_data.get("idx_to_roll", {})
                idx_to_name = self.labels_data.get("idx_to_name", {})
                if mlp_conf >= 0.70 and str_idx in idx_to_roll:
                    matched_roll = idx_to_roll[str_idx]
                    student_name = idx_to_name.get(str_idx, f"Student {matched_roll}")
                    confidence = mlp_conf
                    recog_state = "HIGH_CONFIDENCE"

            if recog_state in ["HIGH_CONFIDENCE", "REVIEW_REQUIRED"] and matched_roll != "UNKNOWN":
                student = db_manager.get_student(matched_roll)
                if student:
                    student_name = student.get("name", student_name)

                # Sustained detection count for auto-marking attendance
                cur_count = self.session_counts.get(matched_roll, 0) + 1
                self.session_counts[matched_roll] = cur_count

                already_marked = matched_roll in self.session_marked
                marked_now = False

                if auto_mark and cur_count >= self.consecutive_frames_needed and not already_marked:
                    success, msg, _ = db_manager.mark_attendance(subj, matched_roll, student_name)
                    self.session_marked.add(matched_roll)
                    marked_now = True

                is_present = already_marked or marked_now
                color = (19, 185, 16) if is_present else (8, 179, 234)
                status_text = "Verified Present" if is_present else f"Verifying ({cur_count}/{self.consecutive_frames_needed})"

                display_title = f"{student_name} ({int(confidence * 100)}%)"
                display_sub = f"ID: {matched_roll} • {status_text}"
                self._draw_banner(annotated_frame, x, y, w, h, display_title, display_sub, color)

                detections.append({
                    "roll": matched_roll,
                    "name": student_name,
                    "confidence": round(confidence, 3),
                    "marked": is_present,
                    "status": status_text,
                    "subject": subj
                })
            else:
                self._draw_banner(annotated_frame, x, y, w, h, "UNKNOWN PERSON", "Unregistered • Visitor", (68, 68, 239))
                detections.append({
                    "roll": "UNKNOWN",
                    "name": "UNKNOWN PERSON",
                    "confidence": round(confidence, 3),
                    "marked": False,
                    "status": "Unregistered",
                    "subject": subj
                })

        return annotated_frame, detections
