from deepface import DeepFace
import numpy as np
import base64
import io
from pathlib import Path
from PIL import Image

FACES_FOLDER = "C:\\Faces"

class FaceRecognitionService:
    def __init__(self):
        self.faces_folder        = Path(FACES_FOLDER)
        self.known_embeddings    = []
        self.known_employee_data = []
        self._warmup_model()
        self.load_from_folder()

    def _warmup_model(self):
        print("[FaceService] Warming up ArcFace model...")
        try:
            dummy = np.zeros((112, 112, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path=dummy,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=False,
                align=False
            )
            print("[FaceService] Model warmup complete.")
        except Exception as e:
            print(f"[FaceService] Warmup error: {e}")

    def load_from_folder(self):
        print(f"[FaceService] Loading fresh from folder: {self.faces_folder}")
        if not self.faces_folder.exists():
            print(f"[FaceService] Faces folder not found: {self.faces_folder}. No embeddings loaded.")
            return
        images = sorted(self.faces_folder.glob("*.jpg"))
        print(f"[FaceService] Found {len(images)} image(s) in folder:")
        for img in images:
            print(f"[FaceService]   File={img.name}")
        self._embed_sources([(img, None, None) for img in images])

    def _embed_sources(self, sources):
        self.known_embeddings    = []
        self.known_employee_data = []
        success_count = 0
        skip_count    = 0

        for image_path, face_id, employee_code in sources:
            image_path = Path(image_path)
            if not image_path.exists():
                print(f"[FaceService] SKIP (missing file): {image_path}")
                skip_count += 1
                continue
            try:
                print(f"[FaceService] Embedding: {image_path.name} | FaceID={face_id} | EmpCode={employee_code}...")
                reps = DeepFace.represent(
                    img_path=str(image_path),
                    model_name="ArcFace",
                    detector_backend="opencv",
                    enforce_detection=True,
                    align=True
                )
                if reps:
                    self.known_embeddings.append(np.array(reps[0]["embedding"]))
                    self.known_employee_data.append({
                        "face_id":       face_id,
                        "employee_code": employee_code,
                        "image_path":    str(image_path),
                        "filename":      image_path.name,
                    })
                    print(f"[FaceService] OK: {image_path.name}")
                    success_count += 1
                else:
                    print(f"[FaceService] SKIP (no representation returned): {image_path.name}")
                    skip_count += 1
            except Exception as e:
                print(f"[FaceService] SKIP (exception): {image_path.name} — {e}")
                skip_count += 1

        print(f"[FaceService] Embedding done. Success={success_count}, Skipped={skip_count}, Total loaded={len(self.known_embeddings)}")
        for i, emp in enumerate(self.known_employee_data):
            print(f"[FaceService]   [{i+1}] File={emp['filename']} | EmpCode={emp['employee_code']} | FaceID={emp['face_id']}")

    def reload(self, db_records=None):
        print(f"[FaceService] reload() called. DB records received: {len(db_records) if db_records else 0}")

        if db_records:
            print("[FaceService] Building embeddings from DB records:")
            sources = []
            for r in db_records:
                image_path = r.get("ImagePath", "")
                face_id    = r.get("FaceID")
                emp_code   = r.get("EmployeeCode")
                exists     = Path(image_path).exists()
                print(f"[FaceService]   FaceID={face_id} | EmpCode={emp_code} | ImagePath={image_path} | Exists={exists}")
                sources.append((Path(image_path), face_id, emp_code))
            self._embed_sources(sources)
        else:
            print("[FaceService] No DB records provided. Loading fresh from folder...")
            self.load_from_folder()

        print(f"[FaceService] Reload complete. Total embeddings: {len(self.known_embeddings)}")
        return {"reloaded": True, "count": len(self.known_embeddings)}

    @staticmethod
    def cosine_similarity(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

    def recognize(self, image_base64, threshold=0.40):
        print("[FaceService] recognize() called.")
        try:
            if image_base64.startswith("data:image"):
                image_base64 = image_base64.split(",")[1]

            print("[FaceService] Decoding base64 image...")
            img_np = np.array(
                Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
            )
            print(f"[FaceService] Image shape: {img_np.shape}")

            print("[FaceService] Running DeepFace.represent on input image...")
            representations = DeepFace.represent(
                img_path=img_np,
                model_name="ArcFace",
                detector_backend="opencv",
                enforce_detection=True,
                align=True,
                normalization="ArcFace"
            )

            if not representations:
                print("[FaceService] No face detected in input image.")
                return {"success": False, "message": "No face detected"}

            print(f"[FaceService] Face detected. Embedding length: {len(representations[0]['embedding'])}")
            selfie_embedding = np.array(representations[0]["embedding"])

            if not self.known_embeddings:
                print("[FaceService] No registered faces loaded in memory.")
                return {"success": False, "message": "No registered faces in database"}

            print(f"[FaceService] Comparing against {len(self.known_embeddings)} stored embeddings...")
            best_match         = None
            highest_similarity = -1

            for idx, stored_embedding in enumerate(self.known_embeddings):
                similarity = self.cosine_similarity(selfie_embedding, stored_embedding)
                emp        = self.known_employee_data[idx]
                print(f"[FaceService]   [{idx+1}] {emp.get('filename')} | EmpCode={emp.get('employee_code')} | Similarity={round(similarity * 100, 2)}%")
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = emp

            confidence = round(float(highest_similarity) * 100, 2)
            print(f"[FaceService] Best match: {best_match.get('filename') if best_match else 'None'} | Confidence: {confidence}% | Threshold: {threshold * 100}%")

            if best_match and highest_similarity >= threshold:
                print(f"[FaceService] MATCH: EmpCode={best_match.get('employee_code')} | FaceID={best_match.get('face_id')} | File={best_match.get('filename')}")
                return {
                    "success":    True,
                    "match":      best_match,
                    "confidence": confidence,
                    "message":    "Face recognized successfully"
                }
            else:
                print(f"[FaceService] NO MATCH. Best confidence {confidence}% did not meet threshold {threshold * 100}%.")
                return {
                    "success":          False,
                    "confidence":       confidence,
                    "message":          f"No matching face found (Best confidence: {confidence}%)",
                    "threshold_used":   threshold,
                    "total_known_faces": len(self.known_embeddings)
                }

        except Exception as e:
            print(f"[FaceService] recognize() exception: {e}")
            return {"success": False, "message": str(e)}


face_service = FaceRecognitionService()