from flask import Flask, request, jsonify
from face_service import face_service

app = Flask(__name__)

@app.route("/recognize", methods=["POST"])
def recognize():
    print("[Flask /recognize] Request received.")
    data = request.get_json()
    if not data or "imageBase64" not in data:
        print("[Flask /recognize] Missing imageBase64 in payload.")
        return jsonify({"success": False, "message": "No imageBase64 provided"}), 400
    print(f"[Flask /recognize] imageBase64 length: {len(data['imageBase64'])}")
    result = face_service.recognize(data["imageBase64"])
    print(f"[Flask /recognize] Result: success={result.get('success')} | message={result.get('message')}")
    return jsonify(result)

@app.route("/reload", methods=["POST"])
def reload():
    print("[Flask /reload] Request received.")
    data    = request.get_json() or {}
    records = data.get("records", [])
    print(f"[Flask /reload] Records to reload: {len(records)}")
    for i, r in enumerate(records):
        print(f"[Flask /reload]   [{i+1}] FaceID={r.get('FaceID')} | EmployeeCode={r.get('EmployeeCode')} | ImagePath={r.get('ImagePath')}")
    result = face_service.reload(records)
    print(f"[Flask /reload] Reload result: {result}")
    return jsonify(result)

@app.route("/health", methods=["GET"])
def health():
    print(f"[Flask /health] Loaded faces: {len(face_service.known_embeddings)}")
    return jsonify({"status": "ok", "loaded_faces": len(face_service.known_embeddings)})

if __name__ == "__main__":
    print("[Flask] Starting server on 0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000)