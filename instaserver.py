from flask import Flask, request, jsonify
from deepface import DeepFace
import requests
from PIL import Image
import io
import os  # Needed for Render's PORT

app = Flask(__name__)

@app.route("/verify", methods=["POST"])
def verify():
    try:
        # Get uploaded file and URL
        image_file = request.files.get("image")
        image_url = request.form.get("image_url")

        # ✅ Corrected check: trigger error only if missing
        if not image_file or not image_url:
            return jsonify({"Error": "Provide one image file and one image url !!!"}), 400

        # Process uploaded image
        img1 = Image.open(image_file.stream)
        image1_path = "temp1.jpg"
        img1.save(image1_path)

        # Fetch image from URL
        response = requests.get(image_url)
        if response.status_code != 200:
            return jsonify({"error": "Could not fetch image from url"}), 400

        img2 = Image.open(io.BytesIO(response.content))
        img2_path = "temp2.jpg"
        img2.save(img2_path)

        # DeepFace verification
        result = DeepFace.verify(
            img1_path=image1_path,
            img2_path=img2_path,
            model_name="ArcFace"
        )

        # Return single word result
        return "true" if result['verified'] else "false"

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use Render's PORT if provided, otherwise fallback to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0",port=port, debug=True)