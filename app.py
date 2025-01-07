from flask import Flask, request, jsonify
import numpy as np
import joblib
from PIL import Image
import base64
import io

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load("model/combined_model.pkl")
scaler = joblib.load("model/combined_scaler.pkl")  # Load the saved scaler

def preprocess_image(base64_image):
    """Convert base64 image to scaled RGB average."""
    # Decode the base64 image
    image_data = base64.b64decode(base64_image)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")

    # Convert image to numpy array and compute average RGB
    pixels = np.array(image)
    avg_rgb = pixels[:, :, :3].mean(axis=(0, 1))  # Average over R, G, B channels

    # Scale the RGB values using the loaded scaler
    scaled_rgb = scaler.transform([avg_rgb])  # Scale the data to match training
    return scaled_rgb

@app.route("/predict", methods=["POST"])
def predict_color():
    """Predict the color based on input image."""
    try:
        # Extract the base64 image from the request
        data = request.get_json()
        base64_image = data.get("image")
        if not base64_image:
            return jsonify({"error": "No image provided"}), 400

        # Preprocess the image
        input_data = preprocess_image(base64_image)

        # Predict the color using the trained model
        predicted_color = model.predict(input_data)[0]
        return jsonify({"predicted_color": predicted_color})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
