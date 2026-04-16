from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create upload folder
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model("mpeg7_model.h5")

# Class labels
class_names = ['apple', 'bell', 'bird', 'bone', 'bottle', 'tree']

# Image preprocessing
def prepare_image(image):
    image = image.convert('L')
    image = image.resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=(0, -1))
    return img_array

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    confidence = None
    filename = None
    image_path = None   # ✅ IMPORTANT

    if request.method == "POST":
        file = request.files.get("file")

        if file and file.filename != "":
            filename = secure_filename(file.filename)

            # Save file
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            image_path = filepath  # ✅ store for HTML

            # Process image
            img = Image.open(filepath)
            processed = prepare_image(img)

            pred = model.predict(processed)
            prediction = class_names[np.argmax(pred)]
            confidence = round(float(np.max(pred)), 3)

    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        filename=filename,
        image_path=image_path   # ✅ ALWAYS PASS
    )

if __name__ == "__main__":
    app.run(debug=True)