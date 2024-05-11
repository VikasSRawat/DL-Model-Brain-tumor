from flask import Flask, request, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
from io import BytesIO

app = Flask(__name__)
CORS(app)

model = load_model('./model1.h5')

# Image size during training
img_size = (224, 224)

def preprocess_image(img):
    img = img.resize(img_size)
    img_array = image.img_to_array(img)

    # Expand the dimensions to match the model input shape (batch_size, height, width, channels)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # rescale=1./255
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Receive the image from the frontend and preprocess it
        image_data = request.files['image'].read()
        image = Image.open(BytesIO(image_data))
        processed_image = preprocess_image(image)

        # Making the prediction
        prediction = model.predict(processed_image)[0]
        # Map the prediction to class names based on the folder
        class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']
        result = {"prediction": class_names[np.argmax(prediction)], "confidence": float(prediction[np.argmax(prediction)])}

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
