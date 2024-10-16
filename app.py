import sqlite3
from flask import Flask, jsonify, render_template, request, redirect, session, url_for
from werkzeug.security import generate_password_hash, check_password_hash
from roboflow import Roboflow
import json
import supervision as sv
import os
import uuid
from inference_sdk import InferenceHTTPClient, InferenceConfiguration
import cv2
import pandas as pd
from joblib import load
from io import BytesIO
import base64
from PIL import Image

# Load your dataset
df = pd.read_csv("dataset/updated_skincare_products.csv")

app = Flask(__name__)
app.secret_key = '4545'
loaded_model = load("model/final_model.h5")
rf_skin = Roboflow(api_key="8RSJzoEweFB7NxxNK6fg")
project_skin = rf_skin.workspace().project("skin-detection-pfmbg")
model_skin = project_skin.version(2).model
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="Gqf1hrF7jdAh8EsbOoTM"
)

class_mapping = {
    "Jenis Kulit Wajah - v6 2023-06-17 11-53am": "oily skin",
    "-": "normal/dry skin"
}

# Function to recommend products based on skin conditions
def recommend_products_based_on_classes(classes):
    recommendations = []
    df_columns_lower = [column.lower() for column in df.columns]
    for skin_condition in classes:
        skin_condition_lower = skin_condition.lower()
        if skin_condition_lower in df_columns_lower:
            original_column = df.columns[df_columns_lower.index(skin_condition_lower)]
            filtered_products = df[df[original_column] == 1][['Brand', 'Name', 'Price', 'Ingredients']]
            filtered_products['Ingredients'] = filtered_products['Ingredients'].apply(lambda x: ', '.join(x.split(', ')[:5]))
            products_list = filtered_products.head(3).to_dict(orient='records')  # Limit to top 3 recommendations
            recommendations.append((skin_condition, products_list))
    return recommendations
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    image_data = data.get('image')

    if image_data:
            # Decoding the image and saving it
            image_data = image_data.split(",")[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(BytesIO(image_bytes))
            image_filename = str(uuid.uuid4()) + '.jpg'
            image_path = os.path.join('static', image_filename)
            image.save(image_path)

            print(f"Image saved at: {image_path}")

            # Run prediction using the saved image
            skin_result = model_skin.predict(image_path, confidence=15, overlap=30).json()
            print("Skin result:", skin_result)

            # Process the predictions
            skin_labels = {item["class"] for item in skin_result["predictions"]}
            print("Skin labels:", skin_labels)

            # Oiliness detection
            custom_configuration = InferenceConfiguration(confidence_threshold=0.3)
            with CLIENT.use_configuration(custom_configuration):
                oilyness_result = CLIENT.infer(image_path, model_id="oilyness-detection-kgsxz/1")

            print("Oiliness result:", oilyness_result)

            # Continue with the rest of your logic...

                
            if not oilyness_result['predictions']:
                skin_labels.add("dryness")
            else:
                oilyness_classes = {class_mapping.get(prediction['class'], prediction['class']) for prediction in oilyness_result['predictions'] if prediction['confidence'] >= 0.3}
                skin_labels.update(oilyness_classes)

            # Annotate image with detected labels
            image = cv2.imread(image_path)
            detections = sv.Detections.from_inference(skin_result)
            label_annotator = sv.LabelAnnotator()
            bounding_box_annotator = sv.BoxAnnotator()
            annotated_image = bounding_box_annotator.annotate(scene=image, detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections)

            # Save annotated image
            annotated_image_path = os.path.join('static', 'annotations_0.jpg')
            cv2.imwrite(annotated_image_path, annotated_image)

            # Get product recommendations based on detected skin conditions
            recommended_products = recommend_products_based_on_classes(skin_labels)

            # Prepare prediction data for the template
            prediction_data = {
                'classes': list(skin_labels),
                'recommendations': recommended_products,
                'annotated_image': f'/static/annotations_0.jpg'
            }

            return jsonify(prediction_data), 200

    return jsonify({}), 400

@app.route('/analysis')
def analysis():
    classes = request.args.get('classes', '[]')
    recommendations = request.args.get('recommendations', '[]')
    annotated_image = request.args.get('annotated_image', '')

    return render_template(
        'analysis.html',
        classes=json.loads(classes),
        recommendations=json.loads(recommendations),
        annotated_image=annotated_image
    )


@app.route('/scan')
def scan():
        return render_template('scan.html') 
@app.route('/', methods=['GET'])
def home():
        return render_template('index.html') 
@app.route('/instruction')
def instruction():
        return render_template('instruction.html') 
@app.route('/camera')
def camera():
        return render_template('camera.html') 
@app.route('/survey1')
def survey1():
        return render_template('Q1.html') 
@app.route('/survey2')
def survey2():
        return render_template('Q2.html') 
@app.route('/survey3')
def survey3():
        return render_template('Q3.html') 
@app.route('/survey4')
def survey4():
        return render_template('Q4.html') 
@app.route('/ingrerec')
def ingrerec():
        return render_template('ingredientrec.html') 
@app.route('/productrec')
def productrec():
        return render_template('productrec.html') 
@app.route('/allproduct')
def allproduct():
        return render_template('Product.html') 
@app.route('/contact')
def contact():
        return render_template('contact.html') 

if __name__ == '__main__':
    app.run(debug=True)
