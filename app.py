import logging
from flask import Flask, jsonify, request, send_from_directory
from dotenv import load_dotenv
import os
import uuid
import predict
from flask_cors import CORS

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

API_KEY = os.getenv('API_KEY')
BACKEND_PORT = os.getenv('BACKEND_PORT')
DEBUG = os.getenv('DEBUG', 'False').lower() == 'true'

# Configure logging
logging.basicConfig(level=logging.DEBUG if DEBUG else logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Path to store uploaded images
UPLOAD_FOLDER = "uploaded/"
PREDICTED_FOLDER = "./predicted/"

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(PREDICTED_FOLDER):
    os.makedirs(PREDICTED_FOLDER)

# Helper functions for standardized responses
def success_response(data=None, message=""):
    return jsonify({
        'status': 'success',
        'data': data,
        'message': message
    }), 200

def error_response(message, status_code=400):
    return jsonify({
        'status': 'error',
        'message': message
    }), status_code

# Error handling for 404 and 500 errors
@app.errorhandler(404)
def not_found(error):
    return error_response("Resource not found", 404)

@app.errorhandler(500)
def server_error(error):
    return error_response("Internal server error", 500)

# Middleware for API key verification
@app.before_request
def require_api_key():
    api_key = request.headers.get('x-api-key')
    if api_key != API_KEY:
        logging.warning("Unauthorized access attempt")
        return error_response('Unauthorized', 401)
    logging.info("API key verified successfully")

# Endpoint for predicting the image by UUID and returning the binary PNG
@app.route('/predict/<image_uuid>', methods=['GET'])
def run_prediction(image_uuid):
    image_path = os.path.join(UPLOAD_FOLDER, f"{image_uuid}.tif")
    
    if not os.path.exists(image_path):
        logging.error(f"Image with UUID {image_uuid} not found")
        return error_response('Image not found', 404)

    try:
        # Call the predict_full_image function from predict.py
        predict.predict_full_image(image_path)  # No need to pass the model

        # Return the binary prediction image file
        binary_image_filename = f"{image_uuid}_binary_full.png"
        return send_from_directory(PREDICTED_FOLDER, binary_image_filename)
    except Exception as e:
        logging.error(f"Error processing prediction: {e}")
        return error_response(f'Error processing prediction: {str(e)}', 500)

# Endpoint for uploading a user picture
@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        logging.error("No file part in the request")
        return error_response('No file part', 400)
    
    file = request.files['file']
    
    if file.filename == '':
        logging.error("No selected file")
        return error_response('No selected file', 400)
    
    if file:
        try:
            # Generate a unique UUID for the image
            image_uuid = str(uuid.uuid4())
            filename = f"{image_uuid}.tif"  # Save it as .tif
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            # Save the uploaded image
            file.save(file_path)
            logging.info(f"File {file.filename} uploaded successfully as {filename}")

            return success_response({'image_uuid': image_uuid}, 'File uploaded successfully')
        except Exception as e:
            logging.error(f"Error saving file: {e}")
            return error_response(f'Error saving file: {str(e)}', 500)

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(BACKEND_PORT), debug=DEBUG)
