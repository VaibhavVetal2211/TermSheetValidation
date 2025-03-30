from flask import Flask, request, jsonify
import logging
from app.model import Model  # Import the consolidated model
from werkzeug.utils import secure_filename
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the model
model = Model()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'jpg', 'jpeg', 'png', 'xls', 'xlsx', 'csv', 'txt'}

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def create_app():
    logger.info("Creating Flask app...")
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = 'uploads'
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

    @app.route('/health', methods=['GET'])
    def health_check():
        """Health check endpoint."""
        logger.info("Health check endpoint called.")
        return {"status": "healthy"}, 200

    @app.route('/process', methods=['POST'])
    def process_document():
        """Process a document by accepting various file formats."""
        logger.info("Process document endpoint called.")

        # Check if a file is included in the request
        if 'file' not in request.files:
            logger.warning("No file provided.")
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['file']

        # Check if the file has a valid name and extension
        if file.filename == '':
            logger.warning("No file selected.")
            return jsonify({'error': 'No file selected'}), 400

        if not allowed_file(file.filename):
            logger.warning("File type not allowed.")
            return jsonify({'error': 'File type not allowed'}), 400

        # Save the file to the uploads folder
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        logger.info(f"File saved at {file_path}.")

        try:
            # Process the file using the model
            logger.info("Processing the file with the model.")
            result = model.process_document(file_path)
            logger.info("File processed successfully.")
            return jsonify(result), 200
        except Exception as e:
            logger.error(f"Error during document processing: {e}")
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up: Remove the uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"File {file_path} removed.")

    return app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True)