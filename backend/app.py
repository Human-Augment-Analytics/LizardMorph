import utils
import os
import visual_individual_performance

from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # To allow cross-origin requests from React frontend

# Set the upload folder path
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'test_auto')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/upload', methods=['POST'])
def upload():
    image = request.files.get('image')
    
    if image:
        image_path = os.path.join(f"{app.config['UPLOAD_FOLDER']}", image.filename)
        image.save(image_path)

    utils.predictions_to_xml("better_predictor_auto.dat", folder=f"{app.config['UPLOAD_FOLDER']}")
    utils.dlib_xml_to_pandas("output.xml")
    utils.dlib_xml_to_tps("output.xml")

    output_image_paths = visual_individual_performance.create_image('output.xml', 'outputs')

    return send_file(output_image_paths[0], mimetype='image/png')

@app.route('/output.xml', methods=['GET'])
def get_output_file():
    return send_from_directory('./', 'output.xml')

if __name__ == '__main__':
    app.run(debug=True)
