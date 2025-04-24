import utils
import os
import visual_individual_performance
import xray_preprocessing
from flask import Flask, jsonify, request, send_from_directory, send_file
from flask_cors import CORS, cross_origin
import flask
from base64 import b64encode

app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'upload')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
COLOR_CONTRAST_FOLDER = os.path.join(os.getcwd(), 'color_constrasted')
app.config['COLOR_CONTRAST_FOLDER'] = COLOR_CONTRAST_FOLDER
TPS_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), 'tps_download')
app.config['TPS_DOWNLOAD_FOLDER'] = TPS_DOWNLOAD_FOLDER
IMAGE_DOWNLOAD_FOLDER = os.path.join(os.getcwd(), 'image_download')
app.config['IMAGE_DOWNLOAD_FOLDER'] = IMAGE_DOWNLOAD_FOLDER
INVERT_IMAGE_FOLDER = os.path.join(os.getcwd(), 'invert_image')
app.config['INVERT_IMAGE_FOLDER'] = INVERT_IMAGE_FOLDER


# Create the uploads folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
if not os.path.exists(COLOR_CONTRAST_FOLDER):
    os.makedirs(COLOR_CONTRAST_FOLDER)
if not os.path.exists(TPS_DOWNLOAD_FOLDER):
    os.makedirs(TPS_DOWNLOAD_FOLDER)
if not os.path.exists(IMAGE_DOWNLOAD_FOLDER):
    os.makedirs(IMAGE_DOWNLOAD_FOLDER)
if not os.path.exists(INVERT_IMAGE_FOLDER):
    os.makedirs(INVERT_IMAGE_FOLDER)

@app.route('/data', methods=['POST'])
@cross_origin()
def upload():
    image = request.files.get('image')
    
    if image:
        image_path = os.path.join(f"{app.config['UPLOAD_FOLDER']}", image.filename)
        image.save(image_path)
        xray_preprocessing.process_images(UPLOAD_FOLDER, COLOR_CONTRAST_FOLDER)
        visual_individual_performance.invert_images(UPLOAD_FOLDER, INVERT_IMAGE_FOLDER)
        
    utils.predictions_to_xml("better_predictor_auto.dat", folder=f"{app.config['UPLOAD_FOLDER']}")
    utils.dlib_xml_to_pandas("output.xml")
    utils.dlib_xml_to_tps("output.xml")

    data = visual_individual_performance.parse_xml_for_frontend('output.xml')
    data['path'] = os.path.join(f"{app.config['UPLOAD_FOLDER']}", data['name'])

    return jsonify(data)


# @app.route('/image', methods=['POST'])
# @cross_origin()
# def get_input_image()
#     image_filename = request.args.get('image_filename')

#     if not image_filename:
#         return jsonify({'error': 'image_filename query parameter is required'}), 400


#     image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_filename)

#     if not os.path.exists(image_path):
#         return jsonify({'error': 'Image not found'}), 404

#     return send_from_directory(app.config['UPLOAD_FOLDER'], image_filename, as_attachment=False)

@app.route('/image', methods=['POST'])
@cross_origin()
def get_input_image():
    image_filename = request.args.get('image_filename')
    if not image_filename:
        return jsonify({'error': 'image_filename query parameter is required'}), 400

    image_data = {}
    print(image_filename)


    folders = {
        'color_contrasted': os.path.join(app.config['COLOR_CONTRAST_FOLDER'], f'processed_{image_filename}'),
        'invert_image': os.path.join(app.config['INVERT_IMAGE_FOLDER'], f'inverted_{image_filename}'),
        'image_upload': os.path.join(app.config['UPLOAD_FOLDER'], image_filename)
    }

    for idx, image_path in enumerate(folders.values(), start=1):

        if not os.path.exists(image_path):
            return jsonify({'error': f'Image not found in folder {idx}'}), 404


        with open(image_path, 'rb') as f:
            image_data[f'image{idx}'] = b64encode(f.read()).decode('utf-8')

    return jsonify(image_data)

@app.route('/endpoint', methods=['POST'])
@cross_origin()
def process_scatter_data():
    data = request.json

    coords = data.get('coords')
    name = data.get('name')[:-4]

    if not coords or not name:
        return jsonify({'error': 'Missing required data: coords or name'}), 400

    tps_file_path = os.path.join(TPS_DOWNLOAD_FOLDER, f'{name}.tps')
    print(tps_file_path)

    with open(tps_file_path, 'w', encoding='utf-8', newline='\n') as tps_file:
        tps_file.write(f"LM={len(coords)}\n")
        for point in coords:
            tps_file.write(f"{point['x']} {point['y']}\n")
        tps_file.write(f'IMAGE={name}')
    visual_individual_performance.create_image(tps_file_path, IMAGE_DOWNLOAD_FOLDER)

    return jsonify({'message': 'TPS file generated successfully', 'tps_file': tps_file_path})


if __name__ == '__main__':
    app.run(debug=True)
