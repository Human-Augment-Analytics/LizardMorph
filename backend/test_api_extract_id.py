import urllib.request
import urllib.parse
import json
import os
import mimetypes

# Configuration
API_URL = "http://localhost:3005"
IMAGE_PATH = "sample_image/0003_dorsal.jpg"  # Adjust if needed

def post_json(url, data, headers={}):
    req = urllib.request.Request(url, data=json.dumps(data).encode('utf-8'), headers={'Content-Type': 'application/json', **headers})
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None

def upload_image(url, filepath, headers={}):
    boundary = '---boundary'
    body = []
    
    filename = os.path.basename(filepath)
    mime_type = mimetypes.guess_type(filepath)[0] or 'application/octet-stream'
    
    # Image field
    body.append(f'--{boundary}'.encode())
    body.append(f'Content-Disposition: form-data; name="image"; filename="{filename}"'.encode())
    body.append(f'Content-Type: {mime_type}'.encode())
    body.append(b'')
    with open(filepath, 'rb') as f:
        body.append(f.read())
    
    # View type field
    body.append(f'--{boundary}'.encode())
    body.append(b'Content-Disposition: form-data; name="view_type"')
    body.append(b'')
    body.append(b'dorsal')
    
    body.append(f'--{boundary}--'.encode())
    body.append(b'')
    
    body_bytes = b'\r\n'.join(body)
    
    req = urllib.request.Request(url, data=body_bytes, headers={
        'Content-Type': f'multipart/form-data; boundary={boundary}',
        **headers
    })
    
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        print(f"HTTP Error {e.code}: {e.read().decode()}")
        return None

def post_form(url, data, headers={}):
    data_encoded = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(url, data=data_encoded, headers=headers)
    try:
        with urllib.request.urlopen(req) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
         print(f"HTTP Error {e.code}: {e.read().decode()}")
         return None

def test_extract_id():
    # 1. Start Session
    print("Starting session...")
    resp = post_json(f"{API_URL}/session/start", {})
    if not resp:
        print("Failed to start session")
        return
        
    print(resp)
    session_id = resp.get("session_id")
    headers = {"X-Session-ID": session_id}
    
    # 2. Upload Image
    image_path = IMAGE_PATH
    print(f"Uploading image {image_path}...")
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        # Try finding absolute path
        abs_path = os.path.abspath(image_path)
        if os.path.exists(abs_path):
             image_path = abs_path
        else:
             print("Cannot find image, skipping upload.")
             return

    resp = upload_image(f"{API_URL}/data", image_path, headers=headers)
    print(resp)
    
    if not resp:
         print("Failed to upload image")
         return
         
    # 3. Extract ID
    print("Extracting ID...")
    data = {"image_filename": os.path.basename(image_path)}
    resp = post_form(f"{API_URL}/extract_id", data, headers=headers)
    
    print("Response:")
    print(resp)

if __name__ == "__main__":
    test_extract_id()
