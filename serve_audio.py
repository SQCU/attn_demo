# serve_audio.py
import http.server
import socketserver
import os
import yaml

# Load config to find the audio directory and port
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

PORT = config['encodec_service']['audio_server_port']
DIRECTORY = config['encodec_service']['output']['audio_dir']

class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

print(f"Serving audio from '{DIRECTORY}' on port {PORT}")
print("To access from your local machine, use http://YOUR_SERVER_IP:{PORT}")
with socketserver.TCPServer(("", PORT), Handler) as httpd:
    httpd.serve_forever()