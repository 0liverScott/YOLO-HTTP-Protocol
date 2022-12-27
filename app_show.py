import datetime
import os
import socket
from flask import request, Flask

app = Flask(__name__)

def history_clear(paths, amount):
    for path in paths:
        files = os.listdir(path)
        files.sort()
        while len(files) > amount:
            os.remove(f'{path}/{files[0]}')
            files.remove(files[0])

@app.route('/', methods=['POST'])
def tinker():
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    SERVER_IP = '192.168.50.162'
    SERVER_PORT = 28282
    SERVER_ADDR = (SERVER_IP, SERVER_PORT)
    history = 99999

    files = request.files.getlist("file")

    rgb_path = f"files/show/rgb"
    ir_path = f"files/show/ir"
    debug_path = f"files/show/debug"

    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(ir_path, exist_ok=True)

    for file in files:
        if file.filename == 'msx.jpg':
            file.save(os.path.join(ir_path, f'{date}.jpg'))
            print(file.filename, f'saved in {ir_path}/{date}.jpg')
        else:
            file.save(os.path.join(rgb_path, f'{date}.jpg'))
            print(file.filename, f'saved in {rgb_path}/{date}.jpg')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(SERVER_ADDR)
        client_socket.send(date.encode())

    history_clear((rgb_path, ir_path, debug_path), history)

    return "done"


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)  # Fowarded Address = http://1.209.33.170:22170/
