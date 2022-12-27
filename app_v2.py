import datetime
import os
import requests
import socket
import shutil
import time

from flask import request, Flask

app = Flask(__name__)

@app.route('/', methods=['POST'])
def tinker():
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    SERVER_IP = '192.168.50.162'
    SERVER_PORT = 28282
    SERVER_ADDR = (SERVER_IP, SERVER_PORT)
    history_amount = 999

    files = request.files.getlist("file")
    deviceCd = request.values.get("deviceID")

    rgb_path = f"files/{deviceCd}/rgb"
    ir_path = f"files/{deviceCd}/ir"
    path_for_rm = f"files/{deviceCd}"

    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(ir_path, exist_ok=True)

    path = f"files/{deviceCd}/{date}"
    path_for_rm = f"files/{deviceCd}"

    input_path = f"{path}/input"
    output_path = f"{path}/output"

    os.makedirs(input_path, exist_ok=True)
    os.makedirs(output_path, exist_ok=True)

    for file in files:
        if file.filename == 'msx.jpg':
            file.save(os.path.join(ir_path, f'{date}.jpg'))
            print(file.filename, f'saved in {ir_path}/{date}.jpg')
        else:
            file.save(os.path.join(rgb_path, f'{date}.jpg'))
            print(file.filename, f'saved in {rgb_path}/{date}.jpg')

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(SERVER_ADDR)
        client_socket.send(path.encode())

    while len(os.listdir(path_for_rm)) > history_amount:
        f = os.listdir(path_for_rm)
        f.sort()
        shutil.rmtree(rf'{path_for_rm}/{f[0]}')

    return "done"

@app.route('/save', methods=['POST'])
def just_save():
    files = request.files.getlist("file")
    deviceCd = request.values.get("deviceID")
    date = datetime.datetime.now().strftime("%y%m%d_%H%M%S")

    rgb_path = f"files/{deviceCd}/rgb"
    ir_path = f"files/{deviceCd}/ir"

    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(ir_path, exist_ok=True)

    for file in files:
        if file.filename == 'msx.jpg':
            file.save(os.path.join(ir_path, f'{date}.jpg'))
            print(file.filename, f'saved in {ir_path}/{date}.jpg')
        else:
            file.save(os.path.join(rgb_path, f'{date}.jpg'))
            print(file.filename, f'saved in {rgb_path}/{date}.jpg')

    return "done"

# @app.route('/', methods=['POST'])
# def broker():
#     IP = '192.168.0.170'
#     PORT = 29292
#     SIZE = 1024
#     ADDR = (IP, PORT)
#     url_to = 'http://192.168.0.170:5000/'
#
#     with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
#         server_socket.bind(ADDR)
#         server_socket.listen()
#         print('booted')
#         while True:
#             client_socket, client_addr = server_socket.accept()
#             msg = client_socket.recv(SIZE)
#             client_socket.sendall("done".encode())
#             client_socket.close()
#             print({'device_id': device_id, 'event': event, 'date': date})
#             requests.post(url_to, json={'device_id': device_id, 'event': event, 'date': date})


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=False)  # Fowarded Address = http://1.209.33.170:22170/
