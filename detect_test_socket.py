import os
import shutil
import socket
import datetime

SERVER_IP = '192.168.50.162'
SERVER_PORT = 28282
SIZE = 1024
SERVER_ADDR = (SERVER_IP, SERVER_PORT)

path = f'files/socket/{datetime.datetime.now().strftime("%y%m%d_%H%M%S")}'
input_path = f'{path}/input'
output_path = f'{path}/output'
os.makedirs(input_path, exist_ok=True)
os.makedirs(output_path, exist_ok=True)

test_path = 'files/socket/test2/input'

shutil.copy2(f'{test_path}/photo.jpg', input_path)
shutil.copy2(f'{test_path}/msx.jpg', input_path)

for _ in range(1):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client_socket:
        client_socket.connect(SERVER_ADDR)
        client_socket.send(path.encode())
