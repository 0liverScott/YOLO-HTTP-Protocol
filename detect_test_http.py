import requests

url = 'http://192.168.50.136:5000'
test_path = 'files/socket/test2/input'

files = [('file', open(f'{test_path}/photo.jpg', 'rb')), ('file', open(f'{test_path}/msx.jpg', 'rb'))]

device = 'test'

requests.post(url, files=files, data={"deviceID": device})

