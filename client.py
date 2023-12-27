# -*- coding: utf-8 -*-
import requests
import json

url = "http://invisiondev.iptime.org:33002/"
files = {'file': ("test.png", open('test.png', 'rb'))}
res = requests.post(url, files=files)

if res.status_code == 200:
    received_data = res.json()
    print(received_data)