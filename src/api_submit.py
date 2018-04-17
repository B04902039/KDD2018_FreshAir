
# coding: utf-8

import requests

files={'files': open('../queue/default.csv','rb')}

data = {
    "user_id": "LoChiChen",   #user_id is your username which can be found on the top-right corner on our website when you logged in.
    "team_token": "f62b378aef32f97c0eb7eda878af2fe6bf7e8dc16ab76f2aedfb55522ba15967", #your team_token.
    "description": 'your description',  #no more than 40 chars.
    "filename": "file_name", #your filename
}

url = 'https://biendata.com/competition/kdd_2018_submit/'

response = requests.post(url, files=files, data=data)

print(response.text)


