#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import json

import whisper
from flask import Flask, request, make_response

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    model = whisper.load_model("base")
    f = request.files['audio']
    f.save('/tmp/audio.wav')
    resp = make_response()
    try:
        result = model.transcribe('/tmp/audio.wav')
        resp.status_code = 200
        resp.content_type = 'application/json'
        resp.data = json.dumps({
            "status": "ok",
            "text": result['text']
        })
    except IOError:
        resp = make_response()
        resp.status_code = 500
    return resp

@app.route('/train', methods=['POST'])
def train():
    model = whisper.load_model("base")
    f = request.files['audio']
    f.save('/tmp/audio.wav')
    text = request.form['text']
    resp = make_response()
    resp.status_code = 200
    resp.content_type = 'application/json'
    resp.data = json.dumps({
        "status": "ok",
        "text": text
    })
    return resp