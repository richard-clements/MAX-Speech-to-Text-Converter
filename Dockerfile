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

FROM python:3.11-bookworm
RUN apt install ffmpeg

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
RUN python load_model.py

# check file integrity
RUN sha512sum -c sha512sums.txt

EXPOSE 5000

CMD python app.py
