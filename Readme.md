# Detr Object Detection using PyCharm and HuggingFace models

based on this LearnOpenCV.com tutorial:

https://learnopencv.com/detr-overview-and-inference/#aioseo-predicting-on-video-dataset

## YouTube Video

OpenCV and PyCharm

https://www.youtube.com/watch?v=yqV-kG9wMhM

## Target OS

* MacOS

## Data

```shell
curl -L "https://www.dropbox.com/scl/fi/ekllt8pwum1cu41ohsddh/inference_data.zip?rlkey=b1iih9q1mct5vcnwiyw98ouup&st=uncr8qho&dl=1" -o 
inference_data.zip

```

## Setup

* install pip-tools

```shell
pip install pip-tools
```

* create requirements.txt

```shell
pip-compile
```

* sync requirements.txt with local python environment
```shell
pip-sync
```

## Video Example

update the `object-detection-video.py` file with a path to the video to use

```shell

python object-detection-video.py
```

