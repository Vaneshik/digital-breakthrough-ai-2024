from fastapi import FastAPI, File, UploadFile
from typing import Union
from fastapi.responses import JSONResponse, HTMLResponse
import shutil
import os
from random import randbytes

import matplotlib
matplotlib.use('Agg')  # Use the Agg backend
import matplotlib.pyplot as plt

from whisper_gpt import Whisper
from tagging import Tagger
from svaston_detected import ForbiddenDetector
from video_interest import Interester
from object_detection import Detector


VIDEO_DIR = None
FILE_NAME = None
TRANSCRIBITION_TEXT = None

project_path = '/Users/vaneshik/hack/CP_CODE'

process_videos = Whisper(
    os.path.join(project_path, 'models/whisper_small_folder')
)

interester = Interester(
    os.path.join(project_path, 'models/VTSum/vtsum_tt.pth')
)

forbidden_detector = ForbiddenDetector(
    os.path.join(project_path, 'models/YOLOv3_FORBIDDEN/best.pt')
)

tagger = Tagger(
		"fabiochiu/t5-base-tag-generation",
		"/Users/vaneshik/hack/CP_CODE/models/T5_TAGGING_folder")

detector = Detector(os.path.join(project_path, 'models/YOLOv3/yolov3.weights'),
                    os.path.join(project_path, 'models/YOLOv3/yolov3.cfg'),
                    os.path.join(project_path, 'models/YOLOv3/coco.names'))


app = FastAPI()


@app.get("/")
def root():
    with open("index.html", "r") as f:
        meow = f.read()
        
    return HTMLResponse(content=meow, status_code=200)


@app.get("/transcribe")
def transcribe():
    global TRANSCRIBITION_TEXT
    
    if VIDEO_DIR is None:
        return JSONResponse(content={"error": "No video uploaded"}, status_code=400)
    
    result = process_videos(VIDEO_DIR)
    TRANSCRIBITION_TEXT = next(iter(result.values()))

    return result


@app.get("/tag")
def tag(transcription: Union[str, None] = None):
    if TRANSCRIBITION_TEXT is None and transcription is None:
        return JSONResponse(content={"error": "transcribe text or provide transcription via get parameter"}, status_code=400)
    tags = tagger(transcription if TRANSCRIBITION_TEXT is None else TRANSCRIBITION_TEXT)
    return tags


@app.get("/detect_object")
def detect_object():
    if FILE_NAME is None:
        return JSONResponse(content={"error": "No video uploaded"}, status_code=400)
    return detector(os.path.join(VIDEO_DIR, FILE_NAME))


@app.get("/detect_forbidden")
def detect_forbidden():
    x = forbidden_detector.predict_bad_symbols(VIDEO_DIR)
    return x


@app.get("/attention_graphic")
def attention_graphic():
    x = interester.get_interests_summary(os.path.join(VIDEO_DIR, FILE_NAME))
    
 # Assuming x[0] is an ndarray
    # plt.imshow(x[0])
    # plt.axis('off')  # Hide axes
    # plt.savefig("summary.png", bbox_inches='tight', pad_inches=0)
    
                
    return x[1]

@app.get("/get_all")
def get_all():
    return transcribe(), tag(), detect_object(), detect_forbidden(), attention_graphic()

#video upload by providing filename
@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    global VIDEO_DIR
    global FILE_NAME

    try:
        FILE_NAME = file.filename

        upload_dir = "upload-" + randbytes(8).hex()
        VIDEO_DIR = upload_dir

        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, file.filename)
    
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        return JSONResponse(content={"filename": file.filename, "message": "Upload successful"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
