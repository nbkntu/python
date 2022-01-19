import uvicorn
import subprocess
import os
from fastapi import FastAPI
#from yolov3_tf2 import detect

app = FastAPI()

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

# image_id = 'img1', image_path = './data/girl.png'
@app.get('/{image_id}/{image_file_name}')
def get_bounding_boxes(image_id: str, image_file_name: str):
    image_path = "./data/" + image_file_name
    #detect2.detect(image_id, image_path)
    output = subprocess.check_output(['python', './detect.py', '--image', image_path])
    print("output = ", output)
    return {'message': f'image id: {image_id}, image path: {image_path}'}

# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    os.chdir(os.getcwd() + '/yolov3_tf2')
    uvicorn.run(app, host='127.0.0.1', port=8000)
    #get_bounding_boxes(image_id="img1", image_file_name="girl.png")