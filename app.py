import uvicorn
import subprocess
import os
from fastapi import FastAPI
import detect2

app = FastAPI()

@app.get('/{name}')
def get_name(name: str):
    return {'message': f'Hello, {name}'}

# image_id = 'img1', image_path = './data/girl.png'
@app.get('/{image_id}/{image_file_name}')
def get_bounding_boxes(image_id: str, image_file_name: str):
    image_path = "./yolov3_tf2/data/" + image_file_name
    boxes, scores, classes, nums, img_shape = detect2.detect(image_id, image_path, i_classes='./yolov3_tf2/data/coco2.names', i_yolo_max_boxes=2)
    npboxes = boxes.numpy()
    npboxes = npboxes.reshape((npboxes.shape[1], npboxes.shape[2]))
    classes_list = []
    print(classes.numpy().flatten().tolist())
    for i in range(npboxes.shape[0]):
        npboxes[i][0] = npboxes[i][0]*img_shape[1]
        npboxes[i][1] = npboxes[i][1]*img_shape[0]
        npboxes[i][2] = npboxes[i][2]*img_shape[1]
        npboxes[i][3] = npboxes[i][3]*img_shape[0]
    
    print(npboxes.tolist())
    #output = subprocess.check_output(['python', './detect.py', '--image', image_path])
    #print("output = ", output)
    return {'message': f'image id: {image_id}, image path: {image_path}',
            'bounding_box': npboxes.tolist(),
            'classes': classes.numpy().flatten().tolist()
        }

# Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    #os.chdir(os.getcwd() + '/yolov3_tf2')
    uvicorn.run(app, host='127.0.0.1', port=8000)
    #get_bounding_boxes(image_id="img1", image_file_name="street.jpg")