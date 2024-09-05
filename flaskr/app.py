from flask import Flask,render_template,redirect,request
import matplotlib.pylab as plt
import numpy as np
import cv2
import base64
import re

from ultralytics import YOLO
import os
import glob



app =Flask(__name__)
model = YOLO("D:/python/tenbou_calculation/runs/segment/train7/weights/best.pt")
colors = [(0, 0, 200), (0, 200, 0),(200,0,0),(0,0,0)]
linewidth = 5
fontScale = 1.5
fontFace = cv2.FONT_HERSHEY_SIMPLEX
thickness = 5



@app.route('/',methods=['GET','POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        score = 0
        stream = request.files['image'].stream
        img_array = np.asarray(bytearray(stream.read()),dtype=np.uint8)
        img = cv2.imdecode(img_array,1)
        img = cv2.resize(img,(1000,1000))
        try:
            img,score = calculation(img)
        except:
            pass
        _,img_array = cv2.imencode('.png',img)
        img_array = base64.b64encode(img_array)
        uploadimage_base64_string = re.sub('b\'|\'','',str(img_array))
        filebinary = f'data:image/png;base64,{uploadimage_base64_string}'
        return render_template('result.html',image=filebinary,score=score)

@app.route('/result')
def result():
    return render_template('result.html')








#参考にしたコード https://dev.classmethod.jp/articles/yolov8-instance-segmentation/

def overlay(image, mask, color, alpha, resize=None):
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()
    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
    return image_combined


def label(box, img, color, label, line_thickness=3):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    text_size = cv2.getTextSize(
        label, 0, fontScale=fontScale, thickness=line_thickness
    )[0]
    #cv2.rectangle(
    #    img, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 2), color, -1
    #)  # fill
    #cv2.putText(
    #    img,
    #    label,
    #    (x1, y1 - 3),
    #    fontFace,
    #    fontScale,
    #    [225, 255, 255],
    #    thickness=line_thickness,
    #    lineType=cv2.LINE_AA,
    #)

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        linewidth,
    )



def calculation(img):
    scores = [100,1000,5000,10000]
    score = 0
    #os.makedirs(output_path, exist_ok=True)

    #files = glob.glob("{}/*.png".format(input_path))
    #basename = os.path.splitext(os.path.basename(file))[0]
    #image = cv2.imread(file)

    h, w, _ = img.shape

    results = model(img, conf=0.60, iou=0.5)
    result = results[0]

    if result.masks is not None:
        for r in results:
            boxes = r.boxes
            conf_list = r.boxes.conf.tolist()

        for i, (seg, box) in enumerate(zip(result.masks.data.cpu().numpy(), boxes)):

            seg = cv2.resize(seg, (w, h))
            #img = overlay(img, seg, colors[int(box.cls)], 0.5)

            class_id = int(box.cls)
            score += scores[class_id]
            box = box.xyxy.tolist()[0]
            class_name = result.names[class_id]
            label(
                box,
                img,
                colors[class_id],
                "{} {:.2f}".format(class_name, conf_list[i]),
                line_thickness=3,
            )
        return img,score






if __name__ == "__main__":
    app.run(debug=True)