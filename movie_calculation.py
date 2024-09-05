import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

model = YOLO("D:/python/tenbou_calculation/runs/segment/train7/weights/best.pt")

colors = [(0, 0, 200), (0, 200, 0),(200,0,0),(0,0,0)]
linewidth = 5
fontScale = 1.5
fontFace = cv2.FONT_HERSHEY_SIMPLEX
thickness = 5




def label(box, img, color, label, line_thickness=3):
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])

    text_size = cv2.getTextSize(
        label, 0, fontScale=fontScale, thickness=line_thickness
    )[0]
    cv2.rectangle(
        img, (x1, y1), (x1 + text_size[0], y1 - text_size[1] - 2), color, -1
    )  # fill
    cv2.putText(
        img,
        label,
        (x1, y1 - 3),
        fontFace,
        fontScale,
        [225, 255, 255],
        thickness=line_thickness,
        lineType=cv2.LINE_AA,
    )

    cv2.rectangle(
        img,
        (x1, y1),
        (x2, y2),
        color,
        linewidth,
    )



def main():
    cap = cv2.VideoCapture("D:/python/tenbou_calculation/1000.mp4")
    #os.makedirs(output_path, exist_ok=True)

    #files = glob.glob("{}/*.png".format(input_path))
    while True:
        ok,frame = cap.read()
        if not ok:break
        #basename = os.path.splitext(os.path.basename(file))[0]
        #image = cv2.imread(file)

        h, w, _ = frame.shape

        results = model(frame, conf=0.60, iou=0.5)
        result = results[0]

        if result.masks is not None:
            for r in results:
                boxes = r.boxes
                conf_list = r.boxes.conf.tolist()

            for i, (seg, box) in enumerate(zip(result.masks.data.cpu().numpy(), boxes)):

                seg = cv2.resize(seg, (w, h))
                #image = overlay(image, seg, colors[int(box.cls)], 0.5)

                class_id = int(box.cls)
                box = box.xyxy.tolist()[0]
                class_name = result.names[class_id]
                label(
                    box,
                    frame,
                    colors[class_id],
                    "{} {:.2f}".format(class_name, conf_list[i]),
                    line_thickness=3,
                )
        cv2.imshow("movie",frame)
        cv2.waitKey(1)
        #output_filename = "{}/{}.png".format(output_path, basename)
        #print(output_filename)
        #cv2.imwrite(output_filename, image)
        #break











#cap = cv2.VideoCapture("D:/python/tenbou_calculation/1000.mp4")


#while True:
#    ok,frame = cap.read()
#    if not ok:break
#    cv2.imshow("movie",frame)
#    cv2.waitKey(1)
#    #break

#cap.release()
#cv2.destroyAllWindows()

main()