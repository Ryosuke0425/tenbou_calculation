import random
from PIL import Image
import glob
import numpy as np

def main():
  current = "D:/python/tenbou_calculation/"
  for k in range(3000):
    labels = ["100","1000","5000","10000"]
    training_type = "train" if k < 2400 else "val"
    back_full = np.full((1080, 1920), False)
    back = Image.open(random.choice(glob.glob(current + "material/back/*.jpg")))
    back = back.resize((1920,1080))


    output_texts = []
    for i in range(150):
      label = random.choice(labels)
      file = random.choice(glob.glob(current + "material/" + label + "/*_t.png"))
      img = Image.open(file)
      w = img.size[0]
      h = img.size[1]

      with open(file.replace("_t.png","_l.txt")) as f:
        text = f.read()

      pos = text.split(",")[:-1]
      pos = list(map(float,pos))

      x = random.randint(0,1920-w)
      y = random.randint(0,1080-h)
      img_cut =  np.all((np.array(img) != [255,255,255,0]),axis=-1)
      if np.count_nonzero(img_cut & back_full[y:y+h,x:x+w]) / np.count_nonzero(img_cut) > 0.0:
        continue

      for j in range(len(pos)):
        if j % 2 == 0:
          pos[j] += x
          pos[j] /= 1920
        else:
          pos[j] += y
          pos[j] /= 1080
      pos.insert(0,labels.index(label))
      pos = list(map(str,pos))
      output_text = " ".join(pos)
      output_texts.append(output_text)
      back_full[y:y+h,x:x+w] = (back_full[y:y+h,x:x+w] | img_cut)
      back.paste(img,(x,y),mask=img)
    with open(current + "dataset/yolo_data/" + training_type + "/labels/" + str(k).zfill(4) + ".txt",mode="w") as f:
      f.write("\n".join(output_texts))
    back.save(current + "dataset/yolo_data/" + training_type + "/images/" + str(k).zfill(4) + ".png")


if __name__ == "__main__":
    main()