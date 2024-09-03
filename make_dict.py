from detectron2.structures import BoxMode
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2
import pandas as pd

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    # print(df.head())
    return df
  
def get_chestimagenome(img_dir, json_dir, csv):
    df = read_csv(csv)
    filenames = df['dicom_id'].tolist()
    imgname = df['path'].tolist()
    json_files = [f + '_SceneGraph.json' for f in filenames]
    image_id  = 0
    imgname_jpg = [f.split('.')[0] + '.jpg' for f in imgname]
    
    imgs_anns = {}
    dataset_dicts = []
    for json_file in json_files:
        with open(os.path.join(json_dir, json_file)) as f:
            data = json.load(f)
            imgs_anns.update(data)
            imgs_anns[json_file] = data
        
        filename = os.path.join(img_dir,imgname_jpg[image_id])
        height, width = cv2.imread(filename).shape[:2]
    
        for idx, v in enumerate(imgs_anns[json_file].values()):
            record = {}
            
            record["file_name"] = filename
            record["image_id"] = v["image_id"]
            record["height"] = height
            record["width"] = width
        
            annos = v["reason_for_exam"]
            objs_json = v["objects"]
            objs = []
            for _, anno in objs_json.items():
                anno = anno["object_id"]
                print(anno)
                x1 = anno["original_x1"]
                y1 = anno["original_y1"]
                x2 = anno["original_x2"]
                y2 = anno["original_y2"]
                poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                poly = [p for point in poly for p in point]

                obj = {
                    "bbox": [x1,y1,x2,y2],
                    "bbox_mode": BoxMode.XYXY_ABS,
                    "segmentation": [poly],
                    "category_id": 0,
                }
                objs.append(obj)
            record["annotations"] = objs
            dataset_dicts.append(record)
            
        image_id += 1
    return dataset_dicts

# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")

get_chestimagenome("../data_thang_faster_rcnn", "../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph", '../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/splits/train.csv')

