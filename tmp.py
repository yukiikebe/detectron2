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

category_mapping = {}
class_names = 0
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

def read_csv(csv_file):
    df = pd.read_csv(csv_file)
    # print(df.head())
    return df

def get_category_id_from_object_id(object_id, category_mapping=None):
    category_name = object_id.split('_')[-1].strip().lower()
    
    if category_name in category_mapping:
        return category_mapping[category_name]
    else:
        new_category_id = len(category_mapping)
        category_mapping[category_name] = new_category_id
        return new_category_id
  
def get_dataset_dicts(img_dir, json_dir, csv):
    df = read_csv(csv)
    filenames = df['dicom_id'].tolist()
    imgname = df['path'].tolist()
    json_files = [f + '_SceneGraph.json' for f in filenames]
    image_id  = 0
    imgname_jpg = [f.split('.')[0] + '.jpg' for f in imgname]
    
    dataset_dicts = []
    for json_file in json_files:
        try:
            with open(os.path.join(json_dir, json_file)) as f:   
                imgs_anns = json.load(f)
        except FileNotFoundError:
            print(f"File not found: {json_file}. Skipping to the next file.")
            continue 
        
        filename = os.path.join(img_dir,imgname_jpg[image_id])
        if os.path.exists(filename):
            image = cv2.imread(filename)

            if image is not None:
                height, width = image.shape[:2]
            else:
                print(f"Failed to load image: {filename}")
        else:
            print(f"File not found: {filename}")
        height, width = cv2.imread(filename).shape[:2]
    
        record = {}
        record["file_name"] = filename
        # import pdb; pdb.set_trace()
        record["image_id"] = imgs_anns["image_id"]
        record["height"] = height
        record["width"] = width
    
        # annos = imgs_anns["reason_for_exam"]
        objs_json = imgs_anns["objects"]
        objs = []
        for obj_json in objs_json:
            object_id = obj_json["object_id"]
            x1 = obj_json["original_x1"]
            y1 = obj_json["original_y1"]
            x2 = obj_json["original_x2"]
            y2 = obj_json["original_y2"]
            poly = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            poly = [p for point in poly for p in point]
            category_id = get_category_id_from_object_id(object_id, category_mapping)
            
            obj = {
                "bbox": [x1,y1,x2,y2],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": category_id,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
            
        image_id += 1
    
    return dataset_dicts, category_id

for d in ["train", "val"]:
    data_dicts, class_names = get_dataset_dicts("../data_thang_faster_rcnn", "../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph", '../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/splits/train.csv')
    # print(category_mapping)
    # print(d)
    # import pdb; pdb.set_trace()
    DatasetCatalog.register("chestimagenome_" + d, lambda d=data_dicts: data_dicts)
    MetadataCatalog.get("chestimagenome_" + d).set(thing_classes=list(category_mapping.keys()))
chestimagenome_metadata = MetadataCatalog.get("chestimagenome_train")
print('last class_names:', class_names)
# for d in ["train", "val"]:
#     DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts("balloon/" + d))
#     MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
# balloon_metadata = MetadataCatalog.get("balloon_train")

# dataset_dicts = get_dataset_dicts("../data_thang_faster_rcnn", "../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph", '../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/splits/train.csv')
#for multiple classes
print('class_names:', class_names)
print('#################')
from detectron2.engine import DefaultTrainer
import tensorboard
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"))
cfg.DATASETS.TRAIN = ("chestimagenome_train",)
print("Output dir:",cfg.OUTPUT_DIR)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")  # Let training initialize from model zoo
cfg.SOLVER.IMS_PER_BATCH = 2  # This is the real "batch size" commonly known to deep learning people
cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
cfg.SOLVER.MAX_ITER = 300    # 300 iterations seems good enough for this toy dataset; you will need to train longer for a practical dataset
cfg.SOLVER.STEPS = []       # do not decay learning rate
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128   # The "RoIHead batch size". 128 is faster, and good enough for this toy dataset (default: 512)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = class_names  # only has one class (ballon). (see https://detectron2.readthedocs.io/tutorials/datasets.html#update-the-config-for-new-datasets)
cfg.MODEL.RETINANET.NUM_CLASSES = class_names
# NOTE: this config means the number of classes, but a few popular unofficial tutorials incorrect uses num_classes+1 here.
print('start train')
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg) 
trainer.resume_or_load(resume=False)
trainer.train()

#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=1

# cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")  # path to the model we just trained
# cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7   # set a custom testing threshold
# predictor = DefaultPredictor(cfg)

