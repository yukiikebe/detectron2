xhost +local:`docker inspect --format='{{ .Config.Hostname }}' detectron2`

docker run --gpus all -it --rm \
  --shm-size=8gb --env="DISPLAY" --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  --volume="/media/yuki/Seagate12T_Thang/mimic-cxr-jpg:/home/appuser/detectron2_repo/data_thang_faster_rcnn:rw" \
  --volume="/home/yuki/research/detectron2:/home/appuser/detectron2_repo/detectron2:rw" \
  --name=detectron2 yuki/detectron bash
  