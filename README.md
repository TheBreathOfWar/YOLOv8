# YOLOv8
## test task

## Stage I: clone

### ```$ git clone https://github.com/TheBreathOfWar/YOLOv8 -b dev```

## Stage II: install requirements

### ```$ cd YOLOv8```
### ```$ pip install -r requirements.txt```

## Stage III: run

### ```--image``` - path to input image
### ```--selection_type``` - 1 for boxes or 0 for masks
### ```-selection_color``` - color for baxes and masks in HEX format without # (FFFFFF)

## Example:

### ```$ python main.py --image input.jpg --selection_color 8B00FF --selection_type 0```

<image src="input.jpg" alt="Before">
<image src="output.jpg" alt="After">
