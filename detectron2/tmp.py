import json

# Replace 'your_file.json' with the path to your JSON file
with open('../../data_thang_faster_rcnn/chest-imagenome-dataset-1.0.0/silver_dataset/scene_graph/663260a5-d713385d-6026c67d-253e238d-f3d9a3c2_SceneGraph.json', 'r') as file:
    data = json.load(file)

# Print the entire JSON content
# print(json.dumps(data, indent=4))

# Or inspect specific parts of the JSON
for key in data.keys():
    print(f"Top-level key: {key}")
    
    # Check if the value of this key is a dictionary
    if isinstance(data[key], dict):
        # If it is, iterate through its keys
        for key2 in data[key].keys():
            print(f"    Nested key: {key2}")

    # print(data[key])
    # print("\n\n")
    