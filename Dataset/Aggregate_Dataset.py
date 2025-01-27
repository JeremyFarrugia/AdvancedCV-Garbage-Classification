import os
from PIL import Image

"""
This script is provided as a utility to aggregate the individual parts of the dataset into a single directory.
"""


parent_dir = os.path.dirname(__file__)
dataset_paths = [folder for folder in os.listdir(parent_dir) if os.path.isdir(os.path.join(parent_dir, folder))]
print(dataset_paths)

target_dir = os.path.join(parent_dir, 'Dataset') # Modify as you require

c: int = 0

for fragment in dataset_paths:
    fragment_path = os.path.join(parent_dir, fragment)
    for label in os.listdir(fragment_path):
        label_path = os.path.join(fragment_path, label)
        target_path = os.path.join(target_dir, label)
        for image in os.listdir(label_path):
            source_image_path = os.path.join(label_path, image)
            target_image_path = os.path.join(target_path, image)
            os.makedirs(target_path, exist_ok=True)
            with Image.open(source_image_path) as img:
                img.save(target_image_path)
                c += 1
            print(f"Saved {target_image_path}")
print(f"""
\n-----------------------------------------------------\n
Dataset aggregation complete. The aggregated dataset is located at:
{target_dir}
and contains the following labels:
{os.listdir(target_dir)}
and a total of {c} images.
""")