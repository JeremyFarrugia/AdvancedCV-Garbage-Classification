from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import os

def extract_metadata(image_path):
    print(f"Extracting metadata from {image_path}")
    image = Image.open(image_path)
    metadata = image._getexif()
    if metadata:
        for tag_id, value in metadata.items():
            tag_name = TAGS.get(tag_id, tag_id)
            print(f"{tag_name}: {value}")
    else:
        print("No EXIF metadata found.")

def extract_gps_info(image_path):
    image = Image.open(image_path)
    exif_data = image._getexif()
    if exif_data:
        for tag_id in exif_data:
            tag_name = TAGS.get(tag_id, tag_id)
            if tag_name == "GPSInfo":
                gps_data = exif_data[tag_id]
                for gps_tag in gps_data:
                    gps_tag_name = GPSTAGS.get(gps_tag, gps_tag)
                    print(f"{gps_tag_name}: {gps_data[gps_tag]}")
    else:
        print("No EXIF metadata found.")


script_dir = os.path.dirname(__file__)
folders = os.listdir(script_dir)
folders = [os.path.join(script_dir, folder) for folder in folders if os.path.isdir(os.path.join(script_dir, folder))]
print (folders)
folder = folders[0]
folder = os.path.join(script_dir, "General")
print(folder)
files = os.listdir(folder)

print("\nExtracting metadata from IMG_4296.JPG\n\n")
extract_gps_info(os.path.join(folder, "IMG_4296.JPG"))
print("\nExtracting GPS info from IMG_4296.JPG\n\n")
extract_gps_info(os.path.join(folder, "IMG_4296(E).JPG"))

"""print(files)
for file in files:
    if file.endswith(".JPG"):
        extract_metadata(os.path.join(folder, file))
        extract_gps_info(os.path.join(folder, file))
        print()"""
