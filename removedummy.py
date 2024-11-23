


# Git only tracks files, so to give an idea of the folder structure I place dummy files in the folders
# I created this script to remove that file for your convenience

# Once image collection is done we'll combine all the data but for our convenience I created separate folders.
# Change things as you please

import os

script_path = os.path.dirname(os.path.realpath(__file__))
count = 0 # I just like seeing numbers :))

selfDestruct = False # Set to True if you want this file to delete itself

# Walk through the directory and delete the dummy files
for root, dirs, files in os.walk(script_path):
    for file in files:
        if 'dummy' in file and (file != 'removedummy.py' and not selfDestruct):
            os.remove(os.path.join(root, file))
            print(f"Deleted '{file}' in folder '{root}'")
            count += 1

print(f"Deleted {count} files")