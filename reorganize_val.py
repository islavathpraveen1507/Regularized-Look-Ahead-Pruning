import os
import shutil

# Adjust this path if your val folder lives elsewhere
val_dir = './data/tiny-imagenet-200/val'
images_dir = os.path.join(val_dir, 'images')
annotations_file = os.path.join(val_dir, 'val_annotations.txt')

with open(annotations_file, 'r') as f:
    lines = f.readlines()

moved = 0
skipped = 0
for line in lines:
    parts = line.strip().split('\t')
    filename, class_id = parts[0], parts[1]

    class_folder = os.path.join(val_dir, class_id)
    os.makedirs(class_folder, exist_ok=True)

    src = os.path.join(images_dir, filename)
    dst = os.path.join(class_folder, filename)

    if os.path.exists(src):
        shutil.move(src, dst)
        moved += 1
    else:
        skipped += 1

print(f"Moved {moved} images into class folders.")
print(f"Skipped {skipped} images (not found — check for a mismatch).")

# Optional: remove the now-empty flat images folder
if os.path.isdir(images_dir) and not os.listdir(images_dir):
    os.rmdir(images_dir)
    print("Removed empty original images/ folder.")
