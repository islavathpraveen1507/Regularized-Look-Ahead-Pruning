import os
import shutil

src_train = './data/imagenet100/train'
src_val = './data/imagenet100/val'

dst_train = './data/imagenet100_dryrun/train'
dst_val = './data/imagenet100_dryrun/val'

os.makedirs(dst_train, exist_ok=True)
os.makedirs(dst_val, exist_ok=True)

classes = sorted(
    d for d in os.listdir(src_train)
    if os.path.isdir(os.path.join(src_train, d))
)[:100]

print(f"Number of classes: {len(classes)}")

for cls in classes:

    src_train_cls = os.path.join(src_train, cls)
    src_val_cls = os.path.join(src_val, cls)

    dst_train_cls = os.path.join(dst_train, cls)
    dst_val_cls = os.path.join(dst_val, cls)

    os.makedirs(dst_train_cls, exist_ok=True)
    os.makedirs(dst_val_cls, exist_ok=True)

    # Training: 10 images
    train_imgs = sorted(os.listdir(src_train_cls))[:10]

    if len(train_imgs) < 10:
        print(f"WARNING: {cls} has only {len(train_imgs)} training images")

    for img in train_imgs:
        shutil.copy2(
            os.path.join(src_train_cls, img),
            os.path.join(dst_train_cls, img)
        )

    # Validation: 5 images
    if os.path.exists(src_val_cls):

        val_imgs = sorted(os.listdir(src_val_cls))[:5]

        if len(val_imgs) < 5:
            print(f"WARNING: {cls} has only {len(val_imgs)} validation images")

        for img in val_imgs:
            shutil.copy2(
                os.path.join(src_val_cls, img),
                os.path.join(dst_val_cls, img)
            )

print("\nDry-run dataset created successfully.")
print(f"Training:   {dst_train}")
print(f"Validation: {dst_val}")