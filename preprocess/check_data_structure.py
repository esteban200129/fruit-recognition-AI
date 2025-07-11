import os

raw_images_dir = "data/fruit360"
for fruit in os.listdir(raw_images_dir):
    fruit_dir = os.path.join(raw_images_dir, fruit)
    if os.path.isdir(fruit_dir):
        print(f"{fruit}: {len(os.listdir(fruit_dir))} images")