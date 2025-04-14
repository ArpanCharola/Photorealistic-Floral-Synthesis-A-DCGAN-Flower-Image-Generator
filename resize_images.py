from PIL import Image
import os

def resize_images_in_folder(folder_path, size=(299, 299)):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder_path, filename)
            img = Image.open(img_path).convert('RGB')
            img = img.resize(size)
            img.save(img_path)
            print(f"✅ Resized: {filename}")

# Update these paths as needed
resize_images_in_folder("real_images/flowers")
resize_images_in_folder("generated_images/fake_flowers")
