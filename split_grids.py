from PIL import Image
import os

# Input and output folders
SAMPLES_DIR = "generated_images/samples"
OUTPUT_DIR = "generated_images/fake_flowers"
GRID_SIZE = 8  # 8x8 = 64 images per grid

os.makedirs(OUTPUT_DIR, exist_ok=True)

def split_grid(image_path, output_dir, prefix):
    img = Image.open(image_path)
    width, height = img.size
    cell_width = width // GRID_SIZE
    cell_height = height // GRID_SIZE

    count = 0
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            left = col * cell_width
            upper = row * cell_height
            right = left + cell_width
            lower = upper + cell_height
            cropped = img.crop((left, upper, right, lower))
            cropped.save(os.path.join(output_dir, f"{prefix}_{count:03}.png"))
            count += 1

    print(f"✅ Processed {count} images from {os.path.basename(image_path)}")

# Run the splitter on all sample grids
for filename in os.listdir(SAMPLES_DIR):
    if filename.endswith(".png"):
        full_path = os.path.join(SAMPLES_DIR, filename)
        prefix = os.path.splitext(filename)[0]
        split_grid(full_path, OUTPUT_DIR, prefix)
