from PIL import Image
import os
import random

def create_concat_images(folder_a, folder_b, folder_c, output_dir, unit_size=100, number=None):
    """
    Creates and saves four images:
    1. Left big image (2x2)
    2. Top row of images (6x1)
    3. Bottom row of images (6x1)
    4. Combined final image (8x2)
    
    Args:
        folder_a (str): Path to folder with large left image (2x2)
        folder_b (str): Path to folder with top row images (1x1)
        folder_c (str): Path to folder with bottom row images (1x1)
        output_dir (str): Directory to save output images
        unit_size (int): Size of one unit in pixels (default 100px)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Calculate dimensions
    total_width = 8 * unit_size
    total_height = 2 * unit_size
    left_size = 2 * unit_size
    small_size = unit_size
    right_width = 6 * unit_size

    # Get random images from each folder
    img_a = random.choice([f for f in os.listdir(folder_a) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    imgs_b = random.sample([f for f in os.listdir(folder_b) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], 6)
    imgs_c = random.sample([f for f in os.listdir(folder_c) if f.lower().endswith(('.png', '.jpg', '.jpeg'))], 6)

    # Create and save left big image (2x2)
    left_img = Image.open(os.path.join(folder_a, img_a))
    left_img = left_img.resize((left_size, left_size), Image.Resampling.LANCZOS)
    left_img.save(os.path.join(output_dir, f'left_image_{number}.jpg'))

    # Create and save top row (6x1)
    top_row = Image.new('RGB', (right_width, small_size), 'white')
    for i, img_name in enumerate(imgs_b):
        img = Image.open(os.path.join(folder_b, img_name))
        img = img.resize((small_size, small_size), Image.Resampling.LANCZOS)
        top_row.paste(img, (i * small_size, 0))
    top_row.save(os.path.join(output_dir, f'top_row_{number}.jpg'))

    # Create and save bottom row (6x1)
    bottom_row = Image.new('RGB', (right_width, small_size), 'white')
    for i, img_name in enumerate(imgs_c):
        img = Image.open(os.path.join(folder_c, img_name))
        img = img.resize((small_size, small_size), Image.Resampling.LANCZOS)
        bottom_row.paste(img, (i * small_size, 0))
    bottom_row.save(os.path.join(output_dir, f'bottom_row_{number}.jpg'))

    # Create combined final image (8x2)
    result = Image.new('RGB', (total_width, total_height), 'white')
    result.paste(left_img, (0, 0))
    result.paste(top_row, (left_size, 0))
    result.paste(bottom_row, (left_size, small_size))
    result.save(os.path.join(output_dir, f'combined_image_{number}.jpg'))

    

    return {
        'left_image': os.path.join(output_dir, f'left_image_{number}.jpg'),
        'top_row': os.path.join(output_dir, f'top_row_{number}.jpg'),
        'bottom_row': os.path.join(output_dir, f'bottom_row_{number}.jpg'),
        'combined': os.path.join(output_dir, f'combined_image_{number}.jpg')
    }

# Example usage
if __name__ == "__main__":
    random.seed(1066)
    number = 121
    # Replace these paths with your actual folder paths
    folder_a = f"./data/celeba/{number}"  # Folder with 2x2 images
    folder_b = f"./data/outputs/celeba/clean_t2i/{number}"  # Folder with top row 1x1 images
    folder_c = f"./data/outputs/celeba/ace_t2i/{number}"  # Folder with bottom row 1x1 images
    output_dir = "./data/visualization/"  # Directory to save all output images
    
    # Creates four images in the output directory:
    # - left_image.jpg (2x2)
    # - top_row.jpg (6x1)
    # - bottom_row.jpg (6x1)
    # - combined_image.jpg (8x2)
    image_paths = create_concat_images(folder_a, folder_b, folder_c, output_dir, unit_size=100, number=number)
    
    print("Generated images:")
    for name, path in image_paths.items():
        print(f"{name}: {path}")