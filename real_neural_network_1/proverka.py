import os
from PIL import Image
import cv2
import numpy as np

def is_image_damaged(file_path):
    """
    Checks the integrity of an image using PIL and its resizability using OpenCV.

    Arguments:
        file_path (str): The path to the image file.

    Returns:
        bool: True if the image is damaged or cannot be resized, False otherwise.
    """
    try:
        # PIL integrity check
        with Image.open(file_path) as img:
            img.verify()  # Verifies the integrity of the image file
        
        # Attempt to load the image using OpenCV
        img_cv2 = cv2.imread(file_path)

        # Check if OpenCV successfully loaded the image
        if img_cv2 is None:
            print(f"Error: OpenCV could not load the image at path '{file_path}'.")
            return True
        
        # Check if the image is empty (e.g., if the file is corrupted but PIL didn't error)
        if img_cv2.size == 0:
            print(f"Error: The image loaded by OpenCV is empty at path '{file_path}'.")
            return True

        # Attempt to resize the image using OpenCV
        # Choose a small target size just to check functionality
        target_width = 320
        target_height = 320
        resized_img = cv2.resize(img_cv2, (target_width, target_height), interpolation=cv2.INTER_AREA)

        # Check if resizing was successful
        if resized_img is None or resized_img.size == 0:
            print(f"Error: OpenCV could not resize the image at path '{file_path}'.")
            return True

        return False
    except Exception as e:
        print(f"General error checking image '{file_path}': {e}")
        return True

def process_images_in_directory(directory_path):
    """
    Processes all image files in a given directory to check their integrity and resizability.
    Provides a summary of the results, including a list of damaged images.

    Arguments:
        directory_path (str): The path to the directory containing image files.
    """
    if not os.path.isdir(directory_path):
        print(f"Error: Directory '{directory_path}' does not exist.")
        return

    print(f"Processing images in directory: {directory_path}")
    
    # List of common image extensions
    image_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.webp')

    total_images_checked = 0
    damaged_images = [] # List to store names of damaged images

    for filename in os.listdir(directory_path):
        if filename.lower().endswith(image_extensions):
            total_images_checked += 1
            file_path = os.path.join(directory_path, filename)
            print(f"\nChecking image: {filename}")
            if is_image_damaged(file_path):
                print(f"Result: Image '{filename}' is damaged or cannot be processed.")
                damaged_images.append(filename) # Add to the list if damaged
            else:
                print(f"Result: Image '{filename}' is OK and can be resized.")
    
    print("\n--- Processing Summary ---")
    if total_images_checked == 0:
        print("No image files found in the specified directory.")
    else:
        print(f"Total images checked: {total_images_checked}")
        print(f"Damaged images found: {len(damaged_images)}")
        if damaged_images:
            print("List of damaged images:")
            for img_name in damaged_images:
                print(f"- {img_name}")
        else:
            print("No damaged images found. All images are OK!")
    print("--------------------------")
    print("Finished processing all images in the directory.")


# Example usage:
# Replace "путь/к/вашей/директории/с/изображениями" with the actual path to your image directory
directory_to_check = "C:\\Users\\Sirius\\Desktop\\neuronetwork\\real_neural_network\\Images" 

# Note: The code to create a dummy directory and image has been removed.
# Please ensure the directory specified in 'directory_to_check' exists and contains images.

process_images_in_directory(directory_to_check)
