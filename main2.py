import numpy as np
import rasterio
import os

def process_all_tiff_files(input_directory, output_directory):
    # Create the output directory if it doesn't already exist
    os.makedirs(output_directory, exist_ok=True)

    # Loop through each file in the specified input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a TIFF file
        if filename.endswith(".tif"):
            input_filepath = os.path.join(input_directory, filename)
            # Create a unique name for the output file
            output_filename = f"processed_{filename}"
            output_filepath = os.path.join(output_directory, output_filename)

            try:
                with rasterio.open(input_filepath) as src:
                    # Ensure the file has at least two bands to read
                    if src.count < 2:
                        print(f"Skipping '{filename}': Not enough bands found (expected at least 2).")
                        continue

                    # Read the first two bands (typically red and green, or band 1 and band 2)
                    # We specify 1 and 2 to ensure we read specific bands.
                    band1 = src.read(1)
                    band2 = src.read(2)

                    # Initialize 'total' as a 64-bit float array to prevent overflow
                    # when summing pixel values, especially with higher bit-depth images.
                    total = np.zeros(band1.shape, dtype=np.float64)

                    # Add the two bands together
                    total += band1
                    total += band2

                    # Calculate the average of the two bands
                    total /= 2

                    # Get the profile from the source file and update it for the output
                    profile = src.profile
                    profile.update(
                        dtype=rasterio.uint8,  # Output as 8-bit unsigned integer
                        count=1,               # Output will have one band
                        compress='lzw'         # Apply LZW compression for efficiency
                    )

                    # Write the processed band to a new TIFF file
                    with rasterio.open(output_filepath, 'w', **profile) as dst:
                        # Convert the total array to uint8 before writing
                        dst.write(total.astype(rasterio.uint8), 1)
                
                print(f"Successfully processed: '{filename}' -> '{output_filename}'")

            except rasterio.errors.RasterioIOError as e:
                print(f"Error processing '{filename}': {e}. This file might be corrupted or unreadable.")
            except Exception as e:
                print(f"An unexpected error occurred while processing '{filename}': {e}")

# --- How to use this function ---
if __name__ == "__main__":
    # Define your input directory where the original TIFF files are located
    input_dir = '01_Train_Val_Oil_Spill_images\\Oil'
    # Define your output directory where the processed TIFF files will be saved
    output_dir = 'processed_oil_spill_images'

    process_all_tiff_files(input_dir, output_dir)