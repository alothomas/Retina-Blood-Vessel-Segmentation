import os
import tifffile
os.chdir(r"N:\\00_Exchange\\Alois")

base_dir = "Skeleton_analysis_full"

# Output directory for decomposed data
output_base_dir = os.path.join(base_dir, "Decomposed_data")

# Iterate through each folder in the base directory
for folder in ["11C7"]:
    folder_path = os.path.join(base_dir, folder)
    output_folder_path = os.path.join(output_base_dir, folder)

    os.makedirs(output_folder_path, exist_ok=True)

    # Iterate through each .tif file in the folder
    for file in os.listdir(folder_path):
        if file.endswith(".tif"):
            # Construct file paths
            file_path = os.path.join(folder_path, file)
            file_base_name = file.split('.')[0]  # Remove .tif extension

            # Create a subfolder for the current file in the output directory
            file_output_folder = os.path.join(output_folder_path, file_base_name)
            os.makedirs(file_output_folder, exist_ok=True)

            # Read the tif file including all its layers
            with tifffile.TiffFile(file_path) as tif:
                for i, page in enumerate(tif.pages):
                    # Define the output path for the current layer
                    output_path = os.path.join(file_output_folder, f"{file_base_name}_layer_{i}.tif")
                    
                    # Save the current layer as a new tif file
                    with tifffile.TiffWriter(output_path) as tif_writer:
                        tif_writer.save(page.asarray())

print("Layer extraction and saving completed.")
