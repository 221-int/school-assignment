import os
import cv2
import argparse
from functions import process_image

def main():
    parser = argparse.ArgumentParser(description="Process images and save results.")
    parser.add_argument("input_folder", type=str, help="Directory containing images to process.")
    args = parser.parse_args()
    
    input_folder = args.input_folder
    output_folder = os.path.join(input_folder, "Results")
    failed_files = []
    
    # Ensure output directory exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Get list of files in the input folder
    file_list = sorted(os.listdir(input_folder))
    
    # Process each image file in the input folder
    for filename in file_list:
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            # Process image and save result
            processed_img = process_image(img_path, failed_files)
            if processed_img is not None:
                cv2.imwrite(output_path, processed_img)
                print(f"Successfully processed and saved: {output_path}")
            else:
                print(f"Failed to process image: {filename}")
    
    # Summarize failed files
    if not failed_files:
        print("All files were processed successfully.")
    else:
        print(f"Some files failed to process: {len(failed_files)} files failed. Re-try processing them.")
        for failed_file in failed_files:
            print(f"Failed file: {failed_file}")

# Entry point of the script
if __name__ == "__main__":
    main()