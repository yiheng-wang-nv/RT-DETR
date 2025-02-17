#!/bin/bash

# ngc dataset upload --source real-colon-annotations.zip --desc "real colon dataset annotations and csv files" real_colon_dataset_annotationds
# Loop through all files with the suffix "_frames.zip"
# for file in *_frames.zip; do
#   # Check if the file exists
#   if [[ -f "$file" ]]; then
#     # Remove the .zip suffix to get the base filename
#     base_filename="${file%.zip}"
#     echo "Uploading $file"
#     ngc dataset upload --source "$file" --desc "real colon dataset frames $base_filename" "real_colon_dataset_$base_filename"
#   fi
# done

ngc dataset upload --source 001-008_frames.zip --desc "real colon dataset frames 001-008_frames" "real_colon_dataset_001-008_frames" --format_type ascii
ngc dataset upload --source 001-009_frames.zip --desc "real colon dataset frames 001-009_frames" "real_colon_dataset_001-009_frames" --format_type ascii
ngc dataset upload --source 002-005_frames.zip --desc "real colon dataset frames 002-005_frames" "real_colon_dataset_002-005_frames" --format_type ascii


