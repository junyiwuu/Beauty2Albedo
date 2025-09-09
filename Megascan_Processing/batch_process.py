# Copyright 2025 Junyi Wu
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

          
import os
import subprocess
import argparse
import sys
import random

def batch_process_library(
        library_dir: str,
        hdri_path: str,
        output_dir: str,
        num_angles: int,
        res: int
):

    if os.path.exists(library_dir):


        folders = os.listdir(library_dir)
        for folder in folders:
            folder_path = os.path.join(library_dir, folder)
            if os.path.isdir(folder_path):
                args = f" --output_dir {output_dir} "
                args += f" --hdri_path {hdri_path} "
                args += f" --num_angles {num_angles} "
                args += f" --asset_folder {folder_path}"
                args += f" --res {res}"

            command = f"blender --background --python blender_script.py -- {args}"

            print(f"Start Render asset folder:  {folder}")
            subprocess.run(
                ["bash", "-c",  command],
                timeout=300,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )
            print( f"Finished! {folder}")

    else:
        print(f"Library path :: {library_dir} not exist")


# remove "_0001" from blender render output, and create filename_lst file for training
def post_process(root_dir: str, filename_lst = "filename_lst", filname_lst_val = "filename_lst_val"):
    print(f"Start post process")
    beauty_dir = os.path.join(root_dir, "Beauty")
    albedo_dir = os.path.join(root_dir, "Albedo")


    # clean out "_0001"
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            base, ext = os.path.splitext(fname)

            if "_0001" in base:
                clean_name = base.replace("_0001", "") + ext

                old_path = os.path.join(dirpath, fname)
                new_path = os.path.join(dirpath, clean_name)

                #rename files
                os.rename(old_path, new_path)

    # write out pair file 
    pairs = []
    for dirpath, _, filenames in os.walk(beauty_dir):
        for fname in filenames:
            if os.path.exists(os.path.join(albedo_dir, fname)):
                pairs.append(f"Beauty/{fname} Albedo/{fname}")
    with open(os.path.join(root_dir, filename_lst), "w") as f:
        for line in pairs:
            f.write(line + "\n")

    val_num = 10
    val_len = min(val_num, len(pairs))
    sample_pairs = random.sample(pairs, val_len)

    with open(os.path.join(root_dir, filname_lst_val), "w") as f:
        for line in sample_pairs:
            f.write(line + "\n")

    
    print(f"Done write filename_lst and filename_lst_paris file in the path: {root_dir}")
    


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_folder", type=str, required=False, default=None, help="an Asset folder")
    parser.add_argument("--hdri_path", type=str, required=False, default=None , help="HDRI directory")
    parser.add_argument("--output_dir" , type=str, required=True, help="set up output render directory")
    parser.add_argument("--num_angles", type=int, default=1, help="Numbers of camera angles")
    parser.add_argument("--res", type=int, default=128, help="Resolution of output images")

    args = parser.parse_args()

    batch_process_library( args.asset_folder,  args.hdri_path, args.output_dir, args.num_angles, args.res)
    post_process(args.output_dir)

if __name__ == "__main__":
    main()


# Example command line:
# python batch_process.py --asset_folder ./src_assets --hdri_path ./HDRI/meadow_2_4k.exr --output_dir ./output --num_angles 6 --res 256