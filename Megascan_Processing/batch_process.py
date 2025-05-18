
 #!/usr/bin/env python3 import subprocess
# 定义一个包含所有资产信息的列表，每个资产使用字典来保存相关参数


# 调用 subprocess 执行命令 subprocess.run(command)


          
import os
import subprocess


def batch_process_library(
        library_dir: str,
        hdri_path: str,
        output_dir: str,
        num_angles: int
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



def main():
    library_dir = r'/home/j/projects/replicatePaper_boost3D/Megascan_Processing/test_megascan'
    hdri_path = r'/home/j/projects/replicatePaper_boost3D/Megascan_Processing/HDRI/meadow_2_4k.exr'
    num_angles = 3
    output_dir = r'/home/j/projects/replicatePaper_boost3D/Megascan_Processing/output'
    batch_process_library(  library_dir,  hdri_path, output_dir, num_angles)


if __name__ == "__main__":
    main()