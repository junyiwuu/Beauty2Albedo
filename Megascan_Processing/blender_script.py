import bpy
import os
import re
import sys
import argparse
import random
import math
import numpy as np
import mathutils
from mathutils import Vector



# blender --background --python blender_script.py -- 
#           --asset_folder /mnt/D/Quixel/Megascans_Library/Downloaded/3d/wood_planks_tezvbcuda 
#           --hdri_path /mnt/D/HDRI/meadow_2_4k.exr 
#           --output_dir ./output 
#           --num_angles 6

# blender --background --python blender_script.py -- --asset_folder /mnt/D/Quixel/Megascans_Library/Downloaded/3d/wood_planks_tezvbcuda --hdri_path /mnt/D/HDRI/meadow_2_4k.exr --output_dir ./output --num_angles 6

# GLOBAL control
LOD_version = 3

# reset the whole scene
def reset_scene():
    # select all object in current scene. ops-> operators
    bpy.ops.object.select_all(action="SELECT")
    # delete all selected objects
    # use_global means delete literaly everything. But set false here , only delete selected
    bpy.ops.object.delete(use_global=False)


    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)

    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def setup_HDRI(hdri_path:str = None, hdri_rotation: int =0):

    # set up world (top level world)
    world = bpy.data.worlds.new("HDRI_world")
    bpy.context.scene.world = world
    # enable node 
    world.use_nodes = True
    
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    #clean defaults nodes
    for node in nodes:
        nodes.remove(node)

    # if no hdri, use point light
    if not hdri_path or not os.path.isfile(hdri_path):

        bpy.ops.object.light_add(type='POINT', location=(3, 2, 4))
        point_light = bpy.context.active_object
        point_light.data.energy = 1000   
        point_light.data.color = (0.8, 0.9, 1.0)  
        point_light.data.shadow_soft_size = 0.2  
        point_light.data.use_contact_shadow = True  
        return
    
    else:

        environment = nodes.new(type = "ShaderNodeTexEnvironment")
        environment.image = bpy.data.images.load(hdri_path) # load image

        background = nodes.new(type="ShaderNodeBackground")
        background.inputs["Strength"].default_value = 1.0

        node_output = nodes.new(type="ShaderNodeOutputWorld")
       
        # link everything
        # env(with image).color --> bg.color --> bg.background(output) -> out.surface (world input)
        links.new(environment.outputs["Color"] , background.inputs["Color"])
        links.new(background.outputs["Background"] , node_output.inputs["Surface"])

        # rotate the HDRI
        mapping = nodes.new(type="ShaderNodeMapping")
        mapping.inputs["Rotation"].default_value=(0,0,math.radians(hdri_rotation))

        texture_coordinate = nodes.new(type="ShaderNodeTexCoord")
        links.new(texture_coordinate.outputs["Generated"], mapping.inputs["Vector"])
        links.new(mapping.outputs["Vector"] , environment.inputs["Vector"])


def setup_camera(obj , camera_angle: tuple,  margin=1.5) -> bpy.types.Object:
    bpy.ops.object.camera_add()
    cam = bpy.context.active_object
    scene = bpy.context.scene

    # calculate bounding box
    # mathutils.Vector(corner): transfer corner into a vector object
    # obj.atrix_world: transfer the local axis to world axis
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    # explicit provide a vector, otherwise sum will start from 0 integer, which will not work here
    bbox_center = sum(bbox_corners, mathutils.Vector()) / 8.0

    dimensions = obj.dimensions
    max_dim = max(dimensions)

    # 如果还没有相机数据，就使用一个默认视角（这里是 50 度）
    fov = cam.data.angle

    base_distance = (max_dim / 2) / math.tan(fov / 2)
    distance = base_distance * margin


    # set camera's location
    theta, phi = camera_angle
    x = bbox_center.x + distance * math.sin(phi) * math.cos(theta)
    y = bbox_center.y + distance * math.sin(phi) * math.sin(theta)
    z = bbox_center.z + distance * math.cos(phi)
    cam.location = (x, y, z)

    # target camera to object
    direction = bbox_center - cam.location
    cam.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()


    return cam


def import_asset(asset_folder: str = None):
    if asset_folder:
        files = os.listdir(asset_folder)

        found_fbx = False
        found_lod_fbx = False

        # First check if there is any fbx
        for file in files:
            filename = str(file.lower())
            if ".fbx" in filename :
                found_fbx = True
                break
        
        if not found_fbx:
            raise ValueError(f"There is no fbx file in {asset_folder}")


        # find if nay suitable lod fbx file
        target_LOD_version = LOD_version
        while target_LOD_version >=0 and not found_lod_fbx:
            for file in files:
                filename = str(file.lower())
                if ".fbx" in filename :
                    match = re.search(r'lod(\d+)', filename)
                    if match and int(match.group(1))==target_LOD_version: 
                        filepath = os.path.join(asset_folder, file)
                        bpy.ops.import_scene.fbx(filepath=filepath)
                        found_lod_fbx = True
                        break

            if not found_lod_fbx:
                target_LOD_version-=1

        # if no suitable lod file, just load the fbx file
        if not found_lod_fbx:
            for file in files:
                filename = str(file.lower())
                if ".fbx" in filename and "lod" not in filename:
                    filepath = os.path.join(asset_folder, file)
                    bpy.ops.import_scene.fbx(filepath=filepath)
                    found_lod_fbx = True
                    break

        # if all above condition not meet
        if not found_lod_fbx:
            raise ValueError(f"There is no suitable LOD file in {asset_folder}")
        
        imported_obj = bpy.context.selected_objects[0]

        # initialize the material nodes
        mat = bpy.data.materials.new(name="PBR_Material")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            nodes.remove(node)




        # set up texture / find textures
        albedo_path  = None
        rough_path  = None
        normal_path = None

        for file in files:
            filename = str(file.lower())
            match = re.search(f"lod(\d+)", filename)
            if "albedo" in filename:
                if match:
                    if int(match.group(1)) == target_LOD_version:
                        albedo_path = os.path.join(asset_folder, file)
                else:
                    albedo_path = os.path.join(asset_folder, file)

            if "roughness" in filename:
                if match:
                    if int(match.group(1)) == target_LOD_version:
                        rough_path = os.path.join(asset_folder, file)
                else:
                    rough_path = os.path.join(asset_folder, file)


            if "normal" in filename:
                if match:
                    if int(match.group(1)) == target_LOD_version:
                        normal_path = os.path.join(asset_folder, file)
                else:
                    normal_path = os.path.join(asset_folder, file)

            # if "displacement" in filename:
            #     disp_path = os.path.join(asset_folder, file)

        # PBR   --------------------------------------------------
        bsdf = nodes.new(type='ShaderNodeBsdfPrincipled')
        bsdf.distribution="MULTI_GGX"

        albedo_node = nodes.new(type="ShaderNodeTexImage")
        albedo_node.image = bpy.data.images.load(filepath = albedo_path)

        rough_node = nodes.new(type="ShaderNodeTexImage")
        rough_node.image = bpy.data.images.load(filepath = rough_path)

        normal_map_node = nodes.new(type="ShaderNodeTexImage")
        normal_map_node.image = bpy.data.images.load(filepath = normal_path)
        normal_node = nodes.new(type="ShaderNodeNormalMap")
        normal_map_node.image.colorspace_settings.name = 'Non-Color'

        links.new(normal_map_node.outputs["Color"] , normal_node.inputs["Color"])

        # disp_node = nodes.new(type="ShaderNodeTexImage")
        # disp_node.image = bpy.data.images.load(filepath = disp_path)

        output = nodes.new(type='ShaderNodeOutputMaterial')

        
        # link everything to bsdf        
        links.new(albedo_node.outputs["Color"] , bsdf.inputs["Base Color"])
        links.new(rough_node.outputs["Color"] , bsdf.inputs["Roughness"])
        links.new(normal_node.outputs["Normal"] , bsdf.inputs["Normal"])
        links.new(bsdf.outputs['BSDF'], output.inputs['Surface'])

     
        # create AOV
        aov_output_node = nodes.new("ShaderNodeOutputAOV")
        aov_output_node.name = "normal_aov"
        aov_output_node.aov_name = "custom_Normal"

        links.new(normal_map_node.outputs["Color"] , aov_output_node.inputs["Color"])

        # apply to the obj  ---------------------------------
        if imported_obj.data.materials:
            imported_obj.data.materials[0] = mat
        else:
            imported_obj.data.materials.append(mat)
        
        bpy.ops.object.shade_smooth()
        return imported_obj
    

    
    # if not setting asset folder, then render a sphere with simple material

    if not asset_folder:
        bpy.ops.mesh.primitive_uv_sphere_add(radius=1.0, location=(0,0,0))
        sphere=bpy.context.active_object
        mat = bpy.data.materials.new(name="SimpleLambertian")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links

        for node in nodes:
            nodes.remove(node)

        # simple lambertian material
        output = nodes.new(type='ShaderNodeOutputMaterial')
        diffuse = nodes.new(type='ShaderNodeBsdfDiffuse')
        diffuse.inputs['Color'].default_value = (0.8, 0.8, 0.8, 1.0)  

        # Link everything
        links.new(diffuse.outputs['BSDF'], output.inputs['Surface'])

        # apply the material to the object
        if sphere.data.materials:
            sphere.data.materials[0] = mat
        else:
            sphere.data.materials.append(mat)

        # smooth
        bpy.ops.object.shade_smooth()
        return sphere
    

def render_pass(output_dir: str, cam:bpy.types.Object, idx:int, asset_folder: str):
    # scene and camera
    scene = bpy.context.scene
    scene.camera = cam
    # setup output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # output transparent background
    scene.render.film_transparent = True

    scene.use_nodes = True
    node_tree = scene.node_tree
    node_tree.nodes.clear()

  
    # setup view layer and enable diffuse color and normal slot
    view_layer = scene.view_layers["ViewLayer"]
    view_layer.use_pass_diffuse_color = True
    # create new aov in view layer
    bpy.ops.scene.view_layer_add_aov()
    normal_aov = view_layer.aovs[-1]
    normal_aov.name="custom_Normal"
    # view_layer.use_pass_normal = True

    # create new renderlayer node
    rl_node = node_tree.nodes.new(type="CompositorNodeRLayers")
    rl_node.location = (0,0)


    # create File Output node
    file_output_node = node_tree.nodes.new(type="CompositorNodeOutputFile")
    file_output_node.location = (300,0)
    file_output_node.base_path = output_dir
    
    file_output_node.file_slots.new("Beauty")
    file_output_node.file_slots.new("Albedo")
    file_output_node.file_slots.new("Normal")
    normal_slot = file_output_node.file_slots["Normal"]

    asset_name = asset_folder.split("/")[-1]
    file_output_node.file_slots["Beauty"].path = f"{asset_name}_beauty{idx+1}_"
    file_output_node.file_slots["Albedo"].path = f"{asset_name}_albedo{idx+1}_"
    file_output_node.file_slots["Normal"].path = f"{asset_name}_normal{idx+1}_"

    # CAN CHOOSE TO EXR FILE
    normal_slot.use_node_format = False
    normal_slot.format.file_format = "OPEN_EXR"
    normal_slot.format.color_depth = "16"
    normal_slot.format.exr_codec="ZIP"

    set_alpha_albedo_node = node_tree.nodes.new(type="CompositorNodeSetAlpha")
    node_tree.links.new( rl_node.outputs["Alpha"] , set_alpha_albedo_node.inputs["Alpha"])

    # connect each file slot to renderlayer
    node_tree.links.new(
        rl_node.outputs["Image"] , file_output_node.inputs["Beauty"]
    )
    
    # link to alpha and to output ALBEDO
    node_tree.links.new(
        rl_node.outputs["DiffCol"] , set_alpha_albedo_node.inputs["Image"]
    )
    node_tree.links.new(
        set_alpha_albedo_node.outputs["Image"] , file_output_node.inputs["Albedo"]
    )


    node_tree.links.new(
        rl_node.outputs["custom_Normal"] , file_output_node.inputs["Normal"]
    )
    

    bpy.ops.render.render(write_still=True)
  


        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset_folder", type=str, required=False, default=None, help="an Asset folder")
    parser.add_argument("--hdri_path", type=str, required=False, default=None , help="HDRI directory")
    parser.add_argument("--output_dir" , type=str, required=True, help="set up output render directory")
    parser.add_argument("--num_angles", type=int, default=1, help="Numbers of camera angles")

    args = parser.parse_args(sys.argv [sys.argv.index("--")+1 : ])

    bpy.context.scene.render.engine = 'BLENDER_EEVEE_NEXT'
    # bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.samples = 128 
    # bpy.context.scene.cycles.device = 'GPU' 
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512
    reset_scene()
    asset = import_asset(args.asset_folder)

    asset_name = args.asset_folder.split("/")[-1]

    angles = []
    for i in range(args.num_angles):
        theta = 2 * math.pi * i / args.num_angles + math.radians(30)
        phi = math.pi / 3 
        angles.append((theta, phi))


    
    for idx, angle in enumerate(angles):
        cam = setup_camera(asset, angle)
        rand = random.randint(1,360)
        setup_HDRI(args.hdri_path , rand)
        pair_folder = asset_name + f"_PAIR_{idx}"
        output_pair_dir = os.path.join(args.output_dir , pair_folder)
        os.makedirs(output_pair_dir, exist_ok=True)
        render_pass(output_pair_dir, cam, idx, args.asset_folder)


if __name__ == "__main__":
    main()