import os 
import nvisii as nv
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import numpy as np
from matplotlib import pyplot as plt
from transform_utils import euler2quat, mat2quat, quat2mat
from scene_utils import init_euler, generate_scene_at_center
from scene_utils import get_rotation, get_contact_objects, get_velocity
from scene_utils import update_visual_objects 
from scene_utils import remove_visual_objects, clear_scene

opt = lambda : None
opt.nb_objects = 12 #20
opt.inscene_objects = 4 #5
opt.scene_type = 'line' # 'random' or 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.nb_scenes = 1 #1000 #2500 #25
opt.nb_frames = 100 #50
opt.outf = '/home/gun/ssd/disk/ur5_tidying_data/videos5'
opt.nb_randomset = 50 #20
opt.obj = 'train' #'train' or 'test'
opt.dataset = 'pybullet' # 'pybullet', 'ycb', 'all'


# # # # # # # # # # # # # # # # # # # # # # # # #
if os.path.isdir(opt.outf):
    print(f'folder {opt.outf}/ exists')
else:
    os.makedirs(opt.outf)
    print(f'created folder {opt.outf}/')
# # # # # # # # # # # # # # # # # # # # # # # # #

# show an interactive window, and use "lazy" updates for faster object creation time 
nv.initialize(headless=False, lazy_updates=True)

# Setup bullet physics stuff
physicsClient = p.connect(p.GUI) # non-graphical version

def get_bounding_box(segmap):
    return

def set_camera_pose(camera, eye, at=(0.5, 0, 0), up=(0, 0, 1)):
    camera.get_transform().look_at(
        at = at,
        up = up,
        eye = eye
        )
    nv.set_camera_entity(camera)

def initialize_nvisii_scene():
    if not opt.noise is True: 
        nv.enable_denoiser()

    # Lets set the scene
    # Change the dome light intensity
    nv.set_dome_light_intensity(1.0)

    # atmospheric thickness makes the sky go orange, almost like a sunset
    nv.set_dome_light_sky(sun_position=(10,10,10), atmosphere_thickness=1.0, saturation=1.0)

    # add a sun light
    sun = nv.entity.create(
        name = "sun",
        mesh = nv.mesh.create_sphere("sphere"),
        transform = nv.transform.create("sun"),
        light = nv.light.create("sun")
    )
    sun.get_transform().set_position((10,10,10))
    sun.get_light().set_temperature(5780)
    sun.get_light().set_intensity(1000)

    floor = nv.entity.create(
        name="floor",
        mesh = nv.mesh.create_plane("floor"),
        transform = nv.transform.create("floor"),
        material = nv.material.create("floor")
    )
    floor.get_transform().set_position((0,0,0))
    floor.get_transform().set_scale((8, 8, 8)) #6, 6, 6
    floor.get_material().set_roughness(0.1)
    floor.get_material().set_base_color((0.8, 0.87, 0.88)) #(0.5,0.5,0.5)

    floor_textures = []
    # texture_files = os.listdir("texture")
    # texture_files = [f for f in texture_files if f.lower().endswith('.png')]
    texture_files = ['wood_table.png'] #['marmite.png']
    for i, tf in enumerate(texture_files):
        tex = nv.texture.create_from_file("tex-%d"%i, os.path.join("texture/", tf))
        floor_tex = nv.texture.create_hsv("floor-%d"%i, tex, hue=0, saturation=.5, value=1.0, mix=1.0)
        floor_textures.append((tex, floor_tex))

    # reset pybullet #
    p.resetSimulation()
    p.setGravity(0,0,-10)

    # Set the collision with the floor mesh
    # first lets get the vertices 
    vertices = floor.get_mesh().get_vertices()

    # get the position of the object
    pos = floor.get_transform().get_position()
    pos = [pos[0],pos[1],pos[2]]
    scale = floor.get_transform().get_scale()
    scale = [scale[0],scale[1],scale[2]]
    rot = floor.get_transform().get_rotation()
    rot = [rot[0],rot[1],rot[2],rot[3]]

    # create a collision shape that is a convex hull
    obj_col_id = p.createCollisionShape(
        p.GEOM_MESH,
        vertices = vertices,
        meshScale = scale,
    )

    # create a body without mass so it is static
    p.createMultiBody(
        baseCollisionShapeIndex = obj_col_id,
        basePosition = pos,
        baseOrientation= rot,
    )    
    return floor, floor_textures

floor, floor_textures = initialize_nvisii_scene()

# Create a camera
camera = nv.entity.create(
    name = "camera",
    transform = nv.transform.create("camera"),
    camera = nv.camera.create_from_fov(
        name = "camera", 
        field_of_view = 0.85,
        aspect = float(opt.width)/float(opt.height)
    )
)
eye = np.array([3, 0, 6])
set_camera_pose(camera, eye=eye)
#camera.get_transform().look_at(
#    at = (0.5,0,0),
#    up = (0,0,1),
#    eye = (3, 0, 6), #(4, 0, 6), #(10,0,4),
#)
#nv.set_camera_entity(camera)

# lets create a bunch of objects 
pybullet_object_path = '/home/gun/Desktop/pybullet-URDF-models/urdf_models/models'
pybullet_object_names = sorted([m for m in os.listdir(pybullet_object_path) \
                    if os.path.isdir(os.path.join(pybullet_object_path, m))])
ycb_object_path = '/home/gun/ssd/disk/YCB_dataset'
ycb_object_names = sorted([m for m in os.listdir(ycb_object_path) \
                    if os.path.isdir(os.path.join(ycb_object_path, m))])
exclusion_list = ['047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane']
for eo in exclusion_list:
    ycb_object_names.remove(eo)

pybullet_train_objects = pybullet_object_names[:80]
pybullet_test_objects = pybullet_object_names[80:]
ycb_train_objects = ycb_object_names[:80]
ycb_test_objects = ycb_object_names[80:]
if opt.obj=='train':
    if opt.dataset=='pybullet':
        urdf_id_names = dict(zip(
            range(len(pybullet_train_objects)), 
            zip(pybullet_train_objects, ['pybullet'] * len(pybullet_train_objects))
            ))
    elif opt.dataset=='ycb':
        urdf_id_names = dict(zip(
            range(len(ycb_train_objects)), 
            zip(ycb_train_objects, ['ycb'] * len(ycb_train_objects))
            ))
    elif opt.dataset=='all':
        urdf_id_names = dict(zip(
            range(len(pybullet_train_objects)+len(ycb_train_objects)), 
            zip(pybullet_train_objects + ycb_train_objects, 
                ['pybullet'] * len(pybullet_train_objects) + ['ycb'] * len(ycb_train_objects)
                )
            ))
elif opt.obj=='test':
    if opt.dataset=='pybullet':
        urdf_id_names = dict(zip(
            range(len(pybullet_test_objects)), 
            zip(pybullet_test_objects, ['pybullet'] * len(pybullet_test_objects))
            ))
    elif opt.dataset=='ycb':
        urdf_id_names = dict(zip(
            range(len(ycb_test_objects)), 
            zip(ycb_test_objects, ['ycb'] * len(ycb_test_objects))
            ))
    elif opt.dataset=='all':
        urdf_id_names = dict(zip(
            range(len(pybullet_test_objects)+len(ycb_test_objects)), 
            zip(pybullet_test_objects + ycb_test_objects, 
                ['pybullet'] * len(pybullet_test_objects) + ['ycb'] * len(ycb_test_objects)
                )
            ))
print('-'*60)
print(len(urdf_id_names), 'objects can be loaded.')
print('-'*60)

pre_spawned_objects = None
for nset in range(opt.nb_randomset):
    # remove spawned objects before #
    if not pre_spawned_objects is None:
        for idx, urdf_id in enumerate(pre_spawned_objects):
            obj_col_id = pybullet_ids[idx]
            p.removeBody(obj_col_id)
        #remove_visual_objects(nv.ids)
        clear_scene()
        floor, floor_textures = initialize_nvisii_scene()

    nb_spawn = min(opt.nb_objects, len(urdf_id_names))
    urdf_selected = np.random.choice(list(urdf_id_names.keys()), nb_spawn, replace=False)
    pre_spawned_objects = urdf_selected

    x = np.linspace(-4, 4, 6)
    y = np.linspace(-4, 4, 6)
    xx, yy = np.meshgrid(x, y, sparse=False)
    xx = xx.reshape(-1)
    yy = yy.reshape(-1)

    pybullet_ids = []
    for idx, urdf_id in enumerate(urdf_selected):
        (object_name, object_type) = urdf_id_names[urdf_id]
        if object_type=='pybullet':
            urdf_path = os.path.join(pybullet_object_path, object_name, 'model.urdf')
        else:
            if object_name.startswith('022_windex_bottle') or object_name.startswith('023_wine_glass') or object_name.startswith('049_'):
                urdf_path = os.path.join(ycb_object_path, object_name, 'tsdf', 'model.urdf')
            else:
                urdf_path = os.path.join(ycb_object_path, object_name, 'poisson', 'model.urdf')
        obj_col_id = p.loadURDF(urdf_path, [xx[idx], yy[idx], 0.5], globalScaling=5.)
        pybullet_ids.append(obj_col_id)
    nv.ids = update_visual_objects(pybullet_ids, "")

    threshold_pose = 0.07
    threshold_rotation = 0.15
    threshold_linear = 0.003
    threshold_angular = 0.003
    pre_selected_objects = pybullet_ids 

    # Lets run the simulation for a few steps. 
    num_exist_frames = len([f for f in os.listdir(f"{opt.outf}") if '.png' in f])
    ns = 0
    while ns < opt.nb_scenes:
        # set floor material #
        roughness = random.uniform(0.1, 0.5)
        floor.get_material().clear_base_color_texture()
        floor.get_material().set_roughness(roughness)

        f_cidx = np.random.choice(len(floor_textures))
        tex, floor_tex = floor_textures[f_cidx]
        floor.get_material().set_base_color_texture(floor_tex)
        floor.get_material().set_roughness_texture(tex)

        # set objects #
        count_scene_trials = 0
        selected_objects = np.random.choice(pybullet_ids, opt.inscene_objects, replace=False)
        while True:
            init_positions = generate_scene_at_center(opt.scene_type, opt.inscene_objects)
            init_rotations = []
            for idx, urdf_id in enumerate(urdf_selected):
                obj_col_id = pybullet_ids[idx]
                if obj_col_id in pre_selected_objects:
                    pos_hidden = [xx[idx], yy[idx], -1]
                    p.resetBasePositionAndOrientation(obj_col_id, pos_hidden, [0, 0, 0, 1])

                if obj_col_id in selected_objects:
                    sidx = np.where(selected_objects==obj_col_id)[0][0]
                    pos_sel = init_positions[sidx]
                    roll, pitch, yaw = 0, 0, 0
                    if urdf_id in init_euler:
                        roll, pitch, yaw = np.array(init_euler[urdf_id]) * np.pi / 2
                    rot = get_rotation(roll, pitch, yaw)
                    init_rotations.append(rot)
                    p.resetBasePositionAndOrientation(obj_col_id, pos_sel, rot)
            init_rotations = np.array(init_rotations)

            print('init:', init_positions)
            init_feasible = True
            for j in range(2000):
                p.stepSimulation()
                if j%10==0:
                    current_poses = []
                    current_rotations = []
                    for idx, urdf_id in enumerate(urdf_selected):
                        obj_col_id = pybullet_ids[idx]
                        if obj_col_id not in selected_objects:
                            continue
                        pos, rot = p.getBasePositionAndOrientation(obj_col_id)
                        current_poses.append(pos)
                        current_rotations.append(rot)

                    current_poses = np.array(current_poses)
                    current_rotations = np.array(current_rotations)
                    #print('current:', current_poses)
                    pos_diff = np.linalg.norm(current_poses[:, :2] - init_positions[:, :2], axis=1)
                    rot_diff = np.linalg.norm(current_rotations - init_rotations, axis=1)
                    #print()
                    if (pos_diff > threshold_pose).any() or (rot_diff > threshold_rotation).any():
                        init_feasible = False
                        break
                    vel_linear, vel_rot = get_velocity(selected_objects)
                    stop_linear = (np.linalg.norm(vel_linear) < threshold_linear)
                    stop_rotation = (np.linalg.norm(vel_rot) < threshold_angular)
                    if stop_linear and stop_rotation:
                        break
            if j==1999:
                init_feasible = False
            if init_feasible:
                break
            count_scene_trials += 1
            if count_scene_trials > 5:
                break

        if not init_feasible: #j==1999: 
            pre_selected_objects = selected_objects
            continue
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)

        for nf in range(opt.nb_frames):
            if nf < opt.nb_frames/2:
                theta = 2 * np.pi * (nf / opt.nb_frames)
            else:
                theta = 2 * np.pi * (1 - nf / opt.nb_frames)
            radius = 3 * np.cos(nf / opt.nb_frames * np.pi / 2)
            #radius = 3 * np.cos(theta/4) #(1 - nf / opt.nb_frames)
            eye = np.array([
                        radius * np.cos(theta), 
                        radius * np.sin(theta), 
                        5 + radius/2
                        ])
            #eye = np.array([3 * np.cos(theta), 3 * np.sin(theta), 6 + np.sin(3 * theta)])
            set_camera_pose(camera, eye=eye, at=(0, 0, 0))
            print(f'rendering scene {str(nset)}-{str(ns).zfill(5)}-{str(nf)}', end='\r')
            nv.render_to_file(
                width=int(opt.width), 
                height=int(opt.height), 
                samples_per_pixel=int(opt.spp),
                file_path=f"{opt.outf}/rgb_{str(num_exist_frames + ns * opt.nb_frames + nf).zfill(5)}.png"
            )
            d = nv.render_data(
                width=int(opt.width),
                height=int(opt.height),
                start_frame=0,
                frame_count=5,
                bounce=0,
                options='depth',
            )
            depth = np.array(d).reshape([int(opt.height), int(opt.width), -1])[:, :, 0]
            np.save(f"{opt.outf}/depth_{str(num_exist_frames + ns * opt.nb_frames + nf).zfill(5)}.npy", depth)
            entity_id = nv.render_data(
                width=int(opt.width),
                height=int(opt.height),
                start_frame=0,
                frame_count=5,
                bounce=0,
                options='entity_id',
            )
            entity = np.array(entity_id).reshape([int(opt.height), int(opt.width), -1])[:,:,0]
            segmap = np.zeros_like(entity)
            for s, obj in enumerate(sorted(selected_objects)):
                segmap[entity==(obj+2)] = s+1
            np.save(f"{opt.outf}/seg_{str(num_exist_frames + ns * opt.nb_frames + nf).zfill(5)}.npy", segmap)
        pre_selected_objects = selected_objects
        ns += 1

p.disconnect()
nv.deinitialize()

