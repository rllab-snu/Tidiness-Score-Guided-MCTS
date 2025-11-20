import os 
import copy
import nvisii as nv
import random
import colorsys
import subprocess 
import math
import pybullet as p 
import numpy as np
from matplotlib import pyplot as plt
from transform_utils import euler2quat, mat2quat, quat2mat
from scene_utils import get_init_euler, generate_scene
from scene_utils import get_rotation, get_contact_objects, get_velocity
from scene_utils import update_visual_objects 
from scene_utils import remove_visual_objects, clear_scene

class TabletopScenes(object):
    def __init__(self, opt):
        self.opt = opt

        # show an interactive window, and use "lazy" updates for faster object creation time 
        nv.initialize(headless=False, lazy_updates=True)

        # Setup bullet physics stuff
        physicsClient = p.connect(p.GUI) # non-graphical version

        # Create a camera
        self.camera = None
        self.set_camera_pose(eye=(0.6, 0, 1.2)) #(3, 0, 6))
        self.set_grid()

        self.initialize_nvisii_scene()
        self.initialize_pybullet_scene()
        self.init_euler = get_init_euler()
        self.urdf_id_names = self.load_objects(opt.dataset, opt.objectset)

        self.threshold = {'pose': 0.07,
                          'rotation': 0.15,
                          'linear': 0.003,
                          'angular': 0.003}
        self.pre_selected_objects = []
        self.current_pybullet_ids = []


    def set_camera_pose(self, eye, at=(0.1, 0, 0), up=(0, 0, 1)):
        if self.camera is None:
            self.camera = nv.entity.create(
                name = "camera",
                transform = nv.transform.create("camera"),
                camera = nv.camera.create_from_fov(
                    name = "camera", field_of_view = 0.85,
                    aspect = float(self.opt.width)/float(self.opt.height)
                ))

        self.camera.get_transform().look_at(at=at, up=up, eye=eye)
        nv.set_camera_entity(self.camera)
        return


    def initialize_nvisii_scene(self):
        if not self.opt.noise is True: 
            nv.enable_denoiser()

        # Change the dome light intensity
        nv.set_dome_light_intensity(1.0)

        # atmospheric thickness makes the sky go orange, almost like a sunset
        nv.set_dome_light_sky(sun_position=(6,6,6), atmosphere_thickness=1.0, saturation=1.0)

        # add a sun light
        sun = nv.entity.create(
            name = "sun",
            mesh = nv.mesh.create_sphere("sphere"),
            transform = nv.transform.create("sun"),
            light = nv.light.create("sun")
        )
        sun.get_transform().set_position((6,6,6))
        sun.get_light().set_temperature(5780)
        sun.get_light().set_intensity(1000)

        floor = nv.entity.create(
            name="floor",
            mesh = nv.mesh.create_plane("floor"),
            transform = nv.transform.create("floor"),
            material = nv.material.create("floor")
        )
        floor.get_transform().set_position((0,0,0))
        floor.get_transform().set_scale((1, 1, 1)) #(6, 6, 6)
        floor.get_material().set_roughness(0.1)
        floor.get_material().set_base_color((0.8, 0.87, 0.88)) #(0.5,0.5,0.5)

        floor_textures = []
        texture_files = os.listdir("texture")
        texture_files = [f for f in texture_files if f.lower().endswith('.png')]

        for i, tf in enumerate(texture_files):
            tex = nv.texture.create_from_file("tex-%d"%i, os.path.join("texture/", tf))
            floor_tex = nv.texture.create_hsv("floor-%d"%i, tex, hue=0, saturation=.5, value=1.0, mix=1.0)
            floor_textures.append((tex, floor_tex))
        self.floor, self.floor_textures = floor, floor_textures
        return

    def initialize_pybullet_scene(self):
        # reset pybullet #
        p.resetSimulation()
        p.setGravity(0,0,-10)

        # Set the collision with the floor mesh
        # first lets get the vertices 
        vertices = self.floor.get_mesh().get_vertices()

        # get the position of the object
        pos = self.floor.get_transform().get_position()
        pos = [pos[0],pos[1],pos[2]]
        scale = self.floor.get_transform().get_scale()
        scale = [scale[0],scale[1],scale[2]]
        rot = self.floor.get_transform().get_rotation()
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
        return

    def set_grid(self):
        x = np.linspace(-0.8, 0.8, 5)
        y = np.linspace(-0.8, 0.8, 5)
        xx, yy = np.meshgrid(x, y, sparse=False)
        self.xx = xx.reshape(-1)
        self.yy = yy.reshape(-1)

    def clear(self):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        # remove spawned objects before #
        for idx, urdf_id in enumerate(self.spawned_objects):
            obj_col_id = pybullet_ids[idx]
            p.removeBody(obj_col_id)
        remove_visual_objects(nv.ids)
        #clear_scene()
        self.spawned_objects = None
        self.pre_selected_objects = []
        self.current_pybullet_ids = []

    def close(self):
        nv.deinitialize()
        p.disconnect()

    def load_objects(self, dataset, objectset):
        # lets create a bunch of objects 
        pybullet_object_path = self.opt.pybullet_object_path
        pybullet_object_names = sorted([m for m in os.listdir(pybullet_object_path) \
                            if os.path.isdir(os.path.join(pybullet_object_path, m))])
        ycb_object_path = self.opt.ycb_object_path
        ycb_object_names = sorted([m for m in os.listdir(ycb_object_path) \
                            if os.path.isdir(os.path.join(ycb_object_path, m))])
        exclusion_list = ['047_plastic_nut', '063-b_marbles', '063-c_marbles', '063-f_marbles', '072-g_toy_airplane']
        for eo in exclusion_list:
            ycb_object_names.remove(eo)

        pybullet_split = 80
        pybullet_ids = ['pybullet-%d'%p for p in range(len(pybullet_object_names))]
        ycb_split = 80
        ycb_ids = ['ycb-%d'%y for y in range(len(ycb_object_names))]
        # urdf_id_names
        # key: {object_type}-{index} - e.g., 'pybullet-0'
        # value: {object_name} - e.g., 'black_marker'
        if dataset=='train':
            if objectset=='pybullet':
                urdf_id_names = dict(zip(
                    pybullet_ids[:pybullet_split], 
                    pybullet_object_names[:pybullet_split]
                    ))
            elif objectset=='ycb':
                urdf_id_names = dict(zip(
                    ycb_ids[:ycb_split], 
                    ycb_object_names[:ycb_split]
                    ))
            elif objectset=='all':
                urdf_id_names = dict(zip(
                    pybullet_ids[:pybullet_split] + ycb_ids[:ycb_split],
                    pybullet_object_names[:pybullet_split] + ycb_object_names[:ycb_split]
                    ))
        elif dataset=='test':
            if objectset=='pybullet':
                urdf_id_names = dict(zip(
                    pybullet_ids[pybullet_split:], 
                    pybullet_object_names[pybullet_split:]
                    ))
            elif objectset=='ycb':
                urdf_id_names = dict(zip(
                    ycb_ids[ycb_split:], 
                    ycb_object_names[ycb_split:]
                    ))
            elif objectset=='all':
                urdf_id_names = dict(zip(
                    pybullet_ids[pybullet_split:] + ycb_ids[ycb_split:],
                    pybullet_object_names[pybullet_split:] + ycb_object_names[ycb_split:]
                    ))
        print('-'*60)
        print(len(urdf_id_names), 'objects can be loaded.')
        print('-'*60)
        return urdf_id_names

    def select_objects(self, nb_objects):
        nb_spawn = min(nb_objects, len(self.urdf_id_names))
        urdf_selected = np.random.choice(list(self.urdf_id_names.keys()), nb_spawn, replace=False)
        return urdf_selected

    def spawn_objects(self, urdf_selected):
        self.spawned_objects = copy.deepcopy(urdf_selected)

        pybullet_ids = []
        for idx, urdf_id in enumerate(urdf_selected):
            object_type = urdf_id.split('-')[0]
            object_name = self.urdf_id_names[urdf_id]
            #(object_name, object_type) = self.urdf_id_names[urdf_id]
            if object_type=='pybullet':
                urdf_path = os.path.join(self.opt.pybullet_object_path, object_name, 'model.urdf')
            else:
                urdf_path = os.path.join(self.opt.ycb_object_path, object_name, 'poisson', 'model.urdf')
                if object_name.startswith('022_windex_bottle') or object_name.startswith('023_wine_glass') or object_name.startswith('049_'):
                    urdf_path = os.path.join(self.opt.ycb_object_path, object_name, 'tsdf', 'model.urdf')
                else:
                    urdf_path = os.path.join(self.opt.ycb_object_path, object_name, 'poisson', 'model.urdf')
            obj_col_id = p.loadURDF(urdf_path, [self.xx[idx], self.yy[idx], 0.15], globalScaling=1.) #5.
            pybullet_ids.append(obj_col_id)

        nv.ids = update_visual_objects(pybullet_ids, "")
        self.pre_selected_objects = copy.deepcopy(pybullet_ids)
        self.current_pybullet_ids = copy.deepcopy(pybullet_ids)

    def set_floor(self, texture_id=-1):
        # set floor material #
        roughness = random.uniform(0.1, 0.5)
        self.floor.get_material().clear_base_color_texture()
        self.floor.get_material().set_roughness(roughness)

        if texture_id==-1: # random texture #
            f_cidx = np.random.choice(len(self.floor_textures))
            tex, floor_tex = self.floor_textures[f_cidx]
        else:
            tex, floor_tex = self.floor_textures[texture_id]
        self.floor.get_material().set_base_color_texture(floor_tex)
        self.floor.get_material().set_roughness_texture(tex)

    def arrange_objects(self, scene_idx, urdf_selected):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)

        # set objects #
        count_scene_trials = 0
        selected_objects = np.random.choice(pybullet_ids, self.opt.inscene_objects, replace=False)
        while True:
            # generate scene in a 'line' shape #
            init_positions = generate_scene(self.opt.scene_type, self.opt.inscene_objects)
            init_rotations = []
            for idx, urdf_id in enumerate(urdf_selected):
                obj_col_id = pybullet_ids[idx]
                # hide objects placed on the table #
                if obj_col_id in self.pre_selected_objects:
                    pos_hidden = [self.xx[idx], self.yy[idx], -1]
                    p.resetBasePositionAndOrientation(obj_col_id, pos_hidden, [0, 0, 0, 1])

                # place new objects #
                if obj_col_id in selected_objects:
                    sidx = np.where(selected_objects==obj_col_id)[0][0]
                    pos_sel = init_positions[sidx]
                    roll, pitch, yaw = 0, 0, 0
                    if urdf_id in self.init_euler:
                        roll, pitch, yaw = np.array(self.init_euler[urdf_id]) * np.pi / 2
                    rot = get_rotation(roll, pitch, yaw)
                    init_rotations.append(rot)
                    p.resetBasePositionAndOrientation(obj_col_id, pos_sel, rot)
            self.pre_selected_objects = []
            init_rotations = np.array(init_rotations)

            # check feasibility #
            init_feasible = False 
            j = 0
            while j<2000 and not init_feasible:
                p.stepSimulation()
                if j%10==0:
                    current_poses = []
                    current_rotations = []
                    for obj_col_id in selected_objects:
                        pos, rot = p.getBasePositionAndOrientation(obj_col_id)
                        current_poses.append(pos)
                        current_rotations.append(rot)

                    current_poses = np.array(current_poses)
                    current_rotations = np.array(current_rotations)
                    pos_diff = np.linalg.norm(current_poses[:, :2] - init_positions[:, :2], axis=1)
                    rot_diff = np.linalg.norm(current_rotations - init_rotations, axis=1)
                    if (pos_diff > self.threshold['pose']).any() or (rot_diff > self.threshold['rotation']).any():
                        break
                    vel_linear, vel_rot = get_velocity(selected_objects)
                    stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                    stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                    if stop_linear and stop_rotation:
                        init_feasible = True
                j += 1
            if init_feasible:
                break
            count_scene_trials += 1
            if count_scene_trials > 5:
                break

        self.pre_selected_objects = copy.deepcopy(selected_objects)
        # if failed to place objects robustly #
        if not init_feasible:
            return False
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        self.render_and_save_scene(selected_objects, scene_idx)
        return True


    def messup_objects(self, obj_idx, scene_idx, urdf_selected):
        pybullet_ids = copy.deepcopy(self.current_pybullet_ids)
        selected_objects = copy.deepcopy(self.pre_selected_objects)

        # save current poses & rots #
        pos_saved, rot_saved = {}, {}
        for obj_col_id in selected_objects:
            pos, rot = p.getBasePositionAndOrientation(obj_col_id)
            pos_saved[obj_col_id] = pos
            rot_saved[obj_col_id] = rot

        # set poses & rots #
        target_col_id = selected_objects[obj_idx]
        idx = pybullet_ids.index(target_col_id)
        urdf_id = urdf_selected[idx]
        place_feasible = False
        count_scene_repose = 0
        while not place_feasible:
            # get the pose of the objects
            pos, rot = p.getBasePositionAndOrientation(target_col_id)
            collisions_before = get_contact_objects()

            pos_new = 0.9*(np.random.rand(3) - 0.5)
            pos_new[2] = 0.15
            roll, pitch, yaw = 0, 0, 0
            if urdf_id in self.init_euler:
                roll, pitch, yaw = np.array(self.init_euler[urdf_id]) * np.pi / 2
            rot = get_rotation(roll, pitch, yaw)
            p.resetBasePositionAndOrientation(target_col_id, pos_new, rot)
            collisions_after = set()
            for _ in range(200):
                p.stepSimulation()
                collisions_after = collisions_after.union(get_contact_objects())

            collisions_new = collisions_after - collisions_before
            if len(collisions_new) > 0:
                place_feasible = False

                # reset non-target objects
                obj_to_reset = set()
                for collision in collisions_new:
                    obj1, obj2 = collision
                    obj_to_reset.add(obj1)
                    obj_to_reset.add(obj2)
                obj_to_reset = obj_to_reset - set([target_col_id])
                for reset_col_id in obj_to_reset:
                    p.resetBasePositionAndOrientation(reset_col_id, pos_saved[reset_col_id], rot_saved[reset_col_id])
            else:
                place_feasible = True
                pos, rot = p.getBasePositionAndOrientation(target_col_id)
                pos_saved[target_col_id] = pos
                rot_saved[target_col_id] = rot
            count_scene_repose += 1
            if count_scene_repose > 10:
                break

        if not place_feasible:
            return False

        for j in range(2000):
            p.stepSimulation()
            if j%10==0:
                vel_linear, vel_rot = get_velocity(selected_objects)
                stop_linear = (np.linalg.norm(vel_linear) < self.threshold['linear'])
                stop_rotation = (np.linalg.norm(vel_rot) < self.threshold['angular'])
                if stop_linear and stop_rotation:
                    break
        nv.ids = update_visual_objects(pybullet_ids, "", nv.ids)
        self.render_and_save_scene(selected_objects, scene_idx)
        return True

    def render_and_save_scene(self, selected_objects, scene_idx):
        nv.render_to_file(
            width=int(self.opt.width), height=int(self.opt.height), 
            samples_per_pixel=int(self.opt.spp),
            file_path=f"{self.opt.outf}/rgb_{str(scene_idx).zfill(5)}.png"
        )
        d = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=5, bounce=0, options='depth',
        )
        depth = np.array(d).reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        np.save(f"{self.opt.outf}/depth_{str(scene_idx).zfill(5)}.npy", depth)
        entity_id = nv.render_data(
            width=int(self.opt.width), height=int(self.opt.height),
            start_frame=0, frame_count=5, bounce=0, options='entity_id',
        )
        entity = np.array(entity_id)
        entity = entity.reshape([int(self.opt.height), int(self.opt.width), -1])[:, :, 0]
        segmap = np.zeros_like(entity)
        for s, obj in enumerate(sorted(selected_objects)):
            segmap[entity == (obj + 2)] = s + 1
        np.save(f"{self.opt.outf}/seg_{str(scene_idx).zfill(5)}.npy", segmap)
        return


if __name__=='__main__':
    opt = lambda : None
    opt.nb_objects = 12 #20
    opt.inscene_objects = 4 #5
    opt.scene_type = 'line' # 'random' or 'line'
    opt.spp = 32 #64 
    opt.width = 500
    opt.height = 500 
    opt.noise = False
    opt.nb_scenes = 1000
    opt.nb_frames = 5
    opt.outf = '/home/gun/ssd/disk/ur5_tidying_data/line-shape/images'
    opt.nb_randomset = 20
    opt.dataset = 'train' #'train' or 'test'
    opt.objectset = 'pybullet' #'pybullet'/'ycb'/'all'
    opt.pybullet_object_path = '/home/gun/Desktop/pybullet-URDF-models/urdf_models/models'
    opt.ycb_object_path = '/home/gun/ssd/disk/YCB_dataset'

    if os.path.isdir(opt.outf):
        print(f'folder {opt.outf}/ exists')
    else:
        os.makedirs(opt.outf)
        print(f'created folder {opt.outf}/')

    ts = TabletopScenes(opt)
    for nset in range(opt.nb_randomset):
        urdf_selected = ts.select_objects(opt.nb_objects)
        ts.spawn_objects(urdf_selected)
        num_exist_frames = len([f for f in os.listdir(f"{opt.outf}") if '.png' in f])
        ns = 0
        while ns < opt.nb_scenes:
            ts.set_floor(texture_id=-1)

            # 1. Spawn objects in a 'Line' shape #
            nf = 0
            print(f'rendering scene {str(nset)}-{str(ns).zfill(5)}-{str(nf)}', end='\r')
            scene_idx = num_exist_frames + ns * opt.nb_frames + nf
            success_placement = ts.arrange_objects(scene_idx, urdf_selected)
            if not success_placement:
                continue

            # 2. Move each object to a random place #
            nf = 1
            while nf < int(opt.nb_frames):
                print(f'rendering scene {str(nset)}-{str(ns).zfill(5)}-{str(nf)}', end='\r')
                scene_idx = num_exist_frames + ns * opt.nb_frames + nf
                success_placement = ts.messup_objects(nf-1, scene_idx, urdf_selected)
                if not success_placement:
                    continue
                nf += 1
            ns += 1
        ts.clear()
    ts.close()

    
