import numpy as np
import pybullet as p 
import nvisii as nv
from scene_utils import update_visual_objects, get_rotation
from collect_scenes import TabletopScenes

opt = lambda : None
opt.scene_type = 'line' # 'random' or 'line'
opt.spp = 32 #64 
opt.width = 500
opt.height = 500 
opt.noise = False
opt.dataset = 'train' #'train' or 'test'
opt.objectset = 'ycb' #'pybullet'/'ycb'/'all'
opt.pybullet_object_path = '/home/gun/Desktop/pybullet-URDF-models/urdf_models/models'
opt.ycb_object_path = '/home/gun/ssd/disk/YCB_dataset'

ts = TabletopScenes(opt)
urdf_ids = sorted(ts.urdf_id_names.keys())
for i in range(5):
    urdf_selected = urdf_ids[20 * i:20 * (i+1)]
    ts.spawn_objects(urdf_selected)

    for idx, obj_col_id in enumerate(ts.current_pybullet_ids):
        eye = [ts.xx[idx]+0.5, ts.yy[idx], 1.2]
        at = [ts.xx[idx], ts.yy[idx], 0]
        ts.set_camera_pose(eye=eye, at=at)

        uid = urdf_selected[idx]
        object_name = ts.urdf_id_names[uid]
        object_type, object_index = uid.split('-')
        while True:
            pos_new = [ts.xx[idx], ts.yy[idx], 0.5]
            if uid in ts.init_euler:
                print('init euler:', ts.init_euler[uid])
                roll, pitch, yaw = np.array(ts.init_euler[uid]) * np.pi / 2
            else:
                print(uid, 'not in init_euler.')
                roll, pitch, yaw = 0, 0, 0
            rot_new = get_rotation(roll, pitch, yaw)
            p.resetBasePositionAndOrientation(obj_col_id, pos_new, rot_new)
            
            for j in range(500):
                p.stepSimulation()
                if j%50==0:
                    nv.ids = update_visual_objects(ts.current_pybullet_ids, "", nv.ids)
                    nv.render(int(opt.width), int(opt.height), int(opt.spp))

            x = input("Set new euler values or press OK to move on to the next object.\n")
            if x.lower()=="x":
                exit()
            elif x.lower()=="ok":
                break
            else:
                if len(x.split(','))==3:
                    euler = [float(e) for e in x.split(',')]
                    ts.init_euler[uid] = euler
                continue
            
    ts.clear()
ts.close()
