import pandas as pd

scenes = {}
scenes_list = ['dining_table', 'office_table', 'workbench', 'tea_table']

objects_in_scene = {}
objects_in_scene['dining_table'] = '''fork, spoon, knife, bread_knife, 
plate, bread_plate, salad_plate, soup_bowl, 
water_glass, wine_glass, 
bread, wine, salt_shaker, pepper_shaker'''
objects_in_scene['office_table'] = '''laptop, stand, pen_stand,
pen, scissors, book,
mugcup, coffee, tumbler,
tissue_box, table_clock, calander,
tabletop_bookshelf, flower_pot'''
objects_in_scene['workbench'] = '''hammer, driver, wrench, pliers,
power_drill, spring_clamp,
tape_measure, duct_tape '''
objects_in_scene['tea_table'] = '''tea_spoon, tea_pot, teacup, saucer,
magazine, vase, wood_tray,
apple, bowl'''
for k in objects_in_scene:
    objects_in_scene[k] = objects_in_scene[k].replace('\n', ' ').replace(' ', '').split(',')

object_models = {}
obj_pybullet = pd.read_csv('pybullet.csv', index_col=0)
obj_ycb = pd.read_csv('ycb.csv', index_col=0)
for o in range(len(obj_pybullet)):
    obj = obj_pybullet.iloc[o].name
    if not obj in object_models:
        object_models[obj] = []
    object_models[obj] += ['pybullet/%s'%m for m in obj_pybullet.iloc[o].values if not pd.isna(m)]

for o in range(len(obj_ycb)):
    obj = obj_ycb.iloc[o].name
    if not obj in object_models:
        object_models[obj] = []
    object_models[obj] += ['ycb/%s'%m for m in obj_pybullet.iloc[o].values if not pd.isna(m)]

print(object_models)
exit()

for s in scenes_list:
    scenes[s] = dict(map(lambda o: (objects_in_scene[s][o], []), range(len(objects_in_scene[s]))))
#TODO

pybullet_model['fork'] = ['fork']
pybullet_model['spoon'] = ['spoon']
pybullet_model['knife'] = ['knife']
pybullet_model['bread_knife'] = ['spoon']
pybullet_model['bread'] = ['spoon']
pybullet_model['plate'] = ['spoon']
pybullet_model['bread_plate'] = ['spoon']
pybullet_model['salad_plate'] = ['spoon']
pybullet_model['soup_bowl'] = ['spoon']
pybullet_model['spoon'] = ['spoon']
pybullet_model['spoon'] = ['spoon']
pybullet_model['spoon'] = ['spoon']

obj_model['fork']['pybullet'] = ['fork']
obj_model['fork']['ycb'] = ['030_fork']
obj_model['spoon']['pybullet'] = ['spoon']
obj_model['spoon']['ycb'] = ['031_spoon']



scenes['dining table'] = {}
scenes['diining table']['pybullet'] = [



        ]

scenes['dining_table']['fork']['pybullet'] = []
