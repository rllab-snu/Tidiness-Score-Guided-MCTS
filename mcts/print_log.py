import os
import json
import numpy as np

logs = os.listdir('data/')
for log in sorted(logs):
    if log.startswith('classification'): continue
    log_path = os.path.join('data/', log)
    print('-'*30)
    print(log)
    print('\tSR\tScore\tEplen')
    successes, scores, eplens = [], [], []
    for scene in os.listdir(log_path):
        scene_path = os.path.join(log_path, scene)
        try:
            data = json.load(open(os.path.join(scene_path, 'perform_data.json'), 'r'))
            #print(scene, data)
            print('%s\t%.2f\t%.3f\t%.3f' %(scene, data['success'], data['score'], data['eplen']))
            successes.append(data['success'])
            scores.append(data['score'])
            eplens.append(data['eplen'])
        except:
            print("%s:\t%d"%(scene,len(os.listdir(scene_path))-1))
            continue
    if len(successes)!=0:
        print('%s\t%.2f\t%.3f\t%.3f' %('AVG', np.mean(successes), np.mean(scores), np.mean(eplens)))
print('-'*30)
