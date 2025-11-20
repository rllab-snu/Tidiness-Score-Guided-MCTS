import os
import json
import numpy as np
from dateutil import parser
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--threshold', type=float, default=0.9)
argparser.add_argument('--after', type=str, default=None)
args = argparser.parse_args()

def filter_log_time(log):
    success = False
    times = [parser.parse(log[i-1].split(',')[0]) for i in range(len(log)) if 'Best Action' in log[i]]
    return times
def get_intervals(times):
    delta_time = []
    for i in range(len(times)-1):
        delta = times[i+1] - times[i]
        delta_time.append(delta)
    return delta_time
def filter_log(log):
    success = False
    success_score = 0
    log = [float(t.replace('\n', '').split(' ')[-1]) for t in log if 'Score' in t]
    if len(log)<2:
        return None
    if log[-1]==log[-2]:
        #if log[-1]>args.threshold:
        #    success = True
        log.pop(-1)
    best_score = np.max(log)
    if np.max(log)>args.threshold:
        success_score = np.min([s for s in log if s>args.threshold])
        success = True
        end = log.index(success_score)+1
        #end = log.index(best_score)+1
        log = log[:end]
    else:
        success_score = best_score
    log.append(success)
    return log, success_score

logs = sorted([d for d in os.listdir('data') if d.startswith('mcts') or d.startswith('Sub')])
if args.after is not None:
    logs = sorted([d for d in logs if d[-9:-5] >= args.after])
    print('Measure metrics for:', logs)

for logname in logs:
    ep_length = []
    ep_success_length = []
    ep_success_score = []
    deltas = []
    scenes = [s for s in os.listdir(os.path.join('data', logname)) if s.startswith('scene')]
    if len(scenes)<=10:
        continue
    print(logname)

    try:
        with open(os.path.join('data', logname, 'config.json'), 'r') as f:
            cfg = json.load(f)
        if 'wandb_off' in cfg:
            if cfg['wandb_off']:
                continue
        if 'iteration_limit' in cfg:
            if cfg['iteration_limit']<1000:
                continue
        if 'iql_path' in cfg:
            iql_path = cfg['iql_path']
            print('IQL:', iql_path)
        if 'scenes' in cfg:
            cfg_scenes = cfg['scenes']
            print('scenes:', cfg_scenes)

        for scene in scenes:
            scene_dir = os.path.join('data', logname, scene)
            logfile = [f for f in os.listdir(scene_dir) if f.endswith('.log')]
            if len(logfile)==0:
                continue
            logfile = logfile[0]
            with open(os.path.join(scene_dir, logfile), 'r') as f:
                x = f.readlines()
            times = filter_log_time(x)
            deltas += get_intervals(times)
            scores, success_score = filter_log(x)
            if success_score !=0:
                ep_success_score.append(success_score)
            if scores is None:
                continue
            if scores[-1]:
                ep_success_length.append(len(scores[:-1]))
            ep_length.append(len(scores[:-1]))
        print('Num episodes:', len(scenes))
        #print('Average time:', np.mean(deltas).seconds)
        print('Success rate: %.3f' %(len(ep_success_length)/len(scenes)))
        if len(ep_success_score)!=0:
            print('Average score: %.3f' %np.mean(ep_success_score))
        print('Episode length: %.3f' %np.mean(ep_success_length))
        #print('Average length:', np.mean(ep_length))
        print('-'*40)
    except:
        print('-'*40)
        continue
