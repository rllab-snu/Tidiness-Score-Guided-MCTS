import os
import json
import numpy as np
import pandas as pd
from dateutil import parser
from argparse import ArgumentParser

argparser = ArgumentParser()
argparser.add_argument('--threshold', type=float, default=0.9)
argparser.add_argument('--after', type=str, default=None)
argparser.add_argument('--out', type=str, default=None)
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

logs = sorted([d for d in os.listdir('data') if not d.startswith('classification')])
#if args.after is not None:
#    logs = sorted([d for d in logs if d[-9:-5] >= args.after])
print('Measure metrics for:', logs)

df = pd.DataFrame(columns=['success', 'score', 'eplen', 'numep', 'seed', 'logname', 'env'])
error_files = []
for logname in logs:
    envs = os.listdir(os.path.join('data', logname))
    for env in sorted(envs):
        scenes = sorted([s for s in os.listdir(os.path.join('data', logname, env)) if s.startswith('scene')])
        try:
            with open(os.path.join('data', logname, env, 'config.json'), 'r') as f:
                cfg = json.load(f)
                seed = cfg['seed']
        except:
            error_files.append(os.path.join(logname, env))
            continue

        ep_length = []
        ep_success_length = []
        ep_success_score = []
        deltas = []
        for scene in scenes:
            scene_dir = os.path.join('data', logname, env, scene)
            try:
                logfile = [f for f in os.listdir(scene_dir) if f.endswith('.log')]
                logfile = logfile[0]
                with open(os.path.join(scene_dir, logfile), 'r') as f:
                    x = f.readlines()
                times = filter_log_time(x)
                deltas += get_intervals(times)
                scores, success_score = filter_log(x)
                if success_score !=0:
                    ep_success_score.append(success_score)
                if scores[-1]:
                    ep_success_length.append(len(scores[:-1]))
                ep_length.append(len(scores[:-1]))
            except:
                error_files.append(os.path.join(logname, env, scene))
                continue

            numep = len(scenes)
            if numep==0 or len(ep_success_length)==0:
                success = '0.00'
                score = '0.000'
                eplen = '0.000'
            else:
                success = '%.2f'%(len(ep_success_length)/len(scenes))
                score = '%.3f'%(np.mean(ep_success_score))
                eplen = '%.3f'%(np.mean(ep_success_length))
            df.loc['%s-%s'%(logname, env)] = [success, score, eplen, numep, seed, logname, env]

            #print('Num episodes:', len(scenes))
            #print('Average time:', np.mean(deltas).seconds)
            #print('Success rate: %.3f' %(len(ep_success_length)/len(scenes)))
            #if len(ep_success_score)!=0:
            #    print('Average score: %.3f' %np.mean(ep_success_score))
            #print('Episode length: %.3f' %np.mean(ep_success_length))
            #print('Average length:', np.mean(ep_length))
            #print('-'*40)
if args.out is not None:
    df.to_csv(args.out, index=False)
print(df)
print('errors:', error_files)
