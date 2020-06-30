
# echo "# ncf-midterm-1" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git remote add origin https://github.com/rex8312/ncf-midterm-1.git
# git push -u origin master


import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from . import config

from IPython import embed


def evals(args, test):
    if args.out_dir:
        out_dir = Path(args.out_dir)
        csv_file = out_dir / config.csv_file.name
    else:
        out_dirs = [p for p in Path(config.root_dir).glob('*') if p.is_dir()]
        if out_dirs:
            out_dirs.sort()
            out_dir = out_dirs[-1]
            csv_file = out_dir / config.csv_file.name
            print(f'csv_file: {csv_file}')
        else:
            print('게임 결과가 없음')

    # embed(); exit()

    with open(csv_file, 'rt') as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines]
        scores = list()
        errors = list()
        play_times = list()
        for line in lines:
            no, p1, p2, map_path, realtime, p1_score, p2_score, error, play_time = line.split(',')
            if int(no) < 45:
                if test:
                    scores.append((p1, p2, float(p1_score)))
                    errors.append((p1, p2, float(error)))
                    play_times.append((p1, p2, float(play_time) / 60))
                else:
                    if p1 != p2 and 'drop_bot' not in (p1, p2):
                        scores.append((p1, p2, float(p1_score)))
                        errors.append((p1, p2, float(error)))
                        play_times.append((p1, p2, float(play_time) / 60))
        df_score = pd.DataFrame(scores, columns=['p1', 'p2', 'win'])
        df_error = pd.DataFrame(errors, columns=['p1', 'p2', 'error'])
        df_play_time = pd.DataFrame(play_times, columns=['p1', 'p2', 'play_time'])

    grouped_score = df_score.groupby(['p1', 'p2'])
    win_ratio = grouped_score.mean().unstack(1).fillna(value=0.0)
    win_count = grouped_score.sum().unstack(1).fillna(value=0)
    total_count = grouped_score.count().unstack(1).fillna(value=0)

    fig, ax = plt.subplots(figsize=(13, 13))
    names = win_ratio.index.to_list()
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    [tick.set_rotation(45) for tick in ax.get_xticklabels()]
    ax.set_xticklabels([n.title() for n in names])
    ax.set_yticklabels([n.title() for n in names])
    values = win_ratio.to_numpy()
    ax.imshow(values, vmin=0.0, vmax=1.0, cmap='gray')
    for j, name_j in enumerate(names):
        for i, name_i in enumerate(names):
            value = values[j, i]
            brightness = 1.0 - value
            if abs(brightness - value) < 0.3:
                if value > 0.5:
                    brightness = value - 0.3
                else:
                    brightness = value + 0.3
            brightness = min(1.0, brightness)
            brightness = max(0.0, brightness)
            # if j != i:
            wc = int(win_count.values[j, i])
            tc = int(total_count.values[j, i])
            if tc > 0:
                text = ax.text(
                    i, 
                    j, 
                    f'{wc:d}/{tc:d}={wc/tc:.2f}', 
                    ha="center", 
                    va="center", 
                    color=(brightness, brightness, brightness)
                )
    # plt.show()
    if test:
        plt.savefig(out_dir / 'TEST_score_as_player1.png')
    else:
        plt.savefig(out_dir / 'score_as_player1.png')

    grouped_error = df_error.groupby(['p1', 'p2'])
    error_ratio = grouped_error.mean().unstack(1).fillna(value=0.0)
    error_count = grouped_error.sum().unstack(1).fillna(value=0)

    fig, ax = plt.subplots(figsize=(13, 13))
    names = win_ratio.index.to_list()
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    [tick.set_rotation(45) for tick in ax.get_xticklabels()]
    ax.set_xticklabels([n.title() for n in names])
    ax.set_yticklabels([n.title() for n in names])
    values = error_ratio.to_numpy()
    ax.imshow(values, vmin=0.0, vmax=1.0, cmap='gray')
    for j, name_j in enumerate(names):
        for i, name_i in enumerate(names):
            value = values[j, i]
            brightness = 1.0 - value
            if abs(brightness - value) < 0.3:
                if value > 0.5:
                    brightness = value - 0.3
                else:
                    brightness = value + 0.3
            brightness = min(1.0, brightness)
            brightness = max(0.0, brightness)
            # if j != i:
            wc = int(error_count.values[j, i])
            tc = int(total_count.values[j, i])
            if tc > 0:
                text = ax.text(
                    i, 
                    j, 
                    f'{wc:d}/{tc:d}={wc/tc:.2f}', 
                    ha="center", 
                    va="center", 
                    color=(brightness, brightness, brightness)
                )
    # plt.show()
    if test:
        plt.savefig(out_dir / 'TEST_error.png')
    else:
        plt.savefig(out_dir / 'error.png')

    grouped_play_time = df_play_time.groupby(['p1', 'p2'])
    _, _, vs = zip(*play_times)
    play_times = grouped_play_time.mean().unstack(1).fillna(value=min(vs))

    fig, ax = plt.subplots(figsize=(13, 13))
    names = win_ratio.index.to_list()
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    [tick.set_rotation(45) for tick in ax.get_xticklabels()]
    ax.set_xticklabels([n.title() for n in names])
    ax.set_yticklabels([n.title() for n in names])
    values = play_times.to_numpy()
    ax.imshow(values, vmin=values.min().min(), vmax=values.max().max(), cmap='gray')
    for j, name_j in enumerate(names):
        for i, name_i in enumerate(names):
            bg = (values[j, i] - values.min().min()) / (values.max().max() - values.min().min())
            font = 1.0 - bg
            if abs(font - bg) < 0.3:
                if value > 0.5:
                    font = bg - 0.3
                else:
                    font = bg + 0.3
            font = min(1.0, font)
            font = max(0.0, font)
            tc = int(total_count.values[j, i])
            if tc > 0:
                text = ax.text(
                    i, 
                    j, 
                    f'{values[j, i]:.2f}', 
                    ha="center", 
                    va="center", 
                    color=(font, font, font)
                )

    # plt.show()
    if test:
        plt.savefig(out_dir / 'TEST_play_time.png')
    else:
        plt.savefig(out_dir / 'play_time.png')

    #
    # TABLE
    #
    win_count.columns = [c[1] for c in win_count.columns]
    total_count.columns = [c[1] for c in total_count.columns]
    error_count.columns = [c[1] for c in error_count.columns]

    p1_scores = win_count.sum(axis=1) / total_count.sum(axis=1)
    p2_scores = 1 - win_count.T.sum(axis=1) / total_count.T.sum(axis=1)
    total_scores = (win_count.sum(axis=1) + (total_count.T - win_count.T).sum(axis=1)) / (total_count.sum(axis=1) + total_count.T.sum(axis=1))

    kv = {
        'win ratio': total_scores,
        '#wins (p1)': win_count.sum(axis=1).astype(np.int32), 
        '#games (p1)': total_count.sum(axis=1).astype(np.int32), 
        'win ratio (p1)': p1_scores,
        '#wins (p2)': (total_count.T - win_count.T).sum(axis=1).astype(np.int32), 
        '#games (p2)': total_count.T.sum(axis=1).astype(np.int32), 
        'win ratio (p2)': p2_scores,
        '#errors': error_count.sum(axis=1).astype(np.int32),
        'play_time': play_times.mean(axis=1),
    }
    summary = pd.concat(kv.values(), axis=1, sort=True)
    summary.index.name = 'bot'
    summary.columns = kv.keys()
    summary = summary.sort_values(by='win ratio', ascending=False)

    print(summary)
    if test:
        summary.to_excel(out_dir / 'TEST_summary.xlsx', sheet_name='Sheet1')
    else:
        summary.to_excel(out_dir / 'summary.xlsx', sheet_name='Sheet1')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str)
    args = parser.parse_args()

    evals(args, True)
    evals(args, False)
    
    
