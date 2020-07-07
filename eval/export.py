
# echo "# ncf-midterm-1" >> README.md
# git init
# git add README.md
# git commit -m "first commit"
# git remote add origin https://github.com/rex8312/ncf-midterm-1.git
# git push -u origin master

import enum
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import scipy.linalg as la
from IPython import embed

from . import config


def export(config):

    df = pd.read_csv(
        config.csv_file, 
        names=['no', 'p1', 'p2', 'map', 'p1_score', 'p2_score', 'error', 'play_time']
    )

    # p1과 p2가 같은 봇이면 제외
    # df = df[df['p1'] != df['p2']] 

    #
    # 승률
    # 
    df_score = df[['p1', 'p2', 'p1_score']]
    df_score.columns = ['p1', 'p2', 'win']
    grouped_score = df_score.groupby(['p1', 'p2'])
    win_ratio = grouped_score.mean().unstack(1).fillna(value=0.0)
    win_count = grouped_score.sum().unstack(1).fillna(value=0)
    total_count = grouped_score.count().unstack(1).fillna(value=0)

    fig, ax = plt.subplots(figsize=(13, 13))
    names = win_ratio.index.to_list()
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    [tick.set_rotation(45) for tick in ax.get_xticklabels()]
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
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
    plt.savefig(config.out_dir / 'score_as_player1.png')

    #
    # 에러
    #
    df_error = df[['p1', 'p2', 'error']]
    grouped_error = df_error.groupby(['p1', 'p2'])
    error_ratio = grouped_error.mean().unstack(1).fillna(value=0.0)
    error_count = grouped_error.sum().unstack(1).fillna(value=0)

    fig, ax = plt.subplots(figsize=(13, 13))
    names = win_ratio.index.to_list()
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    [tick.set_rotation(45) for tick in ax.get_xticklabels()]
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
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
    plt.savefig(config.out_dir / 'error.png')

    #
    # 게임 플레이 시간
    #
    df_play_time = df[['p1', 'p2', 'play_time']]
    grouped_play_time = df_play_time.groupby(['p1', 'p2'])
    min_play_time = df_play_time['play_time'].min()
    play_times = grouped_play_time.mean().unstack(1).fillna(value=min_play_time)

    fig, ax = plt.subplots(figsize=(13, 13))
    names = win_ratio.index.to_list()
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    [tick.set_rotation(45) for tick in ax.get_xticklabels()]
    ax.set_xticklabels(names)
    ax.set_yticklabels(names)
    values = play_times.to_numpy()
    ax.imshow(values, vmin=values.min().min(), vmax=values.max().max(), cmap='gray')
    for j, name_j in enumerate(names):
        for i, name_i in enumerate(names):
            value = values[j, i]
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
    plt.savefig(config.out_dir / 'play_time.png')

    #
    # TABLE
    #
    win_count.columns = [c[1] for c in win_count.columns]
    total_count.columns = [c[1] for c in total_count.columns]
    error_count.columns = [c[1] for c in error_count.columns]

    p1_scores = win_count.sum(axis=1) / total_count.sum(axis=1)
    p2_scores = 1 - win_count.T.sum(axis=1) / total_count.T.sum(axis=1)
    total_wins = (win_count.sum(axis=1) + (total_count.T - win_count.T).sum(axis=1))
    total_n_games = (total_count.sum(axis=1) + total_count.T.sum(axis=1))
    total_scores = total_wins / total_n_games

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
    summary.to_excel(config.out_dir / 'summary.xlsx', sheet_name='Sheet1')

    #
    # 전이 그래프
    #
    if config.args.alpha is None:
        C, pi, ranks = get_ranks(win_ratio, use_inf_alpha=True, inf_alpha_eps=0.01)
    else:
        C, pi, ranks = get_ranks(win_ratio, alpha=config.args.alpha)
    draw_response_graph(config, names, C, pi, ranks)


def get_ranks(win_ratio, alpha=10, use_inf_alpha=False, inf_alpha_eps=0.01):
    win_ratio.columns = [c[1] for c in win_ratio.columns]
    M = win_ratio.values
    names = win_ratio.columns

    # single population, alpha = /inf 경우
 
    #
    #  M --> C: compute transition matrix
    # 
 
    # transition matrix
    C = np.zeros_like(M, dtype=np.float32)
 
    n_strategies = M.shape[0]
    eta = 1 / (n_strategies - 1)
 
    for s in range(n_strategies):  # 현재 전략
        for r in range(n_strategies):  # 다음 전략
            if s != r:  # Compute off-diagonal fixation probabilities
                payoff_rs = M[r, s]
                payoff_sr = M[s, r]
                if use_inf_alpha:
                    if np.isclose(payoff_rs, payoff_sr, atol=1e-14):
                        C[s, r] = eta * 0.5
                    elif payoff_rs > payoff_sr:
                        # r이 s보다 payoff가 높으므로, s -> r 확률 높음
                        C[s, r] = eta * (1 - inf_alpha_eps)
                    else:
                        # s -> r 확률 낮음
                        C[s, r] = eta * inf_alpha_eps
                else:
                    u = alpha * (payoff_rs - payoff_sr)
                    C[s, r] = (1-np.exp(-u)) / (1-np.exp(-n_strategies * u))
        C[s, s] = 1 - sum(C[s, :])  # Diagonals
 
    #
    #  c --> pi
    #
    eigenvals, eigenvecs, _ = la.eig(C, left=True, right=True)
    mask = abs(eigenvals - 1.) < 1e-6
    eigenvecs = eigenvecs[:, mask]
    num_stationary_eigenvecs = np.shape(eigenvecs)[1]
    if num_stationary_eigenvecs != 1:
        raise ValueError(
            f'Expected 1 stationary distribution, but found {num_stationary_eigenvecs}'
        )
    eigenvecs *= 1. / sum(eigenvecs)
    pi = eigenvecs.real.flatten()
 
    # ranks
    ranks = dict()
    for s in range(n_strategies):
        k = f'{pi[s]:.5f}'
        ranks.setdefault(k, list())
        ranks[k].append(names[s])

    ranks = sorted([(float(score), ranks[score]) for score in ranks], reverse=True)
    return C, pi, ranks
 

def draw_response_graph(config, names, C, pi, ranks): 

    # print ranks
    for rank, (score, strategy) in enumerate(sorted(ranks, reverse=True)):
        print(f'rank: {rank+1}, {", ".join(strategy)}, score: {score}')

    df_ranks = pd.DataFrame([(', '.join(ns), score) for score, ns in ranks])
    df_ranks.columns = ['names', 'score']
    df_ranks.index.name = 'rank'
    df_ranks.index = range(1, df_ranks.index.stop + 1)
    print(df_ranks)
    df_ranks.to_excel(config.out_dir / 'rank.xlsx', sheet_name='Sheet1')
 
    edges = list()
    for s, name_s in enumerate(names):
        for r, name_r in enumerate(names):
            if s != r:  # s == r 인 경우 생략
                if C[s, r] >= 1 / len(names):  # 전이확률이 평균보다 작으면 생략
                    name_s_ = f'{name_s}\n{pi[s]:.3f}'
                    name_r_ = f'{name_r}\n{pi[r]:.3f}'
                    edges.append((name_s_, name_r_, C[s, r]))
 
    DG = nx.DiGraph()
    DG.add_weighted_edges_from(edges)
 
    options = {
        'node_color': 'grey',
        'node_size': 12000,
        'edge_color': 'black',
        'with_labels': True, 
        'edge_labels': {(u, v): d["weight"] for u, v, d in DG.edges(data=True)}
    }

    fig, ax = plt.subplots(figsize=(13, 13))
    # pos = nx.spring_layout(DG)
    pos = nx.planar_layout(DG)
    nx.draw(DG, pos=pos, **options)
    nx.draw_networkx_edge_labels(DG, pos=pos, **options)
    plt.savefig(config.out_dir / 'C.png')


if __name__ == '__main__':

    export(config)
