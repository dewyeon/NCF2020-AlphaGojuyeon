#!/usr/bin/env python

__author__ = '박현수(hspark8312@ncsoft.com), NSSOFT Game AI Lab'


import csv
import datetime
import importlib
import itertools
import logging
import multiprocessing as mp
import os
import platform
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from IPython import embed
from termcolor import cprint
from tqdm import tqdm, trange

from eval.play_game import run_play_game
from eval.export import export

from . import config
from .config import args

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def update_bots(config):
    result = list()
    for name in config.repos:
        repo = config.repos[name]
        target = config.root_dir / 'bots' / name
        if target.exists():
            # pull bots
            cmd = f'git pull'
            ret = subprocess.call(shlex.split(cmd), cwd=target)
        else:
            # clone bots
            cmd = f'git clone {repo} {target}'
            ret = subprocess.call(shlex.split(cmd))

        if ret == 0:
            cprint(f'> {name} 업데이트 성공', 'green')
        else:
            cprint(f'> {name} 업데이트 실패', 'red')

        result.append(ret)
        time.sleep(1)
    return result


def play_games(config):
    config.replay_dir.mkdir(exist_ok=True, parents=True)

    # 게임 생성
    match_list = list(itertools.product(config.teams, config.teams))

    excludes = list()
    if config.csv_file.exists():
        with open(config.csv_file, 'rt') as f:
            reader = csv.reader(f)
            for line in reader:
                n, p1, p2, *_ = line
                excludes.append(f'{n}-{p1}-{p2}')

    # 게임 실행
    for n in trange(args.n_rounds):
        for match in tqdm(match_list, leave=False):
            p1, p2 = match

            round_key = f'{n}-{p1}-{p2}'
            if round_key in excludes:
                tqdm.write(f'SKIP: {round_key}')
                time.sleep(0.1)
                continue

            bot1 = config.teams[p1].class_path
            bot2 = config.teams[p2].class_path

            if args.save_replay:
                replay_path = str(config.replay_dir / f'{p1}-{p2}' / f'{p1}-{p2}-{n}.SC2Replay')
            else:
                replay_path = None
            log_path = config.replay_dir / f'{p1}-{p2}' / f'{p1}-{p2}-{n}.log'
            log_path.parent.mkdir(exist_ok=True, parents=True)

            realtime = False
            map_name = config.args.map_name
            timeout = config.args.timeout

            start = time.monotonic()

            try:
                
                result = run_play_game(bot1, bot2, map_name, realtime, timeout, replay_path, log_path)
                tqdm.write(f'Round: {n}, {p1}: {result[0]}, {p2}: {result[1]}, error: {result[2]}')

            except Exception as e:
                result = [0.5, 0.5, 1.0]
                with open(config.system_log_file, 'at') as f:
                    f.write(f'{bot1}, {bot2}\n')
                    f.write(f'{e}\n')

            play_time = int(time.monotonic() - start)

            # 결과 기록
            with open(config.csv_file, 'at') as f:
                writer = csv.writer(f)
                writer.writerow([n, p1, p2, map_name, result[0], result[1], result[2], play_time])

            time.sleep(10)


if __name__ == '__main__':

    cprint(f'* 평가: {config.root_dir}')

    if args.update_bots:
        cprint(f'* 봇 업데이트')
        result = update_bots(config)
        if any(result):
            cprint(f'> 업데이트 실패', 'red')
            exit(1)

    if args.play_games:
        cprint(f'* 토너먼트 시작')
        play_games(config)

    if args.export_results:
        cprint(f'* 결과 분석 및 출력')
        export(config)

    if args.publish_results:
        cprint(f'* 토너먼트 결과 공개')
        pass

    embed(); exit()
