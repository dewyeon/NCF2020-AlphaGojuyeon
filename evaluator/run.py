#!/usr/bin/env python

__author__ = '박현수(hspark8312@ncsoft.com), NSSOFT Game AI Lab'


import argparse
import datetime
import importlib
import logging
import multiprocessing as mp
import os
import sys
import shlex
import subprocess
import time
import platform

import numpy as np
import sc2
from IPython import embed
from sc2 import Difficulty, Race, maps
from sc2.data import Result
from sc2.player import Bot, Computer
from tqdm import tqdm, trange

from . import config
from .config import args

logger = logging.getLogger(__file__)
logger.setLevel(logging.DEBUG)


def run_play_game(bot1, bot2, map_name, realtime, timeout, replay_path, log_path):
    # TODO: 시간초과 검사
    # TODO: 예외처리
    # TODO: 시간측정
    cmd = f'python -m tournament.run --tournament=False --bot1={bot1} --bot2={bot2} --map_name={map_name} --realtime={realtime}'
    if replay_path is not None:
         cmd += f' --replay_path={replay_path}'

    result = [0.5, 0.5, 0.0]
    try:
        # tqdm.write(f'{}')
        tqdm.write(f'[{datetime.datetime.today().isoformat()}] {cmd}')
        pout = subprocess.run(
            shlex.split(cmd), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            timeout=timeout
        )
        lines = pout.stdout.split(b'\n')
        for line in lines:
            line = line.strip()
            if b'INFO:root:Result for player 1 - ' in line:
                if line.endswith(b' Victory'):
                    result[0] = 1.0
                    result[1] = 0.0
                elif line.endswith(b' Defeat'):
                    result[0] = 0.0
                    result[1] = 1.0
            elif b'INFO:root:Result for player 2 - ' in line:
                if line.endswith(b' Victory'):
                    result[0] = 0.0
                    result[1] = 1.0
                elif line.endswith(b' Defeat'):
                    result[0] = 1.0
                    result[1] = 0.0
            elif b'ERROR:sc2_patch.sc2_patch_490.main:' in line:
                result[2] = 1.0

        stdout_lines = pout.stdout.split(b'\n') 
        stdout_lines = [line.rstrip().decode('utf-8') for line in stdout_lines]
        stderr_lines = pout.stderr.split(b'\n')
        stderr_lines = [line.rstrip().decode('utf-8') for line in stderr_lines]
        lines = ['## STDOUT ##\n'] + stdout_lines + ['\n## STDERR ##\n'] + stderr_lines
        log_path.write_text('\n'.join(lines))
        assert sum(result[0:2]) == 1.0, f'result: {result}'

    except subprocess.TimeoutExpired:
        result[2] = 1.0

    return result


def play_game(bot1, bot2, map_name, realtime, replay_path):

    try:
        game_map = maps.get(map_name)
    except KeyError:
        assert os.path.exists(args.map_name + '.SC2Map'), f"지도 파일을 찾을 수 없음!: {args.map_name}"
        game_map = map_name

    bot1 = config.teams[bot1] if bot1 in config.teams else bot1
    bot2 = config.teams[bot2] if bot2 in config.teams else bot2

    # bot 초기화
    bots = list()
    for bot_path in (bot1, bot2):
        try:
            if len(bot_path) == 4 and bot_path.lower().startswith('com'):
                # bot 경로 시작이 com으로 시작하면 기본 AI를 사용함
                level = int(bot_path[3])
                # 4번째 문자는 봇의 난이도를 뜻함
                # 예) com7 -> 기본 AI 난이도 7
                # 난이도는 1~10까지 있음
                assert 1 <= level <= 10
                bot = Computer(Race.Terran, Difficulty(level))
            else:
                # 일반 bot 임포트
                module, name = bot_path.rsplit('.', 1)
                bot_cls = getattr(importlib.import_module(module), name)
                # debug 인자를 반드시 전달함
                # bot_ai = bot_cls(debug=args.debug)
                bot_ai = bot_cls()
                bot = Bot(Race.Terran, bot_ai)
            bots.append(bot)
        except ImportError:
            import traceback
            logger.error(f"bot 클래스를 임포트 할 수 없음: {bot_path}")
            traceback.print_exc()
            exit(1)
    
    result = sc2.run_game(game_map, bots, realtime=realtime, save_replay_as=replay_path)
    return result


if __name__ == '__main__':

    if args.csv_file is None:
        args.csv_file = config.csv_file

    if args.tournament:

        # config.out_dir.mkdir(exist_ok=True)
        config.replay_dir.mkdir(exist_ok=True, parents=True)

        # 게임 생성
        match_list = list()
        for p1 in config.teams:
            for p2 in config.teams:
                if not p2.startswith('com'):
                    # if p1 != p2:
                    match_list.append((p1, p2, args.map_name))

        excludes = list()
        if args.csv_file.exists():
            with open(args.csv_file, 'rt') as f:
                lines = f.readlines()
                for line in lines:
                    n, p1, p2, *_ = line.split(',')
                    excludes.append(f'{n}-{p1}-{p2}')

        # 게임 실행
        for n in trange(args.n_rounds):
            for match in tqdm(match_list):
                p1, p2, map_name = match

                round_key = f'{n}-{p1}-{p2}'
                if round_key in excludes:
                    tqdm.write(f'SKIP: {round_key}')
                    continue

                bot1 = config.teams[p1]
                bot2 = config.teams[p2]

                if args.save_replay:
                    replay_path = str(config.replay_dir / f'{p1}-{p2}' / f'{p1}-{p2}-{n}.SC2Replay')
                else:
                    replay_path = None
                log_path = config.replay_dir / f'{p1}-{p2}' / f'{p1}-{p2}-{n}.log'
                log_path.parent.mkdir(exist_ok=True, parents=True)
                    
                try:
                    start = time.monotonic()
                    result = run_play_game(bot1, 
                                            bot2, 
                                            map_name, 
                                            args.realtime, 
                                            args.timeout, 
                                            replay_path,
                                            log_path)
                    play_time = int(time.monotonic() - start)
                    tqdm.write(f'Round: {n}, {p1}: {result[0]}, {p2}: {result[1]}, error: {result[2]}')
                except Exception as e:
                    result = [0.5, 0.5, 1.0]
                    with open(config.system_log_file, 'at') as f:
                        f.write(f'{bot1}, {bot2}\n')
                        f.write(f'{e}\n')

                # 결과 기록
                with open(args.csv_file, 'at') as f:
                    # if result[2] == 1:
                    #     if result[0] > result[1]:
                    #         error_player = 2
                    #     elif result[0] < result[1]:
                    #         error_player = 1
                    #     else:
                    #         error_palyer = 3
                    # else:
                    #     error_palyer = 0
                    # line = f'{n},{p1},{p2},{map_name},{args.realtime},{result[0]},{result[1]},{error_palyer},{play_time}\n'
                    line = f'{n},{p1},{p2},{map_name},{args.realtime},{result[0]},{result[1]},{result[2]},{play_time}\n'
                    f.write(line)
                time.sleep(10)
    else:
        play_game(args.bot1, args.bot2, args.map_name, args.realtime, args.replay_path)
