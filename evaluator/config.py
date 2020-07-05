
import argparse
import csv
from datetime import datetime
from pathlib import Path

from IPython import embed

from sc2_utils import parse_bool

parser = argparse.ArgumentParser('NC Fellowship 2019-sc2minigame-run_game')
parser.add_argument('--root_path', type=str, default='eval_example')
parser.add_argument('--cont', action='store_true', default=False)
parser.add_argument('--tournament',
                    type=parse_bool,
                    default=True)
parser.add_argument('--n_rounds', type=int, default=1)
# player 설정
parser.add_argument('--bot1',
                    type=str,
                    default='bots.nc_example_v6.drop_bot.DropBot',
                    help='bot 1 (player 1) 설정')
parser.add_argument('--bot2',
                    type=str,
                    default='com7',
                    help='bot 2 (player 2) 설정 (기본값: 기본 컴퓨터 난이도 7)')
# 게임 정보
parser.add_argument('--map_name',
                    type=str,
                    default='NCFellowship-2019_m1_v7',
                    help='경진대회 기본 맵')
# 옵션
parser.add_argument('--realtime',
                    type=parse_bool,
                    default=False,
                    help='false일 때는 빠르게 게임이 실행됨')
parser.add_argument('--debug',
                    type=parse_bool,
                    default=True,
                    help='bot을 생성할 때, debug 인자로 전달함')
parser.add_argument('--render',
                    type=parse_bool,
                    default=False,
                    help='True일때는 rendering option을 사용함')
parser.add_argument('--csv_file',
                    type=str,
                    default=None)
parser.add_argument('--timeout', 
                    type=int,
                    default=1800)
parser.add_argument('--save_replay',
                    type=parse_bool,
                    default=True)                    
parser.add_argument('--replay_path', type=str)
args = parser.parse_args()
args = parser.parse_args()

current = datetime.now().isoformat().replace(':', '-').split('.')[0]

root_dir = Path(args.root_path)
out_dir = root_dir / f'{current}'
if args.cont:
    dirs = [l for l in root_dir.glob('*') if l.is_dir()]
    if len(dirs) > 0:
        dirs.sort()
        out_dir = dirs[-1]

csv_file = out_dir / 'result.csv'
replay_dir = out_dir / 'replays'
system_log_file = out_dir / 'system.log'

# bot 저장소 목록 읽기
with (root_dir / 'config.csv').open() as f:
    reader = csv.reader(f)
    repos = [(name, repo) for name, repo in reader]
    names_, repos_ = zip(*repos)
    assert len(names_) == len(set(names_)), "중복된 bot 이름이 있음"
    assert len(repos_) == len(set(repos_)), "중복된 저장소가 이름이 있음"
    repos = {name: repo for name, repo in repos}
    
teams = {name: f'{root_dir}.{name}.bot.Bot' for name in repos.keys()}

