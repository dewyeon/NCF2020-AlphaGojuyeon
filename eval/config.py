
import argparse
import csv
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from IPython import embed
from termcolor import cprint

from sc2_utils import parse_bool

parser = argparse.ArgumentParser('NC Fellowship 2020-eval')
# 평가 정보가 저장된 경로
parser.add_argument('--root_path', type=str, default='eval_example')
# 작업 목록
parser.add_argument('--update_bots', action='store_true')
parser.add_argument('--play_games', action='store_true')
parser.add_argument('--export_results', action='store_true')
parser.add_argument('--publish_results', action='store_true')
# 옵션
parser.add_argument('--cont', action='store_true', default=False)
parser.add_argument('--max_rounds', type=int, default=1)
parser.add_argument('--map_name', type=str, default='NCF-2020-v4')
parser.add_argument('--timeout', type=int, default=1800)
parser.add_argument('--save_replay', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--replay_path', type=str, default=None)
parser.add_argument('--alpha', type=float, default=None)
parser.add_argument('--init_elo_rating', type=float, default=1000)
args = parser.parse_args()

# 평가용 봇과 결과가 저장되는 폴더
root_dir = Path(args.root_path)
out_dir = root_dir / 'results'
csv_file = out_dir / 'result.csv'
replay_dir = out_dir / 'replays'
system_log_file = out_dir / 'system.log'

# 봇 저장소 목록 읽기
with (root_dir / 'config.csv').open() as f:
    reader = csv.reader(f)
    repos = [(name, repo) for name, repo in reader]
    names_, repos_ = zip(*repos)
    assert len(names_) == len(set(names_)), "중복된 bot 이름이 있음"
    assert len(repos_) == len(set(repos_)), "중복된 저장소가 이름이 있음"
    repos = {name: repo for name, repo in repos}
    
# 봇 정보 저장
Team = namedtuple('Team', 'name, class_path, repo')

teams = dict()
for name in repos.keys():
    teams[name] = Team(name, f'{root_dir}.bots.{name}.bot.Bot', repos[name])

max_rounds = args.max_rounds
init_elo_rating = args.init_elo_rating