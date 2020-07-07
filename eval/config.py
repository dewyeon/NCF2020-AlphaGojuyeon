
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
parser.add_argument('--current', type=str)
# 작업 목록
parser.add_argument('--update_bots', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--play_games', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--export_results', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--publish_results', type=lambda x: x in ('True', 'true', '1'), default=True)
# 옵션
parser.add_argument('--cont', action='store_true', default=False)
parser.add_argument('--n_rounds', type=int, default=1)
parser.add_argument('--map_name', type=str, default='NCF-2020-v4')
parser.add_argument('--timeout', type=int, default=1800)
parser.add_argument('--save_replay', type=lambda x: x in ('True', 'true', '1'), default=True)
parser.add_argument('--replay_path', type=str, default=None)
parser.add_argument('--alpha', type=float, default=None)
args = parser.parse_args()

# 현재 시간
if args.current is None:
    current = datetime.now().isoformat().replace(':', '-').split('.')[0]
else:
    current = args.current

# 평가용 봇과 결과가 저장되는 폴더
root_dir = Path(args.root_path)
out_dir = root_dir / 'results' / f'{current}'
if args.cont:
    # cont 옵션이 True이면, 가장 최근에 사용한 out_dir을 재사용
    dirs = [l for l in (root_dir / 'results').glob('*') if l.is_dir()]
    if len(dirs) > 0:
        dirs.sort()
        out_dir = dirs[-1]

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
