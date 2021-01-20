
__author__ = '고주연 (juyon98@korea.ac.kr)'


# python -m bots.nc_example_v5.bot --server=172.20.41.105
# kill -9 $(ps ax | grep SC2_x64 | fgrep -v grep | awk '{ print $1 }')
# kill -9 $(ps ax | grep bots.nc_example_v5.bot | fgrep -v grep | awk '{ print $1 }')
# ps aux

import os

os.environ['MKL_DEBUG_CPU_TYPE'] = '5'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import pathlib
import pickle
import time

import nest_asyncio
import numpy as np
import sc2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython import embed
from sc2.data import Result
from sc2.ids.ability_id import AbilityId
from sc2.ids.buff_id import BuffId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.player import Bot as _Bot
from sc2.position import Point2
from termcolor import colored, cprint

nest_asyncio.apply()


class Bot(sc2.BotAI):
    """
    example v1과 유사하지만, 빌드 오더 대신, 유닛 비율을 맞추도록 유닛을 생산함
    """
    def __init__(self):
        super().__init__()

    def on_start(self):
        self.cc = self.units(UnitTypeId.COMMANDCENTER).first
        if self.start_location.distance_to(Point2((32.5, 31.5))) < 5.0:
            self.enemy_cc = Point2(Point2((95.5, 31.5)))
        else:
            self.enemy_cc = Point2(Point2((32.5, 31.5)))

    async def on_step(self, iteration: int):
        """

        """
        actions = list() # 이번 step에 실행할 액션 목록

        cc = self.units(UnitTypeId.COMMANDCENTER).first
        cc_abilities = await self.get_available_abilities(cc)
        ravens = self.units(UnitTypeId.RAVEN)
        combat_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC])
        
        if ravens.amount == 0:
            actions.append(cc.train(UnitTypeId.RAVEN))

        elif ravens.amount > 0:
            raven_abilities = await self.get_available_abilities(ravens.first)
            print(raven_abilities)

            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(ravens.first)  # 가장 가까운 적 유닛   
            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if ravens.first.distance_to(self.enemy_cc) < ravens.first.distance_to(enemy_unit):
                target = self.enemy_cc
            else:
                target = enemy_unit
            
            if AbilityId.BUILDAUTOTURRET_AUTOTURRET in raven_abilities and ravens.first.is_idle:
                # 자동포탑 생산 가능하고 밤까마귀가 idle 상태이면, 자동포탑 설치
                if self.enemy_cc==Point2(Point2((95.5, 31.5))):
                    actions.append(ravens.first(AbilityId.BUILDAUTOTURRET_AUTOTURRET, target=Point2(Point2((89.5, 31.5)))))
                else:
                    actions.append(ravens.first(AbilityId.BUILDAUTOTURRET_AUTOTURRET, target=Point2(Point2((38.5, 31.5)))))
            
            try:
                if target.is_cloaked:
                    actions.append(ravens.first(AbilityId.SCAN_MOVE, target=target.position))
            except:
                pass
                
                    
        await self.do_actions(actions)
