
__author__ = '고주연 (juyon98@korea.ac.kr)'

import time
import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId

class Bot(sc2.BotAI):
    """
    빌드오더(해병 5, 불곰 2, 의료선 1) 실행 후,
    유닛 명령 생성
    - 해병은 10명 이상 생성되면 적 유닛과 적 사령부 중 더 가까운 곳을 향해 모여서 이동
        적 유닛 또는 사령부까지 거리가 15미만이고, 
        본인 체력이 50% 이상인 경우 스팀팩 사용
    - 불곰은 해병과 모두 동일하나 5명 이상 생성시 이동
    - 의료선은 자신에게 가장 가까운 체력이 100% 미만인 해병 중 
        체력이 가장 낮은 순으로 해병을 치료함
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.build_order = list() # 생산할 유닛 목록

    def on_start(self):
        """
        새로운 게임마다 초기화
        """
        self.build_order = list()
        self.evoked = dict()

    async def on_step(self, iteration: int):
        actions = list()
        #
        # 빌드 오더 생성
        #
        if len(self.build_order) == 0:
            for _ in range(5):
                self.build_order.append(UnitTypeId.MARINE)
            for _ in range(2):
                self.build_order.append(UnitTypeId.MARAUDER)
            self.build_order.append(UnitTypeId.MEDIVAC)

        #
        # 사령부 명령 생성
        #
        cmdctrs = self.units(UnitTypeId.COMMANDCENTER)
        cmdctrs = cmdctrs.idle
        if cmdctrs.exists:
            cmdctr = cmdctrs.first
            if self.can_afford(self.build_order[0]) and self.time - self.evoked.get((cmdctr.tag, 'train'), 0) > 1.0:
                actions.append(cmdctr.train(self.build_order[0]))
                del self.build_order[0]
                self.evoked[(cmdctr.tag), 'train'] = self.time

        #
        # 해병 명령 생성
        #
        marines = self.units.of_type(UnitTypeId.MARINE)

        for marine in marines:
            enemy_startloc = self.enemy_start_locations[0]
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(marine)
        
            if marine.distance_to(enemy_startloc) < marine.distance_to(enemy_unit):
                target = enemy_startloc
            else:
                target = enemy_unit
        
            if marines.amount > 10:
                actions.append(marine.attack(target))
                use_stimpack = True
            else:
                target = self.start_location + 0.25 * (enemy_startloc.position - self.start_location)
                actions.append(marine.attack(target))
                use_stimpack = False
            
            if use_stimpack and marine.distance_to(target) < 15:
                if not marine.has_buff(BuffId.STIMPACK) and marine.health_percentage > 0.5:
                    if self.time - self.evoked.get((marine.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                        actions.append(marine(AbilityId.EFFECT_STIM))
                        self.evoked[(marine.tag, AbilityId.EFFECT_STIM)] = self.time
            
        #
        # 불곰 명령 생성
        #
        marauders = self.units.of_type(UnitTypeId.MARAUDER)
        for marauder in marauders:
            enemy_startloc = self.enemy_start_locations[0]
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(marauder)
        
            if marauder.distance_to(enemy_startloc) < marauder.distance_to(enemy_unit):
                target = enemy_startloc
            else:
                target = enemy_unit
            
            if marauders.amount > 5:
                actions.append(marauder.attack(target))
                use_stimpack = True
            else:
                target = self.start_location + 0.25 * (enemy_startloc.position - self.start_location)
                actions.append(marauder.attack(target))
                use_stimpack = False
            
            if use_stimpack and marauder.distance_to(target) < 15:
                if not marauder.has_buff(BuffId.STIMPACK) and marauder.health_percentage > 0.5:
                    if self.time - self.evoked.get((marauder.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                        actions.append(marauder(AbilityId.EFFECT_STIM))
                        self.evoked[(marauder.tag, AbilityId.EFFECT_STIM)] = self.time

        #
        # 의료선 명령 생성
        #
        medivacs = self.units(UnitTypeId.MEDIVAC)
        wounded_units = self.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )
        for medivac in medivacs:
            if wounded_units.exists:
                wounded_unit = wounded_units.closest_to(medivac)
                actions.append(medivac(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))

        await self.do_actions(actions)

