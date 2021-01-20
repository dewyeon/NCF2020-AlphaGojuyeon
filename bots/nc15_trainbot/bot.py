
__author__ = '고주연 (juyon98@korea.ac.kr), 홍은수 (deltaori0@korea.ac.kr)'

import time

import sc2
from sc2.ids.ability_id import AbilityId
from sc2.ids.unit_typeid import UnitTypeId
from sc2.ids.buff_id import BuffId


class Bot(sc2.BotAI):
    """
    초반 전략 : 비용이 싼 유닛(해병, 의료선) 15개로 초반부 시간 벌기
    중반부 전략 : 공성 전차를 활용한 중반부 싸움
    후반부 전략 : 밤까마귀를 활용한 방어선 생성, 전투 순양함(야마토 포, 전술 차원 도약)을 사용하여 적 사령부 공격, 유령 전술핵 사용
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

        # 초반 빌드 오더 생성 (해병: 12, 의료선: 1 - 두 번 반복)
        for _ in range(4):
            for _ in range(6):
                self.build_order.append(UnitTypeId.MARINE)
            self.build_order.append(UnitTypeId.MEDIVAC)

        # 중반 빌드 오더 (공성 전차 : 5)
        for _ in range(5):
            self.build_order.append(UnitTypeId.SIEGETANK)

    async def on_step(self, iteration: int):       
        actions = list()

        combat_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED, UnitTypeId.RAVEN, UnitTypeId.BATTLECRUISER, UnitTypeId.GHOST])
        tank_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC, UnitTypeId.MARINE, UnitTypeId.RAVEN, UnitTypeId.BATTLECRUISER, UnitTypeId.GHOST])
        wounded_units = self.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치

        # 후반부 빌드 오더 (밤까마귀: 1, 전투 순양함: 2, 유령: 1, 전술핵: 1)
        self.build_order.append(UnitTypeId.RAVEN)
        for _ in range(2):
            self.build_order.append(UnitTypeId.BATTLECRUISER)
        self.build_order.append(UnitTypeId.GHOST)
        self.build_order.append(AbilityId.BUILD_NUKE)

        #
        # 사령부 명령 생성
        #
        cc = self.units(UnitTypeId.COMMANDCENTER).first
        if self.can_afford(self.build_order[0]) and self.time - self.evoked.get((cc.tag, 'train'), 0) > 1.0:
            # 해당 유닛 생산 가능하고, 마지막 명령을 발행한지 1초 이상 지났음
            actions.append(cc.train(self.build_order[0]))  # 첫 번째 유닛 생산 명령 
            del self.build_order[0]  # 빌드오더에서 첫 번째 유닛 제거
            self.evoked[(cc.tag, 'train')] = self.time

        #
        # 유닛 명령 생성
        #
        for unit in self.units.not_structure:  # 건물이 아닌 유닛만 선택
            enemy_unit = self.enemy_start_locations[0]
            if self.known_enemy_units.exists:
                enemy_unit = self.known_enemy_units.closest_to(unit)  # 가장 가까운 적 유닛

            # 적 사령부와 가장 가까운 적 유닛중 더 가까운 것을 목표로 설정
            if unit.distance_to(enemy_cc) < unit.distance_to(enemy_unit):
                target = enemy_cc
            else:
                target = enemy_unit

            # 해병 명령
            if unit.type_id is UnitTypeId.MARINE:
                if combat_units.amount + tank_units.amount >= 15:   # 나중에 다른 유닛 개수랑 더하는 것으로 수정하기
                    # 전투가능한 유닛 수가 15를 넘으면 적 본진으로 공격
                    actions.append(unit.attack(target))
                    use_stimpack = True
                else:
                    # 적 사령부 방향에 유닛 집결
                    target = self.start_location + 0.25 * (enemy_cc.position - self.start_location)
                    actions.append(unit.attack(target))
                    use_stimpack = False

                if use_stimpack and unit.distance_to(target) < 15:
                    # 유닛과 목표의 거리가 15이하일 경우 스팀팩 사용
                    if not unit.has_buff(BuffId.STIMPACK) and unit.health_percentage > 0.5:
                        # 현재 스팀팩 사용중이 아니며, 체력이 50% 이상
                        if self.time - self.evoked.get((unit.tag, AbilityId.EFFECT_STIM), 0) > 1.0:
                            # 1초 이전에 스팀팩을 사용한 적이 없음
                            actions.append(unit(AbilityId.EFFECT_STIM))
                            self.evoked[(unit.tag, AbilityId.EFFECT_STIM)] = self.time
                
            # 공성 전차 명령
            if unit.type_id is UnitTypeId.SIEGETANK:                
                # 공성 모드로 전환 (사거리 증가 및 범위 공격)
                # print('target=', target, 'distance=', unit.distance_to(target))

                # 사거리 안에 들어오면 바로 공성 모드로 전환
                # if 7 < unit.distance_to(target) < 13:
                #     actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                # else:
                #     actions.append(unit.attack(target))

                # 적 사령부가 사거리에 들어왔을 때 공성 모드로 전환
                if unit.distance_to(enemy_cc) < 13 and unit.health_percentage > 0.3: 
                    actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                else:
                    actions.append(unit.attack(target))

            # Siege Mode 공성 전차 명령
            if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                if unit.distance_to(target) > 13:
                    actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                else:
                    actions.append(unit.attack(target))
            
            # 전투 순양함 명령
            if unit.type_id is UnitTypeId.BATTLECRUISER:       
                # 적 사령부로 전술 차원 도약
                if await self.can_cast(unit, AbilityId.EFFECT_TACTICALJUMP, target=enemy_cc):
                    actions.append(unit(AbilityId.EFFECT_TACTICALJUMP, target=enemy_cc))
                
                actions.append(unit.attack(target))
                
                # 야마토 포 시전 가능하면 시전
                if await self.can_cast(unit, AbilityId.YAMATO_YAMATOGUN, target=target):
                    actions.append(unit(AbilityId.YAMATO_YAMATOGUN, target=target))

            # 의료선 명령
            if unit.type_id is UnitTypeId.MEDIVAC:
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    try:
                        actions.append(unit.move(combat_units.center))
                    except:
                        actions.append(unit(AbilityId.MOVE_MOVE, target=cc))
            
            # 유령 명령
            if unit.type_id is UnitTypeId.GHOST:
                ghost_abilities = await self.get_available_abilities(unit)
                if AbilityId.TACNUKESTRIKE_NUKECALLDOWN in ghost_abilities and unit.is_idle:
                # 전술핵 발사 가능(생산완료)하고 고스트가 idle 상태이면, 적 본진에 전술핵 발사
                actions.append(unit(AbilityId.BEHAVIOR_CLOAKON_GHOST))
                actions.append(unit(AbilityId.TACNUKESTRIKE_NUKECALLDOWN, target=enemy_cc))
            
        await self.do_actions(actions)

