
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

        # 초반 빌드 오더 생성 (해병: 12, 의료선: 3)
        for _ in range(3):
            for _ in range(4):
                self.build_order.append(UnitTypeId.MARINE)
            self.build_order.append(UnitTypeId.MEDIVAC)


    async def on_step(self, iteration: int):       
        actions = list()

        combat_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC, UnitTypeId.SIEGETANK, UnitTypeId.SIEGETANKSIEGED])
        tank_units = self.units.exclude_type([UnitTypeId.COMMANDCENTER, UnitTypeId.MEDIVAC, UnitTypeId.MARINE])
        wounded_units = self.units.filter(
            lambda u: u.is_biological and u.health_percentage < 1.0
        )  # 체력이 100% 이하인 유닛 검색
        enemy_cc = self.enemy_start_locations[0]  # 적 시작 위치

        # 중반 빌드 오더 (공성 전차 : 5)
        for _ in range(5):
            self.build_order.append(UnitTypeId.SIEGETANK)

        #
        # 사령부 명령 생성
        #
        ccs = self.units(UnitTypeId.COMMANDCENTER)  # 전체 유닛에서 사령부 검색
        ccs = ccs.idle  # 실행중인 명령이 없는 사령부 검색
        if ccs.exists:  # 사령부가 하나이상 존재할 경우
            cc = ccs.first  # 첫번째 사령부 선택
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
                if combat_units.amount >= 15:
                    # 전투가능한 유닛(해병, 의료선) 수가 15를 넘으면 적 본진으로 공격
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
                # if tank_units.amount >= 5:
                #     # 공성 전차 수가 5를 넘으면 적 본진으로 공격
                #     actions.append(unit.attack(target))
                # else:
                #     # 적 사령부 방향에 유닛 집결
                #     target = self.start_location + 0.25 * (enemy_cc.position - self.start_location)
                #     actions.append(unit.attack(target))
                
                # 공성 모드로 전환 (사거리 증가 및 범위 공격)
                print('target=', target, 'distance=', unit.distance_to(target))
                if 7 < unit.distance_to(target) < 13:
                    print('siege mode')
                    actions.append(unit(AbilityId.SIEGEMODE_SIEGEMODE))
                else:
                    actions.append(unit.attack(target)) 

            # Siege Mode 공성 전차
            if unit.type_id is UnitTypeId.SIEGETANKSIEGED:
                # if not await self.can_cast(unit, AbilityId.ATTACK_ATTACK):
                if unit.distance_to(target) > 13:
                    actions.append(unit(AbilityId.UNSIEGE_UNSIEGE))
                else:
                    actions.append(unit.attack(target))

            # 의료선 명령
            if unit.type_id is UnitTypeId.MEDIVAC:
                if wounded_units.exists:
                    wounded_unit = wounded_units.closest_to(unit)  # 가장 가까운 체력이 100% 이하인 유닛
                    actions.append(unit(AbilityId.MEDIVACHEAL_HEAL, wounded_unit))  # 유닛 치료 명령
                else:
                    # 회복시킬 유닛이 없으면, 전투 그룹 중앙에서 대기
                    actions.append(unit.move(combat_units.center))
            
        await self.do_actions(actions)

