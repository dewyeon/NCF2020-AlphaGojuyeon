
sc2minigame 튜토리얼
====================

.. note::

   - 현재 이 문서는 2020년 NCFellowship을 위해 준비된 플랫폼과 경진대회에 대한 
     내용을 포함하고 있음
   - 경진대회 플랫폼은 Blizzard™의 StarCraft 2 [#sc2]_, s2client-api [#sc2api]_ 
     그리고 Hannes Karppila의 python-sc2 [#python-sc2]_ 를 기반으로 하고 있음


환경설정 및 기본 예제
--------------------

.. toctree::
   :maxdepth: 1

   빠른 시작 <README.rst>
   시스템 구조 및 게임 규칙 <docs/intro.rst>
   예제 0. Dummy <bots/nc0_dummy/README.rst>
   예제 1. Simple: 간단한 build-order <bots/nc1_simple/README.rst>
   예제 2. Simple 2: 적응형 build-order <bots/nc2_simple2/README.rst>
   예제 3. Simple 3: 집결지 <bots/nc3_simple3/README.rst>


고급 예제
-----------------

.. toctree::
   :maxdepth: 1

   예제 4. Simple PPO: 분산학습 예 <bots/nc4_simple_ppo/README.rst>
   예제 5. Simple PPO: Map feature 처리 추가 <bots/nc4_simple_ppo/README.rst>
   예제 6. Simple PPO: Unit feature 처리 추가 <bots/nc4_simple_ppo/README.rst>



.. API
.. =====

.. .. toctree::
..    :maxdepth: 2

..    bots
..    run_sc2minigame
..    sc2_data
..    sc2_patch
..    sc2_utils


인덱스
------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


.. rubric:: Footnotes

.. [#sc2] StarCraft는 미국 및 다른 국가에서 Blizzard Entertainment Inc. 의 상표 또는 등록상표 입니다.
.. [#sc2api] https://github.com/Blizzard/s2client-api
.. [#python-sc2] https://github.com/Dentosal/python-sc2