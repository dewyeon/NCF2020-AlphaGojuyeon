
예제 0. Dummy
==============

python-sc2의 sc2.BotAI 클래스를 상속한 뒤, 
__init__, on_step 등의 메소드를 오버라이딩해서 봇을 구현한다.

Dummy는 AI를 구현한 가장 간단한 예(아무 것도 하지 않음)를 보여준다.

.. literalinclude:: bot.py
   :pyobject: Bot


AI를 구현할 때는 Dummy와 마찬가지로 sc2.BotAI 클래스를 상속한 객체를 
만들면서 시작한다. 

추가로 경진대회의 운영 편의상 AI 객체의 이름은 Bot으로 하고, 파일 이름은 bot.py,
해당 AI와 관련된 모든 파일은 bots 폴더의 서브폴더에 저장한다.


.. DummyBot 모듈
.. ---------------

.. .. automodule:: bots.nc0_dummy_bot.bot
..     :members:
..     :undoc-members:
..     :show-inheritance:
