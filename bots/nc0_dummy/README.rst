
예제 0. Dummy
==============

Dummy는 AI를 구현한 가장 간단한 예(아무 것도 하지 않는 AI)를 보여준다.

python-sc2에 AI를 구현할 때는, python-sc2의 sc2.BotAI 클래스를 상속한 뒤, 
__init__, on_step 등의 메소드를 오버라이딩해서 봇을 구현하면 된다.


.. literalinclude:: bot.py
   :pyobject: Bot


추가로 경진대회의 운영 편의상 AI 객체의 이름은 Bot으로 하고, 파일 이름은 bot.py,
해당 AI와 관련된 모든 파일은 bots 폴더의 서브폴더에 저장한다.

예를 들어 내가 제출하는 봇의 이름이 My Bot 이라면, 관련된 코드 및 데이터를 bots/my_bot 
폴더에 저장하고, BoatAI 클래스를 상속한 클래스는 bots/my_bot/bot.py 파일에 Bot 으로 한다.
