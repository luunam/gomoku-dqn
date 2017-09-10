import pygame

from agents.agent import Agent


class HumanAgent(Agent):
    def __init__(self):
        Agent.__init__(self)
        self.name = 'human'

    def move(self, gamestate=None):
        print('Human move')
        while True:
            for event in pygame.event.get():
                print(event)
                if event.type == pygame.KEYDOWN and event.key == pygame.K_f:
                    print('Done')
                    event = pygame.event.Event(pygame.USEREVENT, {})
                    pygame.event.post(event)
