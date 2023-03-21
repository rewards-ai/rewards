import pygame
from pathlib import Path


class Track:
    def __init__(self):
        self.track1_image = pygame.image.load(r"C:\Users\acer\Downloads\rl-platform-main (1)\rl-platform-main\rewards.ai\rewards_ai\Environments\CarRacer\CarTester\Assets\track_test_7.png")

    def track1(self):
        return self.track1_image


s = pygame.display.set_mode((800, 700))
t1 = Track().track1()
while True:
    pygame.display.update()
    s.blit(t1, (0, 0))

    for e in pygame.event.get():
        if e == pygame.K_q:
            running = False
        if e.type == pygame.KEYDOWN and e.type == pygame.K_ESCAPE:
            running = False

    print(pygame.mouse.get_pos())