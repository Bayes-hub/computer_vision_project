#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: 李星毅、曾文正

import pygame
from pygame.sprite import Group
from settings import Settings
from game_stats import GameStats
from scoreboard import Scoreboard
from button import Button
from ship import Ship
import game_functions as gf
import cv2
from utils import find_target
import my_kcf
import numpy as np


def run_game(marker_img):
    # Initialize pygame, settings, and screen object.
    global onTracking
    pygame.init()
    ai_settings = Settings()
    screen = pygame.display.set_mode(
        (ai_settings.screen_width, ai_settings.screen_height))
    pygame.display.set_caption("基于SIFT和KCF的运动目标匹配与跟踪：人机交互游戏《外星人入侵》")

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # hog, fixed_window, multiscale
    tracker = my_kcf.KCF(True, True, True)

    target = cv2.imdecode(
        np.fromfile(
            marker_img,
            dtype=np.uint8),
        cv2.IMREAD_GRAYSCALE)
    first_ret, first_frame = cap.read()
    init_rect = find_target(target, first_frame)
    ix = init_rect[0][0]
    iy = init_rect[0][1]
    w = init_rect[3][0] - init_rect[0][0]
    h = init_rect[1][1] - init_rect[0][1]

    initTracking = True

    # Make the Play button.
    play_button = Button(ai_settings, screen, "Play")

    # Create an instance to store game statistics, and a scoreboard.
    stats = GameStats(ai_settings)
    sb = Scoreboard(ai_settings, screen, stats)

    # Make a ship, a group of bullets, and a group of aliens.
    ship = Ship(ai_settings, screen)
    bullets = Group()
    aliens = Group()

    # Create the fleet of aliens.
    gf.create_fleet(ai_settings, screen, ship, aliens)

    ship_x = int(ai_settings.screen_width / 2)

    # Start the main loop for the game.
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if initTracking:  # 在鼠标事件里定义的全局变量，为true代表新选好了框
            cv2.rectangle(frame, (ix, iy), (ix + w, iy + h), (0, 255, 255), 2)
            tracker.init([ix, iy, w, h], frame)

            initTracking = False
            onTracking = True
        elif onTracking:  # 为true代表没有新选框，在跟踪原来的
            boundingbox = tracker.update(frame)
            boundingbox = list(map(int, boundingbox))
            # boundingbox变成了一个列表，内容为x,y,w,h
            print(boundingbox)
            cv2.rectangle(
                frame,
                (boundingbox[0],
                 boundingbox[1]),
                (boundingbox[0] +
                 boundingbox[2],
                 boundingbox[1] +
                 boundingbox[3]),
                (0,
                 255,
                 255),
                1)

            ship_x = int(boundingbox[0] + boundingbox[2] / 2)

        cv2.imshow(
            'Webcam', cv2.resize(
                frame, (480, 300), interpolation=cv2.INTER_CUBIC))

        quit = gf.check_events(
            ai_settings,
            screen,
            stats,
            sb,
            play_button,
            ship,
            aliens,
            bullets)

        if quit:
            break

        if stats.game_active:
            ship.update(ship_x)
            gf.fire_bullet(ai_settings, screen, ship, bullets)
            gf.update_bullets(ai_settings, screen, stats, sb, ship, aliens,
                              bullets)
            gf.update_aliens(ai_settings, screen, stats, sb, ship, aliens,
                             bullets)

        gf.update_screen(ai_settings, screen, stats, sb, ship, aliens,
                         bullets, play_button)

    cap.release()
    cv2.destroyAllWindows()
    pygame.quit()
