import cv2
import matplotlib.pyplot as plt
import os


COLORS = [
    (255, 179, 0),
    (128, 62, 117),
    (255, 104, 0),
    (166, 189, 215),
    (193, 0, 32),
    (206, 162, 98),
    (129, 112, 102),
    (0, 125, 52),
    (246, 118, 142),
    (0, 83, 138),
    (255, 122, 92),
    (83, 55, 122),
    (255, 142, 0),
    (179, 40, 81),
    (244, 200, 0),
    (127, 24, 13),
    (147, 170, 0),
    (89, 51, 21),
    (241, 58, 19),
    (35, 44, 22),
]

class_names = [
    "motorbike",
    "DHelmet",
    "DNoHelmet",
    "P1Helmet",
    "P1NoHelmet",
    "P2Helmet",
    "P2NoHelmet",
    "P0Helmet",
    "P0NoHelmet",
]

class_colors = [
    (0, 255, 255),  # Xanh lơ
    (0, 255, 0),  # Xanh lá
    (0, 0, 255),  # Xanh dương
    (255, 255, 0),  # Vàng
    (255, 0, 255),  # Hồng
    (255, 0, 0),  # Đỏ
    (128, 0, 128),  # Tím
    (0, 128, 128),  # Xanh biển
    (128, 128, 0),  # Xanh rêu
]