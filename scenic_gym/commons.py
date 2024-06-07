import numpy as np
import pygame
from PIL import ImageDraw

def get_font(size= 44):
    fonts = [x for x in pygame.font.get_fonts()]
    default_font = 'ubuntumono'
    font = default_font if default_font in fonts else fonts[0]
    font = pygame.font.match_font(font)
    return pygame.font.Font(font, size)

def should_quit():
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            return True
        elif event.type == pygame.KEYUP:
            if event.key == pygame.K_ESCAPE:
                return True
    return False

def draw_boxes(img, boxes):
    img= ImageDraw.Draw(img, 'RGBA')
    for b in boxes:
        img.rectangle(list(b), fill=(0,0,255, 20), outline=(0,0,255, 50), width=5)

def build_projection_matrix_and_inverse(w, h, fov, inv=False):
    focal = w / (2.0 * np.tan(fov * np.pi / 360.0))
    K = np.identity(3)
    K[0, 0] = K[1, 1] = focal
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0
    if inv: 
        return np.linalg.inv(K)
    else:
        return K