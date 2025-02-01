import pygame
import math
import random

class Trail:
    def __init__(self):
        self.positions = []
        self.max_length = 10
        self.colors = [
            (255, 0, 0), (255, 127, 0), 
            (255, 255, 0), (0, 255, 0),
            (0, 0, 255), (75, 0, 130), 
            (148, 0, 211)
        ]
        
    def update(self, pos):
        self.positions.insert(0, pos)
        if len(self.positions) > self.max_length:
            self.positions.pop()
            
    def draw(self, surface, scale):
        for i, pos in enumerate(self.positions):
            alpha = 255 * (1 - i/self.max_length)
            color = self.colors[i % len(self.colors)]
            s = pygame.Surface((scale, scale), pygame.SRCALPHA)
            pygame.draw.rect(s, (*color, int(alpha)), (0, 0, scale-2, scale-2))
            surface.blit(s, pos)

class FoodGlow:
    def __init__(self):
        self.time = 0
        
    def draw(self, surface, pos, color, scale):
        self.time += 0.1
        glow_size = scale + math.sin(self.time) * 4
        alpha = (math.sin(self.time) + 1) * 127
        
        s = pygame.Surface((int(glow_size*2), int(glow_size*2)), pygame.SRCALPHA)
        pygame.draw.circle(s, (*color, int(alpha)), 
                         (glow_size, glow_size), glow_size)
        surface.blit(s, (pos[0]-glow_size//2, pos[1]-glow_size//2))

class ScorePopup:
    def __init__(self, x, y, score):
        self.x = x
        self.y = y
        self.score = score
        self.lifetime = 60
        self.vy = -2
        self.scale = 1.0  # 添加缩放效果
        
    def update(self):
        self.y += self.vy
        self.lifetime -= 1
        # 添加弹跳和缩放效果
        if self.lifetime > 45:
            self.scale = 1.5 - (60 - self.lifetime) * 0.02
        else:
            self.scale = 1.0
        return self.lifetime > 0
        
    def draw(self, surface):
        alpha = int((self.lifetime / 60) * 255)
        font = pygame.font.Font(None, int(36 * self.scale))  # 应用缩放
        text = font.render(f'+{self.score}', True, (255, 255, 255))
        text.set_alpha(alpha)
        # 居中显示
        text_rect = text.get_rect(center=(self.x + 10, self.y))
        surface.blit(text, text_rect)

class BackgroundEffect:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles = []
        self.timer = 0
        
    def update(self):
        self.timer += 1
        if self.timer % 30 == 0:  # 每30帧创建新粒子
            x = random.randint(0, self.width)
            self.particles.append({
                'x': x,
                'y': 0,
                'speed': random.uniform(1, 3),
                'size': random.randint(1, 3),
                'alpha': 255
            })
            
        for p in self.particles[:]:
            p['y'] += p['speed']
            p['alpha'] -= 2
            if p['y'] > self.height or p['alpha'] <= 0:
                self.particles.remove(p)
                
    def draw(self, surface):
        for p in self.particles:
            pygame.draw.circle(surface, 
                             (255, 255, 255, p['alpha']), 
                             (int(p['x']), int(p['y'])), 
                             p['size'])

# 显式声明要导出的类
__all__ = ['Trail', 'FoodGlow', 'ScorePopup', 'BackgroundEffect']
