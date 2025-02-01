import pygame
import random
import math

class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.size = random.randint(6, 12)  # 增大粒子尺寸
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(8, 15)  # 增加初始速度
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.lifetime = 100  # 调整生命周期
        self.alpha = 255

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.vy += 0.5  # 增加重力效果
        self.lifetime -= 3  # 调整消失速度
        self.alpha = int((self.lifetime / 100) * 255)
        return self.lifetime > 0

    def draw(self, surface):
        if self.alpha > 0:
            alpha_color = (*self.color, int(self.alpha))
            surf = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            pygame.draw.circle(surf, alpha_color, (self.size//2, self.size//2), self.size//2)
            surface.blit(surf, (int(self.x), int(self.y)))

class ParticleSystem:
    def __init__(self):
        self.particles = []

    def create_explosion(self, x, y, color, count=50):
        for _ in range(count):
            particle = Particle(x, y, color)
            # 添加更大的随机偏移
            particle.x += random.randint(-20, 20)
            particle.y += random.randint(-20, 20)
            self.particles.append(particle)

    def create_eat_effect(self, x, y, color):
        """创建吃到食物时的粒子效果"""
        for _ in range(30):  # 增加粒子数量
            particle = Particle(x, y, color)
            particle.size = random.randint(4, 8)
            speed = random.uniform(3, 8)
            angle = random.uniform(0, 2 * math.pi)
            particle.vx = math.cos(angle) * speed
            particle.vy = math.sin(angle) * speed
            particle.lifetime = 60
            self.particles.append(particle)
    
    def create_grow_effect(self, x, y, color):
        """创建蛇生长时的光环效果"""
        for _ in range(15):
            particle = Particle(x, y, color)
            particle.size = random.randint(3, 6)
            # 创建环形扩散效果
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(1, 3)
            particle.vx = math.cos(angle) * speed
            particle.vy = math.sin(angle) * speed
            particle.lifetime = 100  # 较短的生命周期
            self.particles.append(particle)

    def update_and_draw(self, surface):
        self.particles = [p for p in self.particles if p.update()]
        for particle in self.particles:
            particle.draw(surface)

__all__ = ['ParticleSystem']  # 显式声明导出的类
