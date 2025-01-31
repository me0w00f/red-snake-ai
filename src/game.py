import pygame
import random
import numpy as np
from typing import Dict, Tuple

class SnakeGame:
    # 预定义的皮肤配置
    SKINS = {
        "classic": {
            "snake_color": (255, 0, 0),
            "food_color": (0, 255, 0),
            "head_color": (200, 0, 0),
            "pattern": None
        },
        "gold": {
            "snake_color": (255, 215, 0),
            "food_color": (255, 0, 0),
            "head_color": (238, 201, 0),
            "pattern": None
        },
        "neon": {
            "snake_color": (0, 255, 255),
            "food_color": (255, 0, 255),
            "head_color": (0, 200, 200),
            "pattern": None
        }
    }

    def __init__(self, width=640, height=480, scale=20, skin="classic"):
        pygame.init()
        self.width = width
        self.height = height
        self.scale = scale
        self.display = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Red Snake Game')
        self.clock = pygame.time.Clock()
        self.skin = self.SKINS[skin]
        self.custom_skin = None
        self.reset()

    def reset(self):
        self.snake_pos = [[self.width//2, self.height//2]]
        self.snake_direction = [self.scale, 0]
        self.food_pos = self._generate_food()
        self.score = 0
        self.game_over = False
        return self._get_state()

    def _generate_food(self):
        while True:
            x = random.randint(0, (self.width-self.scale)//self.scale) * self.scale
            y = random.randint(0, (self.height-self.scale)//self.scale) * self.scale
            if [x, y] not in self.snake_pos:
                return [x, y]

    def _get_state(self):
        # 创建11个特征的状态向量
        state = np.zeros(11)
        
        head = self.snake_pos[0]
        # 危险位置检测
        state[0] = self._is_collision([head[0]-self.scale, head[1]])  # 左
        state[1] = self._is_collision([head[0]+self.scale, head[1]])  # 右
        state[2] = self._is_collision([head[0], head[1]-self.scale])  # 上
        state[3] = self._is_collision([head[0], head[1]+self.scale])  # 下
        
        # 移动方向
        state[4] = self.snake_direction[0] == -self.scale  # 左
        state[5] = self.snake_direction[0] == self.scale   # 右
        state[6] = self.snake_direction[1] == -self.scale  # 上
        state[7] = self.snake_direction[1] == self.scale   # 下
        
        # 食物相对位置
        state[8] = self.food_pos[0] < head[0]  # 食物在左边
        state[9] = self.food_pos[0] > head[0]  # 食物在右边
        state[10] = self.food_pos[1] < head[1] # 食物在上面
        
        return state

    def _is_collision(self, pos):
        return (pos[0] >= self.width or pos[0] < 0 or
                pos[1] >= self.height or pos[1] < 0 or
                pos in self.snake_pos[1:])

    def step(self, action):
        # 0: 左, 1: 右, 2: 上, 3: 下
        if action == 0:
            self.snake_direction = [-self.scale, 0]
        elif action == 1:
            self.snake_direction = [self.scale, 0]
        elif action == 2:
            self.snake_direction = [0, -self.scale]
        elif action == 3:
            self.snake_direction = [0, self.scale]

        head = [self.snake_pos[0][0] + self.snake_direction[0],
                self.snake_pos[0][1] + self.snake_direction[1]]

        reward = 0
        self.game_over = self._is_collision(head)
        if self.game_over:
            reward = -10
            return self._get_state(), reward, True

        self.snake_pos.insert(0, head)
        
        if head == self.food_pos:
            self.score += 1
            reward = 10
            self.food_pos = self._generate_food()
        else:
            self.snake_pos.pop()
            reward = -0.1

        return self._get_state(), reward, False

    def set_custom_skin(self, snake_color: Tuple[int, int, int], 
                       food_color: Tuple[int, int, int],
                       head_color: Tuple[int, int, int] = None):
        """设置自定义皮肤"""
        self.custom_skin = {
            "snake_color": snake_color,
            "food_color": food_color,
            "head_color": head_color or snake_color,
            "pattern": None
        }
        self.skin = self.custom_skin

    def set_skin(self, skin_name: str):
        """设置预定义皮肤"""
        if skin_name in self.SKINS:
            self.skin = self.SKINS[skin_name]
            self.custom_skin = None
        else:
            raise ValueError(f"Unknown skin: {skin_name}")

    def render(self):
        self.display.fill((0, 0, 0))
        
        # 绘制蛇身
        for i, pos in enumerate(self.snake_pos):
            color = self.skin["head_color"] if i == 0 else self.skin["snake_color"]
            pygame.draw.rect(self.display, color,
                           pygame.Rect(pos[0], pos[1], self.scale-2, self.scale-2))
            
            # 如果有花纹图案，在蛇身上绘制花纹
            if self.skin["pattern"]:
                pattern_rect = pygame.Rect(pos[0]+2, pos[1]+2, self.scale-6, self.scale-6)
                pygame.draw.rect(self.display, self.skin["pattern"], pattern_rect)
        
        # 绘制食物
        pygame.draw.rect(self.display, self.skin["food_color"],
                        pygame.Rect(self.food_pos[0], self.food_pos[1],
                                  self.scale-2, self.scale-2))
        
        # 更新分数显示的字体处理
        try:
            font = pygame.font.SysFont("Arial", 36)  # 尝试使用Arial
        except:
            font = pygame.font.Font(None, 36)  # 如果失败使用默认字体
            
        score_text = font.render(f'Score: {self.score}', True, (255, 255, 255))
        self.display.blit(score_text, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(10)

    def close(self):
        pygame.quit()
