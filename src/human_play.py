import pygame
import sys
import time
from game import SnakeGame

def get_system_font():
    """获取系统中可用的中文字体"""
    available_fonts = [
        "Microsoft YaHei", # Windows
        "WenQuanYi Micro Hei", # Linux
        "Noto Sans CJK SC", # Linux
        "Hiragino Sans GB", # macOS
        "SimHei", # Windows
        None # fallback to default
    ]
    
    for font_name in available_fonts:
        try:
            if font_name is None:
                return pygame.font.Font(None, 74)  # 使用默认字体
            return pygame.font.SysFont(font_name, 74)
        except:
            continue
    
    return pygame.font.Font(None, 74)  # 如果都失败了使用默认字体

def show_menu(screen):
    """显示开始菜单"""
    title_font = get_system_font()
    normal_font = pygame.font.Font(None, 36)
    
    title = title_font.render('贪吃蛇', True, (255, 255, 255))
    text1 = normal_font.render('Press SPACE to Start', True, (255, 255, 255))
    text2 = normal_font.render('1-Classic  2-Gold  3-Neon', True, (255, 255, 255))
    
    while True:
        screen.fill((0, 0, 0))
        screen.blit(title, (screen.get_width()//2 - title.get_width()//2, 100))
        screen.blit(text1, (screen.get_width()//2 - text1.get_width()//2, 250))
        screen.blit(text2, (screen.get_width()//2 - text2.get_width()//2, 300))
        pygame.display.flip()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, "classic"
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    return True, "classic"
                elif event.key == pygame.K_1:
                    return True, "classic"
                elif event.key == pygame.K_2:
                    return True, "gold"
                elif event.key == pygame.K_3:
                    return True, "neon"
    
def game_loop():
    game = SnakeGame()
    start_game, skin = show_menu(game.display)
    if not start_game:
        return
    
    game.set_skin(skin)
    game_active = True
    last_move_time = time.time()
    move_delay = 0.1  # 移动间隔，可以调整
    
    while game_active:
        current_time = time.time()
        
        # 处理输入事件
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_active = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    action = 0
                elif event.key == pygame.K_RIGHT:
                    action = 1
                elif event.key == pygame.K_UP:
                    action = 2
                elif event.key == pygame.K_DOWN:
                    action = 3
                else:
                    continue
                
                # 只在移动间隔到达时执行移动
                if current_time - last_move_time >= move_delay:
                    _, _, done = game.step(action)
                    last_move_time = current_time
                    
                    if done:
                        # 处理死亡动画
                        animation_frames = 0
                        while animation_frames < 60:  # 增加动画帧数
                            game.render()
                            animation_frames += 1
                        
                        game_font = get_system_font()
                        text = game_font.render(f'Game Over! Score: {game.score}', True, (255, 255, 255))
                        game.display.blit(text, (game.width//2 - text.get_width()//2, 
                                               game.height//2 - text.get_height()//2))
                        pygame.display.flip()
                        pygame.time.wait(2000)
                        game_active = False
        
        # 始终渲染游戏状态
        game.render()
        pygame.time.wait(16)  # 约60FPS
    
    game.close()

if __name__ == "__main__":
    game_loop()
