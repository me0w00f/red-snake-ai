import pygame
from game import SnakeGame
from dqn_agent import DQNAgent
import wandb
import numpy as np
import logging
import argparse
import torch
import os
from collections import deque

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train(render=False, effects=False):  # 默认关闭特效
    # 添加GPU信息日志
    if torch.cuda.is_available():
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No GPU available, using CPU")
    
    wandb.init(project="snake-dqn", name="training_run")
    logging.info("Starting training process...")
    
    env = SnakeGame(width=1280, height=720, skin="gold", enable_effects=False)  # 强制关闭特效
    state_size = 11
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 20_000_000  # 2千万回合
    best_score = 0
    scores_window = deque(maxlen=100)
    max_lengths_window = deque(maxlen=100)
    
    # 创建checkpoints目录
    os.makedirs("checkpoints", exist_ok=True)
    
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        max_length = 0
        episode_q_values = []
        
        logging.info(f"Episode {e+1}/{episodes} - Epsilon: {agent.epsilon:.4f}")
        
        while True:
            # 移除渲染相关的代码，只在特定情况下渲染
            if render and e % 10000 == 0:  # 每10000回合才渲染一次
                env.render()
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            max_length = max(max_length, len(env.snake_pos))
            
            if len(agent.memory) > batch_size:
                stats = agent.replay(batch_size)
                if e % 100 == 0:  # 减少日志记录频率
                    wandb.log({
                        "loss": stats['loss'],
                        "td_error": stats['td_error'],
                        "q_value_mean": stats['q_value_mean'],
                        "q_value_max": stats['q_value_max'],
                        "epsilon": agent.epsilon,
                        "step": steps
                    })

            if done:
                break
        
        # 只在需要时更新统计数据
        if e % 10 == 0:  # 每10回合更新一次
            scores_window.append(env.score)
            max_lengths_window.append(max_length)
            
            if env.score > best_score:
                best_score = env.score
                agent.save(f"checkpoints/best_model.pth")
                logging.info(f"New best score! {best_score}")
            
            wandb.log({
                "episode": e,
                "score": env.score,
                "avg_score": np.mean(scores_window),
                "max_score": best_score,
                "total_reward": total_reward,
                "total_steps": steps,
                "max_length": max_length,
                "avg_max_length": np.mean(max_lengths_window),
                "memory_size": len(agent.memory)
            })
        
        # 减少检查点保存频率
        if (e + 1) % 50000 == 0:  # 每50000回合保存一次
            checkpoint_path = f"checkpoints/checkpoint_{e+1}.pth"
            agent.save(checkpoint_path)
            logging.info(f"Saved checkpoint: {checkpoint_path}")
        
        # 减少日志输出频率
        if (e + 1) % 1000 == 0:  # 每1000回合输出一次
            logging.info(f"Episode: {e+1}/{episodes}")
            logging.info(f"Avg Score: {np.mean(scores_window):.2f}")
            logging.info(f"Best Score: {best_score}")
            logging.info(f"Avg Length: {np.mean(max_lengths_window):.2f}")
            logging.info(f"Epsilon: {agent.epsilon:.4f}")
            
    env.close()
    wandb.finish()
    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the game during training')
    args = parser.parse_args()
    train(render=args.render, effects=False)  # 强制关闭特效
