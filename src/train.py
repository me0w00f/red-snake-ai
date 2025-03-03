import pygame
from game import SnakeGame
from dqn_agent import DQNAgent
import wandb
import numpy as np
import logging
import argparse
import torch
import os

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)

def train(render=False, effects=True):  # 添加effects参数
    # 添加GPU信息日志
    if torch.cuda.is_available():
        logging.info(f"GPU available: {torch.cuda.get_device_name(0)}")
    else:
        logging.warning("No GPU available, using CPU")
    
    wandb.init(project="snake-dqn", name="training_run")
    logging.info("Starting training process...")
    
    env = SnakeGame(width=1280, height=720, skin="gold", enable_effects=effects)  # 传递effects参数
    state_size = 11
    action_size = 4
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 2000

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0
        
        logging.info(f"Episode {e+1}/{episodes} - Epsilon: {agent.epsilon:.4f}")
        
        while True:
            if render:
                env.render()
                if env.death_animation:
                    while env.death_animation:
                        env.render()
                        pygame.time.wait(50)  # 等待50毫秒以确保动画播放
            
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1
            
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                wandb.log({
                    "loss": loss,
                    "epsilon": agent.epsilon,
                    "step": steps
                })

            if done:
                agent.update_target_model()
                logging.info(f"Episode: {e+1} - Score: {env.score} - Steps: {steps}")
                break
                
        wandb.log({
            "episode": e,
            "score": env.score,
            "total_reward": total_reward,
            "total_steps": steps
        })
        
        if (e + 1) % 100 == 0:

            model_dir = "models"
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
                logging.info(f"Created directory: {model_dir}")
                
            model_path = os.path.join(model_dir, f"model_checkpoint_{e+1}.pth")
            agent.save(model_path)
            logging.info(f"Saved model checkpoint: {model_path}")
            
    env.close()
    wandb.finish()
    logging.info("Training completed!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true', help='Render the game during training')
    parser.add_argument('--no-effects', action='store_true', help='Disable visual effects')
    args = parser.parse_args()
    train(render=args.render, effects=not args.no_effects)
