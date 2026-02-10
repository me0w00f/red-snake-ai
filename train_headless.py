#!/usr/bin/env python3
"""Headless training script - no display needed, no wandb dependency.

Usage:
    python train_headless.py                    # Train 2000 episodes
    python train_headless.py --episodes 5000    # Custom episode count
    python train_headless.py --resume models/model_checkpoint_1000.pth  # Resume training
"""
import os
os.environ['SDL_VIDEODRIVER'] = 'dummy'
os.environ['SDL_AUDIODRIVER'] = 'dummy'

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import pygame
pygame.init()
screen = pygame.display.set_mode((1, 1))

from game import SnakeGame
from dqn_agent import DQNAgent
import logging
import argparse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training_headless.log'),
        logging.StreamHandler()
    ]
)

def train(episodes=2000, resume_path=None):
    import torch
    logging.info(f"Device: {'CUDA - ' + torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    logging.info(f"Training for {episodes} episodes")

    env = SnakeGame(width=1280, height=720, skin="gold", enable_effects=False)
    state_size = 14
    action_size = 4
    hidden_size = 128
    agent = DQNAgent(state_size, action_size, hidden_size)
    batch_size = 32

    if resume_path:
        logging.info(f"Resuming from {resume_path}")
        agent.load(resume_path)

    best_score = 0
    recent_scores = []
    os.makedirs("models", exist_ok=True)

    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        steps = 0

        while True:
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            steps += 1

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

            if done:
                agent.update_target_model()
                break

        recent_scores.append(env.score)
        if len(recent_scores) > 100:
            recent_scores.pop(0)
        avg_score = sum(recent_scores) / len(recent_scores)

        if env.score > best_score:
            best_score = env.score
            agent.save(os.path.join("models", "model_best.pth"))
            logging.info(f"  ★ New best: {best_score}")

        if (e + 1) % 10 == 0:
            logging.info(
                f"Ep {e+1}/{episodes} | Score: {env.score} | "
                f"Avg(100): {avg_score:.1f} | Best: {best_score} | "
                f"Steps: {steps} | ε: {agent.epsilon:.4f}"
            )

        if (e + 1) % 100 == 0:
            agent.save(os.path.join("models", f"model_checkpoint_{e+1}.pth"))
            logging.info(f"  Saved checkpoint: model_checkpoint_{e+1}.pth")

    agent.save(os.path.join("models", "model_final.pth"))
    logging.info(f"Done! Best: {best_score}, Avg(100): {avg_score:.1f}")
    env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headless DQN Snake Training")
    parser.add_argument('--episodes', type=int, default=2000, help='Number of episodes')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    args = parser.parse_args()
    train(episodes=args.episodes, resume_path=args.resume)
