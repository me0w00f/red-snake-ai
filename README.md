# Snake Game

Welcome to the Snake Game project! This is a simple implementation of the classic Snake game using Python.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Contributing](#contributing)
- [License](#license)

## Introduction

The Snake Game is a popular arcade game where the player controls a snake to eat food and grow in length. The objective is to avoid colliding with the walls or the snake's own body.

This project was created as a learning exercise to demonstrate object-oriented programming principles in Python. The game uses the Pygame library for graphics and event handling, making it a great example of how to build interactive applications.

The game features smooth controls and progressive difficulty, where the snake moves faster as it grows longer. It's a perfect project for beginners looking to understand game development concepts or for those who want to explore Python's capabilities in creating graphical applications.

## Features

- Classic snake gameplay
- Simple and intuitive controls
- Score tracking
- Increasing difficulty

## Game Details

### Controls
- Use **Arrow Keys** to change direction
- Press **P** to pause the game
- Press **ESC** to quit

### Gameplay
- The snake starts with a length of 3 segments
- Each food item eaten adds 1 segment and 10 points
- The snake speed increases every 50 points
- Game ends if the snake hits the walls or itself

### Display
- Score is shown in the top-right corner
- High score is saved between sessions
- Game grid is 20x20 cells
- Food appears randomly on the grid

## AI training
This project serves as an excellent dataset for AI training purposes. The game's deterministic nature and clear rule set make it ideal for:

- Reinforcement Learning algorithms
- Deep Q-Learning implementations
- Path-finding optimization
- Pattern recognition training

The game state can be easily converted into numerical data, allowing AI models to:
- Learn optimal movement patterns
- Predict collision scenarios
- Develop food-seeking strategies
- Analyze performance metrics

## Installation

To run the Snake Game, you need to have Python installed on your system. Follow the steps below to set up the game:

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/snake-game.git
    ```
2. Navigate to the project directory:
    ```sh
    cd snake-game
    ```
3. Install the required dependencies:
    ```sh
    pip install -r requirements.txt
    ```


Use the arrow keys to control the snake. Try to eat the food and avoid collisions!

## Contributing

Contributions are welcome! If you have any ideas or improvements, feel free to open an issue or submit a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.