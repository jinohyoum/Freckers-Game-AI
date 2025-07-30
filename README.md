# Freckers Game Agent

A Python agent for playing the two-player board game Freckers. The agent uses rule-based logic to select legal MOVE or GROW actions based on internal board state.

## Features

- Internal 8×8 board state with lily pads and frog positions
- MOVE action generation using directional vectors and legality checks
- GROW action logic to expand lily pads around existing frogs
- Board state updates after each turn for accurate simulation

## Requirements

- Python 3.12
- No external libraries required

## Running the Agent

Make sure the `agent/` and `referee/` folders are in the same directory.

To play a game between two identical agents:
python -m referee agent agent

To play a game between different agents:
python -m referee <red_module> <blue_module>
