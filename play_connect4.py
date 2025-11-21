#!/usr/bin/env python3
"""
Play Connect 4 against a trained DQN model using raylib-style rendering.
Usage: python play_connect4.py <model_path>
"""

import torch
import numpy as np
import sys
import os

# Check if pygame is available, otherwise use text mode
try:
    import pygame
    HAS_PYGAME = True
except ImportError:
    HAS_PYGAME = False
    print("Warning: pygame not found. Using text-only mode.")
    print("Install pygame with: pip install pygame")

from DQN import DQN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Constants
ROWS = 6
COLUMNS = 7
PLAYER_WIN = 1.0
ENV_WIN = -1.0

# Colors (RGB)
PUFF_RED = (187, 0, 0)
PUFF_CYAN = (0, 187, 187)
PUFF_BACKGROUND = (6, 24, 24)
BOARD_COLOR = (0, 80, 80)
EMPTY_COLOR = (0, 0, 0)

PIECE_SIZE = 96
WIDTH = COLUMNS * PIECE_SIZE
HEIGHT = ROWS * PIECE_SIZE


class Connect4Game:
    """Connect 4 game logic using bitboard representation"""

    def __init__(self):
        self.player_pieces = 0
        self.env_pieces = 0
        self.current_turn = "player"  # or "env"

    def reset(self):
        """Reset the game state"""
        self.player_pieces = 0
        self.env_pieces = 0
        self.current_turn = "player"

    def top_mask(self, column):
        """Get the bit at the top of column"""
        return (1 << (ROWS - 1)) << column * (ROWS + 1)

    def bottom_mask(self, column):
        """Get a bit mask for where a piece played at column would end up"""
        return 1 << column * (ROWS + 1)

    def invalid_move(self, column):
        """Check if a move is invalid (column is full)"""
        mask = self.player_pieces | self.env_pieces
        return (mask & self.top_mask(column)) != 0

    def play(self, column, is_player=True):
        """Play a piece in the given column"""
        if self.invalid_move(column):
            return False

        mask = self.player_pieces | self.env_pieces
        bottom = self.bottom_mask(column)
        # The new piece position is where the carry stops
        # We need to AND with the complement of mask to get only the new position
        new_piece_mask = (mask + bottom) & ~mask

        print(f"  [DEBUG play(col={column}, is_player={is_player})]")
        print(f"    player_pieces before: {bin(self.player_pieces)}")
        print(f"    env_pieces before:    {bin(self.env_pieces)}")
        print(f"    mask (combined):      {bin(mask)}")
        print(f"    bottom_mask:          {bin(bottom)}")
        print(f"    mask + bottom:        {bin(mask + bottom)}")
        print(f"    new_piece_mask:       {bin(new_piece_mask)}")

        if is_player:
            self.player_pieces |= new_piece_mask
            print(f"    player_pieces after:  {bin(self.player_pieces)}")
        else:
            self.env_pieces |= new_piece_mask
            print(f"    env_pieces after:     {bin(self.env_pieces)}")

        return True

    def won(self, pieces):
        """Check if the given pieces form a winning position"""
        # Horizontal
        m = pieces & (pieces >> (ROWS + 1))
        if m & (m >> (2 * (ROWS + 1))):
            return True

        # Diagonal 1
        m = pieces & (pieces >> ROWS)
        if m & (m >> (2 * ROWS)):
            return True

        # Diagonal 2
        m = pieces & (pieces >> (ROWS + 2))
        if m & (m >> (2 * (ROWS + 2))):
            return True

        # Vertical
        m = pieces & (pieces >> 1)
        if m & (m >> 2):
            return True

        return False

    def is_draw(self):
        """Check if the board is full (draw)"""
        mask = self.player_pieces | self.env_pieces
        return mask == 4432406249472

    def get_observation(self):
        """Get the observation array from bitboard representation"""
        obs = np.zeros(42, dtype=np.float32)

        obs_idx = 0
        for i in range(49):
            # Skip the sentinel row
            if (i + 1) % 7 == 0:
                continue

            p0_bit = (self.player_pieces >> i) & 1
            if p0_bit == 1:
                obs[obs_idx] = PLAYER_WIN

            p1_bit = (self.env_pieces >> i) & 1
            if p1_bit == 1:
                obs[obs_idx] = ENV_WIN

            obs_idx += 1

        return obs

    def check_game_over(self):
        """Check if game is over and return (done, reward)"""
        if self.won(self.player_pieces):
            return True, PLAYER_WIN
        if self.won(self.env_pieces):
            return True, ENV_WIN
        if self.is_draw():
            return True, 0.0
        return False, 0.0


class Connect4Renderer:
    """Pygame-based renderer for Connect 4"""

    def __init__(self):
        if not HAS_PYGAME:
            return

        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Connect 4 - Human vs AI")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

    def render(self, game, message=""):
        """Render the current game state"""
        if not HAS_PYGAME:
            return

        self.screen.fill(PUFF_BACKGROUND)

        obs = game.get_observation()

        # Draw the board
        obs_idx = 0
        for i in range(49):
            if (i + 1) % 7 == 0:
                continue

            row = i % (ROWS + 1)
            column = i // (ROWS + 1)

            # Flip y coordinate so row 0 is at bottom
            y = HEIGHT - (row + 1) * PIECE_SIZE
            x = column * PIECE_SIZE

            # Draw board square
            pygame.draw.rect(self.screen, BOARD_COLOR,
                           (x, y, PIECE_SIZE, PIECE_SIZE))

            # Draw piece
            center = (x + PIECE_SIZE // 2, y + PIECE_SIZE // 2)
            radius = PIECE_SIZE // 2 - 4

            if obs[obs_idx] == 0.0:
                pygame.draw.circle(self.screen, EMPTY_COLOR, center, radius)
            elif obs[obs_idx] == PLAYER_WIN:
                pygame.draw.circle(self.screen, PUFF_RED, center, radius)
            else:  # ENV_WIN
                pygame.draw.circle(self.screen, PUFF_CYAN, center, radius)

            obs_idx += 1

        # Draw message
        if message:
            text = self.font.render(message, True, (255, 255, 255))
            text_rect = text.get_rect(center=(WIDTH // 2, 30))
            pygame.draw.rect(self.screen, PUFF_BACKGROUND, text_rect.inflate(20, 10))
            self.screen.blit(text, text_rect)

        pygame.display.flip()
        self.clock.tick(60)

    def get_column_from_mouse(self):
        """Get column index from mouse position"""
        if not HAS_PYGAME:
            return None

        x, _ = pygame.mouse.get_pos()
        return x // PIECE_SIZE

    def handle_events(self):
        """Handle pygame events, return column if clicked, None otherwise"""
        if not HAS_PYGAME:
            return None

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN:
                return self.get_column_from_mouse()
        return None

    def close(self):
        """Close the renderer"""
        if HAS_PYGAME:
            pygame.quit()


def print_board_text(game):
    """Print the board in text mode"""
    obs = game.get_observation()

    print("\n  0   1   2   3   4   5   6")
    print("+" + "---+" * 7)

    # Reshape and flip to display correctly
    board = obs.reshape(ROWS, COLUMNS)
    for row in board:
        print("|", end="")
        for cell in row:
            if cell == 0:
                print("   |", end="")
            elif cell == PLAYER_WIN:
                print(" X |", end="")
            else:
                print(" O |", end="")
        print()
        print("+" + "---+" * 7)
    print()


def print_debug_board(game, title="Board State"):
    """Print board state for debugging"""
    print(f"\n=== {title} ===")
    print(f"Player bitboard: {bin(game.player_pieces)}")
    print(f"Env bitboard: {bin(game.env_pieces)}")
    print("  0   1   2   3   4   5   6")
    print("+" + "---+" * 7)

    # Build board from bitboards directly
    # Rows are indexed 0-5, with 0 at bottom
    # We display top to bottom, so reverse the row order
    for row in range(ROWS - 1, -1, -1):
        print("|", end="")
        for col in range(COLUMNS):
            # Calculate bit position: column * (ROWS + 1) + row
            bit_pos = col * (ROWS + 1) + row

            player_has = (game.player_pieces >> bit_pos) & 1
            env_has = (game.env_pieces >> bit_pos) & 1

            if player_has and env_has:
                print(" B |", end="")  # Both (BUG!)
            elif player_has:
                print(" X |", end="")
            elif env_has:
                print(" O |", end="")
            else:
                print("   |", end="")
        print()
        print("+" + "---+" * 7)
    print()


def get_model_move(policy, obs, game):
    """Get move from the trained model, choosing the best valid move"""
    with torch.no_grad():
        # Flip the observation: model was trained from env's perspective
        # So env pieces should be +1, player pieces should be -1
        obs_flipped = -obs

        print("obs reshaped: ", obs.reshape(6,7))
        print("obs_flipped reshaped: ", obs_flipped.reshape(6,7))
        obs_tensor = torch.tensor(obs_flipped, dtype=torch.float32, device=device).unsqueeze(0)
        print("  [DEBUG get_model_move()]")
        print("Obs for model: ", obs_tensor)
        q_values = policy(obs_tensor)

        # Sort actions by Q-value (highest first)
        sorted_actions = torch.argsort(q_values[0], descending=True).tolist()

        # Pick the first valid action
        for action in sorted_actions:
            if not game.invalid_move(action):
                return action

        # Fallback: find any valid move (shouldn't happen)
        for col in range(COLUMNS):
            if not game.invalid_move(col):
                return col

        # Emergency fallback
        return 0


def play_game_text(policy):
    """Play game in text mode"""
    game = Connect4Game()
    game.reset()

    print("\n" + "="*50)
    print("CONNECT 4 - HUMAN (X) vs AI (O)")
    print("="*50)

    while True:
        print_board_text(game)
        done, reward = game.check_game_over()

        if done:
            if reward == PLAYER_WIN:
                print("🎉 YOU WIN! 🎉")
            elif reward == ENV_WIN:
                print("❌ AI WINS! ❌")
            else:
                print("🤝 DRAW! 🤝")
            break

        # Player turn
        print("Your turn (X):")
        while True:
            try:
                column = int(input("Enter column (0-6): "))
                if 0 <= column < 7 and not game.invalid_move(column):
                    break
                print("Invalid column! Try again.")
            except (ValueError, KeyboardInterrupt):
                print("\nGame interrupted!")
                return

        game.play(column, is_player=True)
        done, reward = game.check_game_over()

        if done:
            print_board_text(game)
            if reward == PLAYER_WIN:
                print("🎉 YOU WIN! 🎉")
            elif reward == ENV_WIN:
                print("❌ AI WINS! ❌")
            else:
                print("🤝 DRAW! 🤝")
            break

        # AI turn
        print("\nAI's turn (O)...")
        obs = game.get_observation()
        ai_column = get_model_move(policy, obs, game)
        game.play(ai_column, is_player=False)
        print(f"AI played column {ai_column}")


def play_game_graphical(policy, renderer):
    """Play game with graphical interface"""
    game = Connect4Game()
    game.reset()

    # AI goes first
    message = "AI's turn (Cyan)..."
    renderer.render(game, message)
    pygame.time.wait(500)

    obs = game.get_observation()
    ai_column = get_model_move(policy, obs, game)
    print(f"\nAI plays column {ai_column}")
    game.play(ai_column, is_player=False)
    print_debug_board(game, "After AI move")

    message = "Your turn (Red)"

    while True:
        renderer.render(game, message)

        done, reward = game.check_game_over()
        if done:
            if reward == PLAYER_WIN:
                message = "YOU WIN!"
            elif reward == ENV_WIN:
                message = "AI WINS!"
            else:
                message = "DRAW!"
            renderer.render(game, message)

            # Wait for click or escape to continue
            waiting = True
            while waiting:
                event = renderer.handle_events()
                if event == "quit" or event is not None:
                    waiting = False
            return

        # Handle events
        event = renderer.handle_events()
        if event == "quit":
            return

        if event is not None and isinstance(event, int):
            column = event
            if 0 <= column < 7 and not game.invalid_move(column):
                # Player move
                print(f"\nPlayer plays column {column}")
                game.play(column, is_player=True)
                print_debug_board(game, "After Player move")

                done, reward = game.check_game_over()
                if done:
                    continue

                # AI move
                message = "AI's turn (Cyan)..."
                renderer.render(game, message)
                pygame.time.wait(500)  # Brief pause

                obs = game.get_observation()
                ai_column = get_model_move(policy, obs, game)
                print(f"\nAI plays column {ai_column}")
                game.play(ai_column, is_player=False)
                print_debug_board(game, "After AI move")

                message = "Your turn (Red)"


def main():
    if len(sys.argv) != 2:
        print("Usage: python play_connect4.py <model_path>")
        print("Example: python play_connect4.py models/my_model.pt")
        sys.exit(1)

    model_path = sys.argv[1]

    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found!")
        sys.exit(1)

    print("Loading model...")

    # Load model
    try:
        policy = torch.load(model_path, map_location=device, weights_only=False)

        print("✓ Model loaded successfully!")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)

    if HAS_PYGAME:
        renderer = Connect4Renderer()
        try:
            while True:
                play_game_graphical(policy, renderer)
                print("\nPlay again? Press any key or click, ESC to quit")
        finally:
            renderer.close()
    else:
        while True:
            play_game_text(policy)
            again = input("\nPlay again? (y/n): ").lower()
            if again != 'y':
                break


if __name__ == "__main__":
    main()
