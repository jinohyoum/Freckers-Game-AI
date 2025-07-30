# COMP30024 Artificial Intelligence, Semester 1 2025
# Project Part B: Game Playing Agent

from referee.game import PlayerColor, Coord, Direction, \
    Action, MoveAction, GrowAction
from .game_state import GameState
from referee.game import MoveAction, GrowAction
import random
from statistics import mean

# Valid directions for simple adjacent moves
DIRECTIONS = {
    "RED": [Direction.Right, Direction.Left, Direction.Down,
            Direction.DownLeft, Direction.DownRight],
    "BLUE": [Direction.Right, Direction.Left, Direction.Up,
             Direction.UpLeft, Direction.UpRight],
}

DIRECTION_DELTAS = {
    Direction.Up: (-1, 0),
    Direction.Down: (1, 0),
    Direction.Left: (0, -1),
    Direction.Right: (0, 1),
    Direction.UpLeft: (-1, -1),
    Direction.UpRight: (-1, 1),
    Direction.DownLeft: (1, -1),
    Direction.DownRight: (1, 1),
}

def render_internal_gameboard(frogs_red, frogs_blue, lilypads) -> str:
    """
    Render the game board in an absolute RED/BLUE perspective:
    - RED frogs shown as 'R'
    - BLUE frogs shown as 'B'
    - Lilypads shown as '*'
    - Empty tiles as '.'
    """
    board = [["." for _ in range(8)] for _ in range(8)]

    for coord in lilypads:
        board[coord.r][coord.c] = "*"

    for coord in frogs_red:
        board[coord.r][coord.c] = "R"

    for coord in frogs_blue:
        board[coord.r][coord.c] = "B"

    lines = []
    lines.append("=== INTERNAL GAME BOARD ===")
    for r, row in enumerate(board):
        lines.append(f"{r}   " + " ".join(row))
    lines.append("    " + " ".join(str(c) for c in range(8)))
    lines.append("===========================")
    return "\n".join(lines)


class Agent:
    """
    This class is the "entry point" for your agent, providing an interface to
    respond to various Freckers game events.
    """

    def __init__(self, color: PlayerColor, **referee: dict):
        """
        This constructor method runs when the referee instantiates the agent.
        Any setup and/or precomputation should be done here.
        """
        self._color = color
        self.state = GameState(color)
        self._turn_count = 1

        match color:
            case PlayerColor.RED:
                print("Testing: I am playing as RED")
            case PlayerColor.BLUE:
                print("Testing: I am playing as BLUE")
                
    def evaluate_state(self, state: GameState) -> int:
        # Define goal row based on color
        our_goal = 7 if self._color == PlayerColor.RED else 0
        opp_goal = 0 if self._color == PlayerColor.RED else 7

        # Assign frog sets
        our_frogs = state.frogs_red if self._color == PlayerColor.RED else state.frogs_blue
        opp_frogs = state.frogs_blue if self._color == PlayerColor.RED else state.frogs_red

        # Distance to goal rows
        our_dist = sum(abs(f.r - our_goal) for f in our_frogs)
        opp_dist = sum(abs(f.r - opp_goal) for f in opp_frogs)

        # Count frogs at goal row
        our_goal_count = sum(1 for f in our_frogs if f.r == our_goal)
        opp_goal_count = sum(1 for f in opp_frogs if f.r == opp_goal)

        # Win/loss detection
        if our_goal_count == 6:
            return -10000  # we win → lowest score (minimizing)
        if opp_goal_count == 6:
            return 10000   # they win → worst case for us

        # Progress metrics
        progress = sum(f.r for f in our_frogs) if self._color == PlayerColor.RED \
                else sum(7 - f.r for f in our_frogs)

        opp_progress = sum(f.r for f in opp_frogs) if self._color == PlayerColor.BLUE \
                    else sum(7 - f.r for f in opp_frogs)

        # Mobility difference
        mobility = len(self.get_all_actions(state, self._color)) - \
                len(self.get_all_actions(state, self._opponent_color()))

        # Minimum progress frog (to discourage stalling)
        min_progress = min(f.r for f in our_frogs) if self._color == PlayerColor.RED \
                    else min(7 - f.r for f in our_frogs)

        # --- Lilypad Accessibility Evaluation ---
        lilypad_accessibility = 0
        for frog in our_frogs:
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    r, c = frog.r + dr, frog.c + dc
                    if 0 <= r < 8 and 0 <= c < 8:
                        if Coord(r, c) in state.lilypads:
                            lilypad_accessibility += 1

        # Expect about 4 accessible lilypads per frog (max 8)
        expected_access = len(our_frogs) * 4
        accessibility_penalty = max(-6, min(6, -0.5 * (expected_access - lilypad_accessibility)))

        # --- Final Heuristic Score ---
        score = (
            (our_dist - opp_dist)
            - 3 * (our_goal_count - opp_goal_count)
            - 0.5 * (progress - opp_progress)
            - 0.1 * mobility
            + accessibility_penalty
        )

        return score

    
    def minimax(self, state: GameState, depth: int, alpha: float, beta: float, maximizing: bool) -> int:
        if depth == 0:
            return self.evaluate_state(state)

        current_color = self._color if maximizing else self._opponent_color()
        actions = self.get_all_actions(state, current_color)

        if not actions:
            return self.evaluate_state(state)

        if maximizing:
            value = float('-inf')
            for action in actions:
                next_state = state.copy()
                next_state.apply_action(current_color, action)
                value = max(value, self.minimax(next_state, depth - 1, alpha, beta, False))
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # β cutoff
            return value
        else:
            value = float('inf')
            for action in actions:
                next_state = state.copy()
                next_state.apply_action(current_color, action)
                value = min(value, self.minimax(next_state, depth - 1, alpha, beta, True))
                beta = min(beta, value)
                if beta <= alpha:
                    break  # α cutoff
            return value
    
    def get_all_actions(self, state: GameState, color: PlayerColor) -> list[Action]:
        directions = DIRECTIONS[color.name]
        frogs = state.frogs_red if color == PlayerColor.RED else state.frogs_blue
        opp_frogs = state.frogs_blue if color == PlayerColor.RED else state.frogs_red

        actions: list[Action] = []

        for frog in frogs:
            # Normal moves
            for direction in directions:
                dr, dc = DIRECTION_DELTAS[direction]
                new_r, new_c = frog.r + dr, frog.c + dc
                if 0 <= new_r < 8 and 0 <= new_c < 8:
                    dest = Coord(new_r, new_c)
                    if dest in state.lilypads and dest not in frogs and dest not in opp_frogs:
                        actions.append(MoveAction(frog, direction))

            # Single jumps
            for direction in directions:
                dr, dc = DIRECTION_DELTAS[direction]
                mid_r, mid_c = frog.r + dr, frog.c + dc
                land_r, land_c = frog.r + 2*dr, frog.c + 2*dc
                if all(0 <= x < 8 for x in (mid_r, mid_c, land_r, land_c)):
                    mid = Coord(mid_r, mid_c)
                    land = Coord(land_r, land_c)
                    if (mid in frogs or mid in opp_frogs) and \
                    land in state.lilypads and \
                    land not in frogs and land not in opp_frogs:
                        actions.append(MoveAction(frog, [direction]))

            # Multi-hop DFS
            visited = set()
            def dfs(current: Coord, path: list[Direction]):
                extended = False
                for direction in directions:
                    dr, dc = DIRECTION_DELTAS[direction]
                    mid_r, mid_c = current.r + dr, current.c + dc
                    land_r, land_c = current.r + 2*dr, current.c + 2*dc
                    if not all(0 <= x < 8 for x in (mid_r, mid_c, land_r, land_c)):
                        continue
                    mid = Coord(mid_r, mid_c)
                    land = Coord(land_r, land_c)
                    if (mid in frogs or mid in opp_frogs) and \
                    land in state.lilypads and \
                    land not in frogs and \
                    land not in opp_frogs and \
                    land not in visited:
                        visited.add(land)
                        dfs(land, path + [direction])
                        visited.remove(land)
                        extended = True
                if not extended and len(path) > 1:
                    actions.append(MoveAction(frog, path))

            dfs(frog, [])

        actions.append(GrowAction())
        return actions


    
    def _opponent_color(self) -> PlayerColor:
        return PlayerColor.RED if self._color == PlayerColor.BLUE else PlayerColor.BLUE

    def action(self, **referee: dict) -> Action:
        """
        Called by the referee when it is the agent's turn.
        Selects the action with the best minimax heuristic value (lower is better).
        """
        actions = self.get_all_actions(self.state, self._color)
        best_score = float('inf')
        best_actions = []

        for action in actions:
            sim = self.state.copy()
            sim.apply_action(self._color, action)
            score = score = self.minimax(sim, 2, float('-inf'), float('inf'), maximizing=False)
            #print(f"[DEBUG] {action} -> {score}")
            if score < best_score:
                best_score = score
                best_actions = [action]
            elif score == best_score:
                best_actions.append(action)

        # Pick randomly among equally-good actions
        best_action = random.choice(best_actions)

        if isinstance(best_action, MoveAction):
            kind = 'JUMP' if isinstance(best_action.directions, list) and len(best_action.directions) > 1 else 'MOVE'
            print(f"[DEBUG] Chosen {kind}: {best_action}")
        else:
            print(f"[DEBUG] Chosen GROW")

        return best_action

    def get_longest_jump_path(self, start: Coord) -> tuple[list[Direction], Coord]:
        visited = set()
        longest_path = []
        final_coord = start

        frogs = self.state.frogs_red if self._color == PlayerColor.RED else self.state.frogs_blue
        opp_frogs = self.state.frogs_blue if self._color == PlayerColor.RED else self.state.frogs_red

        def dfs(current: Coord, path: list[Direction]):
            nonlocal longest_path, final_coord
            extended = False
            for direction in DIRECTIONS[self._color.name]:
                dr, dc = DIRECTION_DELTAS[direction]
                mid_r = current.r + dr
                mid_c = current.c + dc
                dest_r = current.r + 2 * dr
                dest_c = current.c + 2 * dc
                if not (0 <= mid_r < 8 and 0 <= mid_c < 8 and 0 <= dest_r < 8 and 0 <= dest_c < 8):
                    continue
                mid = Coord(mid_r, mid_c)
                dest = Coord(dest_r, dest_c)
                if (mid in frogs or mid in opp_frogs) and \
                dest in self.state.lilypads and \
                dest not in frogs and \
                dest not in opp_frogs and \
                dest not in visited:
                    visited.add(dest)
                    dfs(dest, path + [direction])
                    visited.remove(dest)
                    extended = True
            if not extended and len(path) > len(longest_path):
                longest_path = path
                final_coord = current

        dfs(start, [])
        return longest_path, final_coord

    def update(self, color: PlayerColor, action: Action, **referee: dict):
        """
        Updates internal game state after a player takes a Move or Grow action.
        Properly supports multi-hop jump sequences.
        Uses absolute frog sets: frogs_red and frogs_blue.
        """
        frog_set = self.state.frogs_red if color == PlayerColor.RED else self.state.frogs_blue

        if isinstance(action, MoveAction):
            current = action.coord
            path = action.directions

            # Remove lilypad at the start
            if current in self.state.lilypads:
                self.state.lilypads.remove(current)
                print(f"[DEBUG] Removed lilypad at start {current}")
            else:
                print(f"[WARNING] Tried to remove lilypad at start {current}, but it wasn't found!")

            # Determine final destination only
            dest = current
            for direction in path:
                dr, dc = DIRECTION_DELTAS[direction]
                mid = Coord(dest.r + dr, dest.c + dc)
                jump_r, jump_c = dest.r + 2 * dr, dest.c + 2 * dc
                step_r, step_c = dest.r + dr, dest.c + dc

                jumping = False
                if 0 <= jump_r < 8 and 0 <= jump_c < 8:
                    possible_dest = Coord(jump_r, jump_c)
                    if mid in self.state.frogs_red or mid in self.state.frogs_blue:
                        jumping = True

                dest = Coord(jump_r, jump_c) if jumping else Coord(step_r, step_c)

            # Remove lilypad at final destination
            if dest in self.state.lilypads:
                self.state.lilypads.remove(dest)
                print(f"[DEBUG] Removed lilypad at destination {dest}")
            else:
                print(f"[WARNING] Tried to remove lilypad at destination {dest}, but it wasn't found!")

            # Print movement type
            move_type = "JUMP" if len(path) > 1 else "MOVE"
            print(f"[{move_type} UPDATE] {color.name} moved from {action.coord} to {dest}")

            # Update frog set
            if action.coord not in frog_set:
                print(f"[ERROR] Tried to move {color.name} frog from {action.coord}, but it wasn't found!")
            else:
                frog_set.remove(action.coord)
            frog_set.add(dest)

        elif isinstance(action, GrowAction):
            print(f"[UPDATE] {color.name} played GROW")
            grow_frogs = self.state.frogs_red if color == PlayerColor.RED else self.state.frogs_blue

            for frog in grow_frogs:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r, c = frog.r + dr, frog.c + dc
                        if 0 <= r < 8 and 0 <= c < 8:
                            coord = Coord(r, c)
                            if coord not in self.state.frogs_red and coord not in self.state.frogs_blue:
                                if coord not in self.state.lilypads:
                                    print(f"[GROW] Added lilypad at {coord}")
                                self.state.lilypads.add(coord)

        # Print internal board after the update
        print(render_internal_gameboard(
            frogs_red=self.state.frogs_red,
            frogs_blue=self.state.frogs_blue,
            lilypads=self.state.lilypads
        ))

        self._turn_count += 1
