from referee.game import Coord, PlayerColor, Action

class GameState:
    def __init__(self, color: PlayerColor):
        self.color = color
        self.frogs_red = set()
        self.frogs_blue = set()
        self.lilypads: set[Coord] = set()
        self._turn_count = 0

        # === Lilypad Setup ===
        lilypad_coords = [
            (0,0), (0,7),
            (1,1), (1,2), (1,3), (1,4), (1,5), (1,6),
            (6,1), (6,2), (6,3), (6,4), (6,5), (6,6),
            (7,0), (7,7)
        ]
        for r, c in lilypad_coords:
            self.lilypads.add(Coord(r, c))

        # === Frog Setup ===
        for c in range(1, 7):
            self.frogs_red.add(Coord(0, c))   # RED always on top
            self.frogs_blue.add(Coord(7, c))  # BLUE always on bottom

    def copy(self) -> "GameState":
        new_state = GameState(self.color)
        new_state.frogs_red = set(self.frogs_red)
        new_state.frogs_blue = set(self.frogs_blue)
        new_state.lilypads = set(self.lilypads)
        new_state._turn_count = self._turn_count
        return new_state

    def apply_action(self, color: PlayerColor, action: Action):
        from referee.game import MoveAction, GrowAction
        from .program import DIRECTION_DELTAS

        frogs = self.frogs_red if color == PlayerColor.RED else self.frogs_blue

        if isinstance(action, MoveAction):
            current = action.coord
            path = action.directions

            for direction in (path if isinstance(path, list) else [path]):
                if isinstance(direction, tuple):
                    direction = direction[0]

                dr, dc = DIRECTION_DELTAS[direction]
                mid = Coord(current.r + dr, current.c + dc)
                jump_r = current.r + 2 * dr
                jump_c = current.c + 2 * dc

                if mid in self.frogs_red or mid in self.frogs_blue:
                    dest = Coord(jump_r, jump_c)
                else:
                    dest = Coord(current.r + dr, current.c + dc)

                self.lilypads.discard(current)
                self.lilypads.discard(dest)
                frogs.discard(current)
                frogs.add(dest)
                current = dest

        elif isinstance(action, GrowAction):
            for frog in frogs:
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue
                        r, c = frog.r + dr, frog.c + dc
                        if 0 <= r < 8 and 0 <= c < 8:
                            coord = Coord(r, c)
                            if coord not in self.frogs_red and coord not in self.frogs_blue:
                                self.lilypads.add(coord)

