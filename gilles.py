import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance
from pacman_module.util import Queue


def binomial_coefficient(n, k):
    """
    Calculate the binomial coefficient 'n choose k'.
    """
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)  # Take advantage of symmetry
    coeff = 1
    for i in range(k):
        coeff *= n - i
        coeff //= i + 1
    return coeff


def binomial_pmf(k, n, p):
    """
    Calculate the probability mass function of a binomial distribution.
    """
    return binomial_coefficient(n, k) * (p ** k) * ((1 - p) ** (n - k))


class BeliefStateAgent(Agent):
    """Belief state agent.

    Arguments:
        ghost: The type of ghost (as a string).
    """

    def __init__(self, ghost):
        super().__init__()

        self.ghost = ghost

    def transition_matrix(self, walls, position):
        """Builds the transition matrix

            T_t = P(X_t | X_{t-1})

        given the current Pacman position.

        Arguments:
            walls: The W x H grid of walls.
            position: The current position of Pacman.

        Returns:
            The W x H x W x H transition matrix T_t. The element (i, j, k, l)
            of T_t is the probability P(X_t = (k, l) | X_{t-1} = (i, j)) for
            the ghost to move from (i, j) to (k, l).
        """

        width, height = walls.width, walls.height
        T = np.zeros((width, height, width, height))

        for i in range(width):
            for j in range(height):
                if walls[i][j]:  # Skip if the current position is a wall
                    continue

                valid_moves = []
                # Check all possible moves (up, down, left, right, stay)
                for dx, dy in [(0, 1), (0, -1), (-1, 0), (1, 0), (0, 0)]:
                    new_x, new_y = i + dx, j + dy
                    if (0 <= new_x < width and
                            0 <= new_y < height and
                            not walls[new_x][new_y]):
                        valid_moves.append((new_x, new_y))

                # Calculate probabilities based on ghost type
                prob_distribution = np.zeros(len(valid_moves))
                for idx, (new_x, new_y) in enumerate(valid_moves):
                    new_distance = manhattanDistance((new_x, new_y), position)
                    current_distance = manhattanDistance((i, j), position)
                    distance_diff = new_distance - current_distance

                    if self.ghost == 'afraid':
                        scaling_factor = 1.25
                    elif self.ghost == 'terrified':
                        scaling_factor = 2
                    else:
                        scaling_factor = 1

                    prob_distribution[idx] = np.exp(scaling_factor *
                                                    distance_diff)
                # Normalize the probabilities
                if np.sum(prob_distribution) > 0:
                    prob_distribution /= np.sum(prob_distribution)
                else:
                    # Equal distribution if no preferred move
                    num_moves = len(valid_moves)
                    prob_distribution = np.ones(num_moves) / num_moves
                # Assign probabilities to the transition matrix
                for idx, (new_x, new_y) in enumerate(valid_moves):
                    T[i, j, new_x, new_y] = prob_distribution[idx]
        if np.any(T < 0) or np.any(T > 1):
            print("Invalid values in Transition Matrix (T):\n", T)

        return T

    def observation_matrix(self, walls, evidence, position):
        """Builds the observation matrix

            O_t = P(e_t | X_t)

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The W x H observation matrix O_t.
        """

        width, height = walls.width, walls.height
        O = np.zeros((width, height))

        # Parameters for the binomial noise distribution
        n, p = 4, 0.5

        for i in range(width):
            for j in range(height):
                if walls[i][j]:  # Skip if the position is a wall
                    continue

                true_distance = manhattanDistance((i, j), position)
                noise = evidence - true_distance
                adjusted_noise = noise + n * p

                # Round the adjusted noise to the nearest integer
                adjusted_noise = int(round(adjusted_noise))

                # Calculate the probability of this noise value
                if 0 <= adjusted_noise <= n:
                    probability = binomial_pmf(adjusted_noise, n, p)
                else:
                    probability = 0  # Impossible noise value

                O[i, j] = probability

        if np.any(O < 0) or np.any(O > 1):
            print("Invalid values in Observation Matrix (O):\n", O)

        return O

    def update(self, walls, belief, evidence, position):
        """Updates the previous ghost belief state

            b_{t-1} = P(X_{t-1} | e_{1:t-1})

        given a noisy ghost distance evidence e_t and the current Pacman
        position.

        Arguments:
            walls: The W x H grid of walls.
            belief: The belief state for the previous ghost position b_{t-1}.
            evidence: A noisy ghost distance evidence e_t.
            position: The current position of Pacman.

        Returns:
            The updated ghost belief state b_t as a W x H matrix.
        """

        # Get the transition matrix for the ghost movements
        T = self.transition_matrix(walls, position)
        # Get the observation matrix for the current evidence
        O = self.observation_matrix(walls, evidence, position)

        # Predict the next state based on the transition model
        predicted_belief = np.zeros_like(belief)
        width, height = walls.width, walls.height

        for i in range(width):
            for j in range(height):
                for k in range(width):
                    for l in range(height):
                        predicted_belief[k, l] += T[i, j, k, l] * belief[i, j]

        if np.any(predicted_belief < 0) or np.any(predicted_belief > 1):
            print("Invalid values in predicted_belief:\n", predicted_belief)

        # Update the belief state based on the observation model
        updated_belief = np.zeros_like(belief)
        for i in range(width):
            for j in range(height):
                updated_belief[i, j] = O[i, j] * predicted_belief[i, j]

        # Normalize the updated belief state
        total_belief = np.sum(updated_belief)
        # After calculating total_belief
        if total_belief <= 0:
            print("Invalid total_belief value:", total_belief)
            print("updated_belief before normalization:\n", updated_belief)

        if total_belief > 0:
            updated_belief /= total_belief

        # print("Updated Belief State:\n", updated_belief)
        return updated_belief

    def get_action(self, state):
        """Updates the previous belief states given the current state.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            The list of updated belief states.
        """

        walls = state.getWalls()
        beliefs = state.getGhostBeliefStates()
        eaten = state.getGhostEaten()
        evidences = state.getGhostNoisyDistances()
        position = state.getPacmanPosition()

        new_beliefs = [None] * len(beliefs)

        for i in range(len(beliefs)):
            if eaten[i]:
                new_beliefs[i] = np.zeros_like(beliefs[i])
            else:
                new_beliefs[i] = self.update(
                    walls,
                    beliefs[i],
                    evidences[i],
                    position,
                )

        return new_beliefs


class PacmanAgent(Agent):
    """Pacman agent that tries to eat ghosts given belief states."""

    def __init__(self):
        super().__init__()
        self.stopCount = 0
        self.saved_moves = []
        self.last_target_zone = None
        self.last_three_moves = []

    def is_opposite_move(self, move1, move2):
        opposite_moves = {
            Directions.NORTH: Directions.SOUTH,
            Directions.SOUTH: Directions.NORTH,
            Directions.EAST: Directions.WEST,
            Directions.WEST: Directions.EAST
        }
        return opposite_moves.get(move1) == move2

    def identify_target_zone(self, beliefs, eaten, walls, pacman_position):
        if self.saved_moves:
            # If saved moves are available, use them
            self.stopCount = 0
            return self.saved_moves.pop(0)
        # Initialize target zone and highest probability
        positions = []
        target_zone = None

        pacman_x, pacman_y = pacman_position

        for belief, ghost_eaten in zip(beliefs, eaten):
            if not ghost_eaten:
                ghost_position = np.unravel_index(
                    np.argmax(belief), belief.shape)
                positions.append(ghost_position)

        min_distance = float('inf')
        for position in positions:
            distance = manhattanDistance(position, pacman_position)
            if distance < min_distance:
                min_distance = distance
                closest_ghost = position
        ghost_x, ghost_y = closest_ghost

        x_diff = ghost_x - pacman_x
        y_diff = ghost_y - pacman_y

        if abs(x_diff) > abs(y_diff):
            target_zone = Directions.EAST if x_diff > 0 else Directions.WEST
            if not self.is_legal_move(pacman_position, target_zone, walls):
                target_zone = (Directions.NORTH if y_diff > 0
                               else Directions.SOUTH)
                if not self.is_legal_move(pacman_position, target_zone, walls):
                    target_zone = Directions.STOP
        else:
            target_zone = Directions.NORTH if y_diff > 0 else Directions.SOUTH
            if not self.is_legal_move(pacman_position, target_zone, walls):
                target_zone = (Directions.EAST if x_diff > 0
                               else Directions.WEST)
                if not self.is_legal_move(pacman_position, target_zone, walls):
                    target_zone = Directions.STOP

        if (target_zone == self.last_target_zone and
                target_zone == Directions.STOP):
            self.stopCount += 1
        elif target_zone == Directions.STOP:
            self.stopCount = 1  # first consecutive stop
            self.last_target_zone = target_zone
        else:
            self.stopCount = 0  # Reset the counter if the target_zone changes
            self.last_target_zone = target_zone

        if self.last_three_moves and self.is_opposite_move(target_zone, self.last_three_moves[-1]):
            self.last_three_moves.append(target_zone)
            if len(self.last_three_moves) > 3:
                self.last_three_moves.pop(0)
        else:
            self.last_three_moves = [target_zone]

        # Check for repeated or opposite moves
        if target_zone == self.last_target_zone:
            self.stopCount += 1
        else:
            self.stopCount = 0

        self.last_target_zone = target_zone

        # Check if stopCount has reached the threshold or three consecutive opposite moves
        if self.stopCount >= 5 or (len(self.last_three_moves) == 3 and all(
            self.is_opposite_move(self.last_three_moves[i], self.last_three_moves[i - 1])
            for i in range(1, 3))):
            # Execute the logic as in self.stopCount == 5
            if not self.saved_moves:
                closest_ghost = self.identify_closest_ghost(beliefs, eaten, pacman_position)
                self.saved_moves = self.compute_path(pacman_position, walls, closest_ghost)[:25]
            if self.saved_moves:
                target_zone = self.saved_moves.pop(0)
            else:
                target_zone = self.choose_direction(pacman_position, walls)
            self.stopCount = 0  # Reset the stopCount after taking action
            self.last_three_moves = []  # Clear the last three moves list

        return target_zone

    def is_legal_move(self, position, action, walls):
        # Calculate new position based on action
        direction_deltas = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }

        # Calculate new position based on action
        x, y = position
        dx, dy = direction_deltas.get(action, (0, 0))
        new_x, new_y = x + dx, y + dy

        if not walls[new_x][new_y]:
            return True
        return False

    def choose_direction(self, position, walls):
        possible_actions = [Directions.NORTH, Directions.SOUTH,
                            Directions.EAST, Directions.WEST]
        for action in possible_actions:
            if self.is_legal_move(position, action, walls):
                return action

    def compute_path(self, start, walls, goal):
        """
        Computes the first possible path from the
        start position to the goal position.

        Arguments:
            start: The starting position (x, y).
            walls: The W x H grid of walls.
            goal: The goal position (x, y).

        Returns:
            A list of the first 20 moves (game.Directions)
            representing the computed path.
        """
        visited = set()
        queue = Queue()
        queue.push((start, []))

        while not queue.isEmpty():
            current, path = queue.pop()

            if current == goal:
                return path

            if current in visited:
                continue

            visited.add(current)

            # Add neighbors to the queue
            neighbors = self.get_valid_neighbors(current, walls)
            for neighbor, move in neighbors:
                queue.push((neighbor, path + [move]))

        # If no path found, return an empty list
        return []

    def get_valid_neighbors(self, position, walls):
        """
        Get valid neighbors of a given position.

        Arguments:
            position: The current position (x, y).
            walls: The W x H grid of walls.

        Returns:
            A list of valid neighbors as tuples (neighbor_position, move).
        """
        neighbors = []

        for move in [Directions.NORTH, Directions.SOUTH,
                     Directions.EAST, Directions.WEST]:
            new_position = self.calculate_new_position(position, move)

            if not walls[new_position[0]][new_position[1]]:
                neighbors.append((new_position, move))

        return neighbors

    def calculate_new_position(self, position, move):
        """
        Calculate the new position based on the current position and a move.

        Arguments:
            position: The current position (x, y).
            move: The move (game.Directions).

        Returns:
            The new position as a tuple (x, y).
        """
        direction_deltas = {
            Directions.NORTH: (0, 1),
            Directions.SOUTH: (0, -1),
            Directions.EAST: (1, 0),
            Directions.WEST: (-1, 0)
        }

        dx, dy = direction_deltas.get(move, (0, 0))
        new_position = (position[0] + dx, position[1] + dy)

        return new_position

    def _get_action(self, walls, beliefs, eaten, position):
        """
        Arguments:
            walls: The W x H grid of walls.
            beliefs: The list of current ghost belief states.
            eaten: A list of booleans indicating which ghosts have been eaten.
            position: The current position of Pacman.

        Returns:
            A legal move as defined in `game.Directions`.
        """
        best_action = self.identify_target_zone(beliefs,
                                                eaten, walls, position)

        return best_action

    def get_action(self, state):
        """Given a Pacman game state, returns a legal move.

        ! DO NOT MODIFY !

        Arguments:
            state: a game state. See API or class `pacman.GameState`.

        Returns:
            A legal move as defined in `game.Directions`.
        """

        return self._get_action(
            state.getWalls(),
            state.getGhostBeliefStates(),
            state.getGhostEaten(),
            state.getPacmanPosition(),
        )