import random
import numpy as np

from pacman_module.game import Agent, Directions, manhattanDistance

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
                    if 0 <= new_x < width and 0 <= new_y < height and not walls[new_x][new_y]:
                        valid_moves.append((new_x, new_y))

                # Calculate probabilities based on ghost type
                prob_distribution = np.zeros(len(valid_moves))
                for idx, (new_x, new_y) in enumerate(valid_moves):
                    new_distance = manhattanDistance((new_x, new_y), position)
                    current_distance = manhattanDistance((i, j), position)

                    if self.ghost == 'afraid' or self.ghost == 'terrified':
                        # Higher probability for increasing distance from Pacman
                        prob_distribution[idx] = new_distance - current_distance
                    else:
                        # Equal probability for Fearless ghost
                        prob_distribution[idx] = 1

                # Normalize the probabilities
                if np.sum(prob_distribution) > 0:
                    prob_distribution /= np.sum(prob_distribution)
                else:
                    prob_distribution = np.ones(len(valid_moves)) / len(valid_moves)  # Equal distribution if no preferred move

                # Assign probabilities to the transition matrix
                for idx, (new_x, new_y) in enumerate(valid_moves):
                    T[i, j, new_x, new_y] = prob_distribution[idx]

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

        # Update the belief state based on the observation model
        updated_belief = np.zeros_like(belief)
        for i in range(width):
            for j in range(height):
                updated_belief[i, j] = O[i, j] * predicted_belief[i, j]

        # Normalize the updated belief state
        total_belief = np.sum(updated_belief)
        if total_belief > 0:
            updated_belief /= total_belief

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

    def identify_target_ghost(self, beliefs, eaten):
        # Initialize target position and highest probability
        target_position = None
        highest_prob = 0

        # Iterate over each belief state and eaten status
        for belief, ghost_eaten in zip(beliefs, eaten):
            if not ghost_eaten:  # Only consider ghosts that have not been eaten
                current_max_prob = np.max(belief)
                if current_max_prob > highest_prob:
                    highest_prob = current_max_prob
                    target_position = np.unravel_index(np.argmax(belief), belief.shape)

        return target_position
    
    def identify_target_zone(self, beliefs, eaten, walls, pacman_position):
        # Initialize target zone and highest probability
        positions = []
        target_zone = None

        pacman_x, pacman_y = pacman_position

        for belief, ghost_eaten in zip(beliefs, eaten):
            if not ghost_eaten:
                ghost_position = np.unravel_index(np.argmax(belief), belief.shape)
                positions.append(ghost_position)
                
        midpoint = self.find_midpoint(positions)
        midpoint_x, midpoint_y = midpoint

        x_diff = midpoint_x - pacman_x
        y_diff = midpoint_y - pacman_y

        if abs(x_diff) > abs(y_diff):
            target_zone = Directions.EAST if x_diff > 0 else Directions.WEST
            if not self.is_legal_move(pacman_position, target_zone, walls):
                target_zone = Directions.NORTH if y_diff > 0 else Directions.SOUTH
                if not self.is_legal_move(pacman_position, target_zone, walls):
                    target_zone = Directions.STOP
        else:
            target_zone = Directions.NORTH if y_diff > 0 else Directions.SOUTH
            if not self.is_legal_move(pacman_position, target_zone, walls):
                target_zone = Directions.EAST if x_diff > 0 else Directions.WEST
                if not self.is_legal_move(pacman_position, target_zone, walls):
                    target_zone = Directions.STOP

        return target_zone

    def find_midpoint(self, points):
        n = len(points)
        
        if n == 0:
            return None  # No points to find midpoint
        
        # Calculate the average of x-coordinates and y-coordinates
        avg_x = sum(point[0] for point in points) / n
        avg_y = sum(point[1] for point in points) / n
        
        return (avg_x, avg_y)
    
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
        new_x, new_y = x + dx, y + dy

        if not walls[new_x][new_y]:
            return True
        return False

    def get_new_position(self, position, action, walls):
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
            return (new_x, new_y)
        return None

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

        # possible_actions = [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]

        # best_action = Directions.STOP
        # shortest_distance = float('inf')
        best_action = self.identify_target_zone(beliefs, eaten, walls, position)

        # # Identify the most probable ghost position
        # target_ghost_position = self.identify_target_ghost(beliefs, eaten)

        # for action in possible_actions:
        #     new_position = self.get_new_position(position, action, walls)
        #     if new_position is None or target_ghost_position is None:
        #         # Choose the direction with the highest density of non-eaten ghosts
        #         best_action = self.identify_target_zone(beliefs, eaten, position)
        #     else:
        #         # Calculate the Manhattan distance to the target ghost
        #         distance = manhattanDistance(new_position, target_ghost_position)
                
        #         print(f"Action: {action}, New Position: {new_position}, Distance to Target: {distance}")

        #         # Choose the action that minimizes the distance
        #         if distance < shortest_distance:
        #             shortest_distance = distance
        #             best_action = action

        # print("Chosen Action:", best_action)

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
