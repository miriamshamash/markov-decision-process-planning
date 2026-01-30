import json
import copy

class MDP:
    def __init__(self, json_file_path=None, gamma=0.9, converge_thresh = 0.0001, max_iter = 100):
        if json_file_path:
            with open(json_file_path, 'r') as file:
                data = json.load(file)
            
            self.rows = data['rows']
            self.cols = data['cols']
            self.map = data['map']
            self.actions = ["UP", "DOWN", "LEFT", "RIGHT"]

            self.GAMMA = gamma
            self.CONVERGE_THRESH = converge_thresh
            self.MAX_ITER = max_iter
    
    def transition(self, state, action, next_state):
        """
        Calculates transition probability between states, given an action.

        Args:
            state ((int, int)) : Our current state.
            action (String) : The action we take.
            next_state ((int, int)): The row and column of the state we're transitioning to.

        Returns:
            float : The probabilty of transitioning to the next state from state given our answer [0, 1].
                    We can only move to adjacent states, and cannot move through walls or the bounds of the map.

        """

        row, col = state
        
        intended_next = None
        if action == "UP":
            intended_next = (row - 1, col)
        elif action == "DOWN":
            intended_next = (row + 1, col)
        elif action == "LEFT":
            intended_next = (row, col - 1)
        elif action == "RIGHT":
            intended_next = (row, col + 1)
        else:
            return 0
        
        if (intended_next[0] < 0 or intended_next[0] >= self.rows or 
            intended_next[1] < 0 or intended_next[1] >= self.cols):
            intended_next = state

        elif self.map[intended_next[0]][intended_next[1]] == -1:
            intended_next = state
        
        return 1.0 if intended_next == next_state else 0.0
    
    def reward(self, next_state):
        """
        Gives the reward of transitioning to the next state.

        Args:
            next_state ((int, int)): The row and column of the state we're transitioning to.

        Returns:
            int : The reward of transitioning to next_state.

        """

        next_row, next_col = next_state
        return self.map[next_row][next_col]
    
    def value_iteration(self):
        """
        Performs value iteration on the MDP.

        Args:
            None

        Returns:
            [[float]] : The caculated value of each state, 
            after peforming bellman updates until convergence.

        """
        U = [[0.0 for _ in range(self.cols)] for _ in range(self.rows)]
        delta = 0
        states = [(r, c) for r in range(self.rows) for c in range(self.cols)]

        for i in range(0,self.MAX_ITER):
            U_prev = copy.deepcopy(U)
            delta = 0

            for state in states:
                cur_r, cur_c = state

                # We ignore updating the value for our walls or terminal reward states
                # We ignore walls as we can't move into them due to our transition function

                # For terminal reward states
                # These states have no value for U, as we get 
                # a reward for transitioning *into* them.
                if self.map[cur_r][cur_c] != 0:
                   continue

                # Dirty, unoptimzed value iteration update
                U[cur_r][cur_c] = max( # Take the best action...
                                    sum( # That has the highest expected value..
                                        self.transition(state, a, (next_row, next_col)) * ( # Probability of transition
                                            self.reward((next_row, next_col)) # Transition reward
                                                + self.GAMMA * U[next_row][next_col] # Discounted future reward
                                        ) for next_row, next_col in states) 
                                for a in self.actions)

                # Caculate difference
                delta = max(delta, abs(U_prev[cur_r][cur_c] - U[cur_r][cur_c]))

            # Stop if converge
            if delta < self.CONVERGE_THRESH:
                print("Converged in {0} steps with gamma={1}.".format(i, self.GAMMA))
                break

        return U
    
    def policy_extraction(self, values):
        """
        Extracts a policy from a given value function.

        Args:
            [[float]] : The value function for each state, caculated by value iteration.

        Returns:
            [[str]] : The optimal action for each state (UP, DOWN, LEFT, or RIGHT)

        """
        policy = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        
        for r in range(self.rows):
            for c in range(self.cols):

                # Terminals or walls have None policy value
                if self.map[r][c] != 0:
                    policy[r][c] = None
                    continue

                state = (r, c)
                best_action = None
                best_q = None

                # Evaluate each action's Q value using the bellman equation
                for action in self.actions:
                    q = self.bellman_q_value(state, action, values)
                    
                    if best_q is None or q > best_q:
                        best_q = q
                        best_action = action

                policy[r][c] = best_action

        return policy
    
    def bellman_q_value(self, state, action, values):
        '''
        Computes the Bellman Q-value:

            Q(s, a) = sum_{s'} P(s' | s, a) * ( R(s') + gamma * V(s') )

        where:
            - P(s' | s, a) is the transition probability
            - R(s') is the reward of the next state
            - V(s') is the current value estimate of the next state
            - gamma is the discount factor
        '''
        q = 0.0

        for next_r in range(self.rows):
            for next_c in range(self.cols):
                next_state = (next_r, next_c)

                p = self.transition(state, action, next_state)

                r_next = self.reward(next_state)
                v_next = values[next_r][next_c]

                q += p * (r_next + self.GAMMA * v_next)

        return q

    """
    Printer Functions:
    Pretty print functions were generated by modified by Claude 4 Sonnet and then modified on 11/23/25
    """
    def print_values(self, values):
        """
        Prints the value function.
        Highlights terminal states and walls.

        Args:
            values [[float]]: The value function by value iteration.

        Returns:
            None

        """

        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if self.map[r][c] == -1:
                    row_str.append("  WALL    ")
                elif self.map[r][c] != 0:
                    row_str.append("  TERMINAL")
                else:
                    row_str.append(f"{values[r][c]:10.2e}")
            print(" ".join(row_str))
    
    def print_map(self):
        """
        Prints the gridworld map itself

        Args:
            None

        Returns:
            None

        """
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                row_str.append(f"{self.map[r][c]:3}")
            print(" ".join(row_str))

    def print_policy(self, policy):
        """
        Prints the policy function. Prints "N" for no policy set.

        Args:
            values [[str]]: The policy to be printed.

        Returns:
            None
        """

        arrows = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}
        for r in range(self.rows):
            row_str = []
            for c in range(self.cols):
                if self.map[r][c] == -1:
                    row_str.append("█")
                elif self.map[r][c] > 0:
                    row_str.append(str(self.map[r][c]))
                elif policy[r][c] == None:
                    row_str.append("N")
                else:
                    row_str.append(arrows[policy[r][c]])
            print(" ".join(row_str))


def main():
    mdp = MDP("./maps/multi_reward.json", gamma = 1)
    print("Map")
    mdp.print_map()

    u = mdp.value_iteration()

    print("\nValues Map")
    mdp.print_values(u)
    policy = mdp.policy_extraction(u)

    print("\nPolicy Map")
    mdp.print_policy(policy)

if __name__ == "__main__":
    main()  