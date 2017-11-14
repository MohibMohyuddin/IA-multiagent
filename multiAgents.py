# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to
# http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random
import util

from game import Agent


def closestDot(originPoint, otherPoints):
    """
    It returns the closest point from a origin to one point of a given collection
    """
    closestCost = 9999999

    for point in otherPoints:
        cost = util.manhattanDistance(originPoint, point)
        if cost < closestCost:
            closestCost = cost
            closestPoint = point

    if closestCost is 9999999:
        return None
    else:
        return (closestCost, closestPoint)


class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """

    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(
            gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(
            len(scores)) if scores[index] == bestScore]
        # Pick randomly among the best
        chosenIndex = random.choice(bestIndices)
        #print(legalMoves[chosenIndex])
        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = 0
        phantom_distances = [util.manhattanDistance(
            newPos, ghost_position) for ghost_position in successorGameState.getGhostPositions()]

        closest_food = closestDot(newPos, newFood.asList())
        
        if closest_food is not None:
            score += (-int(closest_food[0]) * 50) - (len(newFood.asList())* 1000)
        if action == 'stop':
            score -= 10000

        closest_phantom = min(phantom_distances)
        if closest_phantom <= 2 and newScaredTimes[phantom_distances.index(closest_phantom)] < closest_phantom:
            score = -float('inf')
        if newScaredTimes[phantom_distances.index(closest_phantom)] > closest_phantom:
            score += 100
        
        #print("ACTION => {} SCORE=> {} --- [ph: {}, cf: {}]").format(action, score, closest_phantom, closest_food)
        "*** YOUR CODE HERE ***"
        return score


def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)
        self.numberOfAgents = None
        self.PACMAN = 0


class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """
    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        self.numberOfAgents = gameState.getNumAgents()

        best_action_score = -float('inf')
        for action in gameState.getLegalActions(self.PACMAN):
            action_score = self.state_value(gameState.generateSuccessor(self.PACMAN, action), 1)
            if action_score > best_action_score:
                best_action_score = action_score
                best_action = action
        
        print("Action: {}, Value: {}".format(best_action, best_action_score))
        return best_action

    def state_value(self, state, layered_depth):
        agent_num = layered_depth%self.numberOfAgents
        """
        We have self.numberOfAgents layers, so our depth will not be the real depth, our real
        depth will be layered_depth/N of agents, in order to consider the response of the ghosts
        as one move!
        """
        if layered_depth >= self.depth * self.numberOfAgents or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        if agent_num == self.PACMAN:
            return self.max_action(state, agent_num, layered_depth)

        if agent_num >= self.PACMAN:
            return self.min_action(state, agent_num, layered_depth)
    

    def min_action(self, state, agent_num, layered_depth):
        """
        We need to modify the proposed min-value funct, as the next state maybe another min_action 
        agent. We use instead our state_value that applies the right function.
        v <- inf
        for a in actions:
            The action gives us a destiantion state
            v <- min(v, state_value(state))
        """
        v = float('inf')
        for agent_action in state.getLegalActions(agent_num):
            destination_state = state.generateSuccessor(agent_num, agent_action)
            v = min(v, self.state_value(destination_state, layered_depth + 1))
        return v
    def max_action(self, state, agent_num, layered_depth):
        """
        Analogue version for max
        """
        v = -float('inf')
        for agent_action in state.getLegalActions(agent_num):
            destination_state = state.generateSuccessor(agent_num, agent_action)
            v = max(v, self.state_value(destination_state, layered_depth + 1))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        self.numberOfAgents = gameState.getNumAgents()
        alpha = -float('inf')
        beta = float('inf')
        best_action_score = -float('inf')
        for action in gameState.getLegalActions(self.PACMAN):
            action_score = self.state_value(gameState.generateSuccessor(self.PACMAN, action), 1, alpha, beta)
            alpha = max(alpha, action_score)
            if action_score > best_action_score:
                best_action_score = action_score
                best_action = action
        
        #print("Action: {}, Value: {}".format(best_action, best_action_score))
        return best_action

    def state_value(self, state, layered_depth, alpha, beta):
        agent_num = layered_depth%self.numberOfAgents
        
        if layered_depth >= self.depth * self.numberOfAgents or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        if agent_num == self.PACMAN:
            return self.max_action(state, agent_num, layered_depth, alpha, beta)

        if agent_num >= self.PACMAN:
            return self.min_action(state, agent_num, layered_depth, alpha, beta)
    

    def min_action(self, state, agent_num, layered_depth, alpha, beta):
        """
        We need to modify the proposed min-value funct, as the next state maybe another min_action 
        agent. We use instead our state_value that applies the right function.
        v <- inf
        for a in actions:
            The action gives us a destiantion state
            v <- min(v, state_value(state))
        """
        v = float('inf')
        for agent_action in state.getLegalActions(agent_num):
            destination_state = state.generateSuccessor(agent_num, agent_action)
            v = min(v, self.state_value(destination_state, layered_depth + 1, alpha, beta))
            
            if v < alpha: return v
            beta = min(beta, v)
        return v

    def max_action(self, state, agent_num, layered_depth, alpha, beta):
        """
        Analogue version for max
        """
        v = -float('inf')
        for agent_action in state.getLegalActions(agent_num):
            destination_state = state.generateSuccessor(agent_num, agent_action)
            v = max(v, self.state_value(destination_state, layered_depth + 1, alpha, beta))
            if v > beta: return v
            alpha = max(alpha, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        self.numberOfAgents = gameState.getNumAgents()

        best_action_score = -float('inf')
        for action in gameState.getLegalActions(self.PACMAN):
            action_score = self.state_value(gameState.generateSuccessor(self.PACMAN, action), 1)
            if action_score > best_action_score:
                best_action_score = action_score
                best_action = action
        
        #print("Action: {}, Value: {}".format(best_action, best_action_score))
        return best_action

    def state_value(self, state, layered_depth):
        agent_num = layered_depth%self.numberOfAgents
        """
        We have self.numberOfAgents layers, so our depth will not be the real depth, our real
        depth will be layered_depth/N of agents, in order to consider the response of the ghosts
        as one move!
        """
        if layered_depth >= self.depth * self.numberOfAgents or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        
        if agent_num == self.PACMAN:
            return self.max_action(state, agent_num, layered_depth)

        if agent_num >= self.PACMAN:
            return self.prob_action(state, agent_num, layered_depth)
    

    def prob_action(self, state, agent_num, layered_depth):
        """
        The prob of each action is the same, so its 1/agent_actions
        v <- 0
        for a in actions:
            The action gives us a destiantion state
            v <- prob state dest * state_value(state)
        """
        v = 0
        number_actions = len(state.getLegalActions(agent_num))
        for agent_action in state.getLegalActions(agent_num):
            destination_state = state.generateSuccessor(agent_num, agent_action)
            #PSEUDO: v := v + (Probability[child] * expectiminimax(child, depth-1))
            v += (1.0/number_actions * self.state_value(destination_state, layered_depth + 1))
        return v
    
    def max_action(self, state, agent_num, layered_depth):
        """
        Analogue version for max
        """
        v = -float('inf')
        for agent_action in state.getLegalActions(agent_num):
            destination_state = state.generateSuccessor(agent_num, agent_action)
            v = max(v, self.state_value(destination_state, layered_depth + 1))
        return v

def food_sorround_fact(point, points):
    score = 0
    FACT = 100
    if((point[0] + 1, point[1]) not in points):
        score += FACT
    if((point[0] - 1, point[1]) not in points):
        score += FACT
    if((point[0], point[1] + 1) not in points):
        score += FACT
    if((point[0], point[1] - 1) not in points):
        score += FACT
    return score
def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: 
        My conditions:
        1. If the state is a win state -> INF
        2. If the state is a lose state -> -INF
        3. PENALIZATIONS: 
            a. If the closest food is far away -> factor = 50 
            b. If we have remaining food -> factor = 1000
            c. PACMAN and the ghost are in the same position -> equals to isLose()
            d. If the ghost is closer than 3 positions -> -INF run away!
        3. BONUS:
            a. If PACMAN have a nearby Scared Ghost -> 3000
            b. If PACMAN eats a nearby Ghost -> INF 

    """
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    newCapsules = currentGameState.getCapsules()
    
    score = 0

    phantom_distances = [util.manhattanDistance(
        newPos, ghost_position) for ghost_position in currentGameState.getGhostPositions()]

    closest_food = closestDot(newPos, newFood.asList())
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return -float('inf')
    if closest_food is not None:
        score += (-int(closest_food[0]) * 50) - (len(newFood.asList())* 1000)
    for capsule in newCapsules:
        score -= (util.manhattanDistance(newPos, capsule) * 200)
    closest_phantom = min(phantom_distances)
    if closest_phantom <= 3 and newScaredTimes[phantom_distances.index(closest_phantom)] < closest_phantom:
        score = -float('inf')
    if newScaredTimes[phantom_distances.index(closest_phantom)] >= closest_phantom:
        score += 3000
        if closest_phantom == newPos:
            score = float('inf')
    
    return score


# Abbreviation
better = betterEvaluationFunction
