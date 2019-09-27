# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

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
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

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
        distance=[]

        gDistance = []

        for state in newGhostStates:
            gDistance.append(manhattanDistance(state.getPosition(), newPos))

        for food in successorGameState.getFood():
            distance.append( manhattanDistance(currentGameState.getPacmanPosition(), food))

        if max(distance) > 5 and min(gDistance) > 5:
             return successorGameState.getScore() - 10 * max(distance)
        elif min(gDistance) < 5:
            return successorGameState.getScore() + 10000 * min(distance) + 1 / 10000 * max(gDistance)
        if min(gDistance) > 10:
            if (max(distance)) > 5:
                return successorGameState.getScore() - 10 * max(distance)
            return successorGameState.getScore() + 10000 * min(distance) - 1 / 10000 * min(gDistance)
        else:
            return successorGameState.getScore() + 10000 * min(distance)

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

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

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

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        v = -float("inf")
        a = Directions.STOP
        for i in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, i)
            value = self.minimizer(state, 0, 1)
            if value > v:
                v = value
                a = i
        return a

    def maximizer(self, state, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float("-inf")
        for action in state.getLegalActions(0):
            v = max(v, self.minimizer(state.generateSuccessor(0, action), depth, 1))
        return v

    def minimizer(self, state, depth, index):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float("inf")
        for action in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                v = min(v, self.maximizer(state.generateSuccessor(index, action), depth + 1))
            else:
                v = min(v, self.minimizer(state.generateSuccessor(index, action), depth, index + 1))
        return v

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        v = -float("inf")
        alpha = -float("inf")
        beta = float("inf")
        a = Directions.STOP
        for i in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, i)
            value = self.minimizer(state, 0, 1, alpha, beta)
            if value > v:
                v = value
                a = i
            alpha = max(alpha, value)
        return a

    def maximizer(self, gameState, depth, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = float("-inf")
        for action in gameState.getLegalActions(0):
            v = max(v, self.minimizer(gameState.generateSuccessor(0, action), depth, 1, alpha, beta))
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def minimizer(self, gameState, depth, index, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)
        v = float("inf")
        for action in gameState.getLegalActions(index):
            if index == gameState.getNumAgents() - 1:
                v = min(v, self.maximizer(gameState.generateSuccessor(index, action), depth + 1, alpha, beta))
            else:
                v = min(v, self.minimizer(gameState.generateSuccessor(index, action), depth, index + 1, alpha, beta))
            if v < alpha:
                return v
            beta = min(beta, v)
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
        "*** YOUR CODE HERE ***"
        v = -float("inf")
        a = Directions.STOP
        for i in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, i)
            value = self.expectimizer(state, 0, 1)
            if value > v:
                v = value
                a = i
        return a

    def maximizer(self, state, depth):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = float("-inf")
        for action in state.getLegalActions(0):
            v = max(v, self.expectimizer(state.generateSuccessor(0, action), depth, 1))
        return v

    def expectimizer(self, state, depth, index):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        v = 0
        for action in state.getLegalActions(index):
            if index == state.getNumAgents() - 1:
                v += (1 / len(state.getLegalActions(index))) * self.maximizer(state.generateSuccessor(index, action), depth + 1)
            else:
                v += (1 / len(state.getLegalActions(index))) * self.expectimizer(state.generateSuccessor(index, action), depth, index + 1)
        return v

def betterEvaluationFunction(currentGameState):

    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return float("-inf")

    score = currentGameState.getScore()
    newFood = currentGameState.getFood()
    ghostPositions = currentGameState.getGhostPositions()
    pos = currentGameState.getPacmanPosition()
    capsuleList = currentGameState.getCapsules()


    ghostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]

    retVal = 0
    distance = []
    distance.append(0)
    gDistance = []

    scaredGhostDistance = []
    cDistance = []
    cDistance.append(0)

    for food in  newFood:
        distance.append(manhattanDistance(pos, food))

    for cap in  capsuleList:
        cDistance.append(manhattanDistance(cap, food))

    for state in ghostStates:
        if state.scaredTimer != 0:
            scaredGhostDistance.append(manhattanDistance(state.getPosition(), pos))
        else:
            gDistance.append(manhattanDistance(state.getPosition(), pos))

    gDistance.extend(scaredGhostDistance)

    # if ( min(gDistance) > 5):
    #     return max(distance) + min(cDistance)

    if ( min(gDistance) > 5):
        return min (distance)

    if ( min(gDistance) < 2   ):
        return 10 * max(distance) - min(gDistance)

    return min(distance) - min(gDistance) + currentGameState.getScore() + len(capsuleList) - \
           len(gDistance) + len(scaredGhostDistance) + min(cDistance)




# Abbreviation
better = betterEvaluationFunction


