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
        answer = -float("inf")
        if successorGameState.isLose():
            return -float("inf")
        for food in currentGameState.getFood().asList():
            temp = -manhattanDistance(food, successorGameState.getPacmanPosition())
            if temp >= answer:
                answer = temp
        return answer
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
        def max_value(gameState, depth):
            answer = -float("inf")
            depth_zero_answer = Directions.STOP
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            if depth == 0:
                for x in gameState.getLegalActions(0):
                    temp = max(answer,min_value(gameState.generateSuccessor(0, x), depth, 1))
                    if temp > answer:
                        depth_zero_answer = x
                        answer = temp
                return depth_zero_answer
            else:
                for x in gameState.getLegalActions(0):
                    temp = min_value(gameState.generateSuccessor(0, x), depth, 1)
                    if temp > answer:
                        answer = temp
                return answer
        def min_value(gameState, depth, agent):
            answer = float("inf")
            next = agent + 1
            if agent == gameState.getNumAgents() - 1:
                next = 0
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            for x in gameState.getLegalActions(agent):
                if next == 0 and depth == self.depth - 1:
                    temp = min(answer, self.evaluationFunction(gameState.generateSuccessor(agent, x)))
                elif next == 0 and not depth == self.depth - 1:
                    temp = min(answer, max_value(gameState.generateSuccessor(agent, x), depth + 1))
                elif not next == 0:
                    temp = min(answer, min_value(gameState.generateSuccessor(agent, x), depth, next))
                if answer > temp:
                    answer = temp
            return answer

        depth = 0
        return max_value(gameState, depth)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def max_value(gameState, depth, alpha, beta):
            answer = -float("inf")
            depth_zero_answer = Directions.STOP
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            if depth == 0:
                for x in gameState.getLegalActions(0):
                    temp = max(answer, min_value(gameState.generateSuccessor(0, x), depth, 1, alpha, beta))
                    if temp > answer:
                        depth_zero_answer = x
                        answer = temp
                    alpha = max(alpha, answer)
                    if answer > beta:
                        return answer
                return depth_zero_answer
            else:
                for x in gameState.getLegalActions(0):
                    temp = max(answer, min_value(gameState.generateSuccessor(0, x), depth, 1, alpha, beta))
                    if temp > answer:
                        answer = temp
                    alpha = max(alpha, answer)
                    if answer > beta:
                        return answer
                return answer
        def min_value(gameState, depth, agent, alpha, beta):
            answer = float("inf")
            next = agent + 1
            if agent == gameState.getNumAgents() - 1:
                next = 0
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            for x in gameState.getLegalActions(agent):
                if next == 0 and depth == self.depth - 1:
                    temp = min(answer, self.evaluationFunction(gameState.generateSuccessor(agent, x)))
                elif next == 0 and not depth == self.depth - 1:
                    temp = min(answer, max_value(gameState.generateSuccessor(agent, x), depth + 1, alpha, beta))
                elif not next == 0:
                    temp = min(answer, min_value(gameState.generateSuccessor(agent, x), depth, next, alpha, beta))
                if answer > temp:
                    answer = temp
                if answer < beta:
                    beta = answer
                if answer < alpha:
                    return answer
            return answer
        depth = 0
        return max_value(gameState, depth, -float("inf"), float("inf"))

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
        def expectimax(gameState, depth):
            answer = Directions.STOP
            max_score = 0
            for x in gameState.getLegalActions(0):
                prev = max_score
                max_score = max(max_score, expected_value(gameState.generateSuccessor(0, x), depth, 1))
                if max_score > prev:
                    answer = x
            return answer
        def max_value(gameState, depth):
            answer = 0
            if depth == 0:
                return self.evaluationFunction(gameState)
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            for x in gameState.getLegalActions(0):
                answer = max(answer, expected_value(gameState.generateSuccessor(0, x), depth, 1))
            return answer
        def expected_value(gameState, depth, agent):
            list = [None]*len(gameState.getLegalActions(agent))
            counter = 0
            if gameState.isWin():
                return self.evaluationFunction(gameState)
            if gameState.isLose():
                return self.evaluationFunction(gameState)
            for x in gameState.getLegalActions(agent):
                next = gameState.generateSuccessor(agent, x)
                if (agent == (gameState.getNumAgents() - 1)):
                    list[counter] = max_value(next, depth - 1)
                    counter+=1
                else:
                    list[counter] = expected_value(next, depth, agent + 1)
                    counter+=1
            import statistics as s
            return s.mean(list)
        depth = self.depth
        return expectimax(gameState, depth)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I tried to prioritize the closest food so that pacman would always go for the closest food regardless of ghost location etc.
    """
    "*** YOUR CODE HERE ***"
    answer = 0
    if currentGameState.isWin():
        return float("inf")
    if currentGameState.isLose():
        return -float("inf")
    for x in currentGameState.getFood().asList():
        answer = min(answer, -1*manhattanDistance(x, currentGameState.getPacmanPosition()))
    return answer + currentGameState.getScore()
# Abbreviation
better = betterEvaluationFunction
