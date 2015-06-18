import random
import math

from deap import base, creator, tools


def drawBoard(board):
    # This function prints out the board that it was passed.

    board = [item or ' ' for item in board]

    message = ""
    # "board" is a list of 10 strings representing the board (ignore index 0)
    message += ('   |   |\n')
    message += (' ' + board[6] + ' | ' + board[7] + ' | ' + board[8] + '\n')
    message += ('   |   |\n')
    message += ('-----------\n')
    message += ('   |   |\n')
    message += (' ' + board[3] + ' | ' + board[4] + ' | ' + board[5] + '\n')
    message += ('   |   |\n')
    message += ('-----------\n')
    message += ('   |   |\n')
    message += (' ' + board[0] + ' | ' + board[1] + ' | ' + board[2] + '\n')
    message += ('   |   |')
    return message


def get_game_board(game_choices):
    board = [None] * 9
    player = game_choices.player_starts and "O" or "X"
    iteration = 0

    while True:
        if iteration > len(game_choices) - 2:
            break

        if player == "X":
            move = getComputerMove(board)
            board[move] = player
            player = "O"
        else:
            iteration += 1
            move = min(0, max(int(math.floor(game_choices[iteration])), 9))
            board[move] = player
            player = "X"

        if isWinner(board, "O") or isWinner(board, "X"):
            break

        if isBoardFull(board):
            break

    return board, iteration


def isBoardFull(board):
    # Return True if every space on the board has been taken. Otherwise return False.
    for i in range(0, 9):
        if isSpaceFree(board, i):
            return False
    return True


def getBoardCopy(board):
    # Make a duplicate of the board list and return it the duplicate.
    dupeBoard = []

    for i in board:
        dupeBoard.append(i)

    return dupeBoard


def isSpaceFree(board, move):
    # Return true if the passed move is free on the passed board.
    return board[move] is None


def makeMove(board, letter, move):
    board[move] = letter


def isWinner(bo, le):
    # Given a board and a player's letter, this function returns True if that player has won.
    # We use bo instead of board and le instead of letter so we don't have to type as much.
    return (
        (bo[6] == le and bo[7] == le and bo[8] == le) or  # across the top
        (bo[3] == le and bo[4] == le and bo[5] == le) or  # across the middle
        (bo[0] == le and bo[1] == le and bo[2] == le) or  # across the bottom
        (bo[6] == le and bo[3] == le and bo[0] == le) or  # down the left side
        (bo[7] == le and bo[4] == le and bo[1] == le) or  # down the middle
        (bo[8] == le and bo[5] == le and bo[2] == le) or  # down the right side
        (bo[6] == le and bo[4] == le and bo[2] == le) or  # diagonal
        (bo[8] == le and bo[4] == le and bo[0] == le)
    )  # diagonal


def chooseRandomMoveFromList(board, movesList):
    # Returns a valid move from the passed list on the passed board.
    # Returns None if there is no valid move.
    possibleMoves = []
    for i in movesList:
        if isSpaceFree(board, i):
            possibleMoves.append(i)

    if len(possibleMoves) != 0:
        return random.choice(possibleMoves)
    else:
        return None


def getComputerMove(board):
    computerLetter = 'X'
    playerLetter = 'O'

    # Here is our algorithm for our Tic Tac Toe AI:
    # First, check if we can win in the next move
    for i in range(0, 9):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, computerLetter, i)
            if isWinner(copy, computerLetter):
                return i

    # Check if the player could win on his next move, and block them.
    for i in range(0, 9):
        copy = getBoardCopy(board)
        if isSpaceFree(copy, i):
            makeMove(copy, playerLetter, i)
            if isWinner(copy, playerLetter):
                return i

    # Try to take one of the corners, if they are free.
    move = chooseRandomMoveFromList(board, [0, 2, 6, 8])
    if move is not None:
        return move

    # Try to take the center, if it is free.
    if isSpaceFree(board, 4):
        return 5

    # Move on one of the sides.
    return chooseRandomMoveFromList(board, [1, 3, 5, 7])


class Fitness(float):
    def __init__(self, *args, **kw):
        super(Fitness, self).__init__(*args, **kw)
        self.board = None
        self.iterations = 0


class Individual(list):
    def __init__(self, *args, **kw):
        super(Individual, self).__init__(*args, **kw)
        self.fitness = None
        self.player_starts = True

    def __str__(self):
        if self.fitness is None or self.fitness.board is None:
            return "No game to be found for this individual"
        else:
            return drawBoard(self.fitness.board)


class GameSolver:
    IND_SIZE = 10
    CXPB, MUTPB = 0.5, 0.2

    def __init__(self):
        self.initialize_creator()
        self.initialize_toolbox()

    def initialize_creator(self):
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

    def initialize_toolbox(self):
        self.toolbox = base.Toolbox()
        self.toolbox.register("individual", self.get_random_game)
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)

        self.toolbox.register("mate", tools.cxTwoPoint)
        self.toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        self.toolbox.register("evaluate", self.determine_fitness)

    def determine_fitness(self, game_choices):
        board, iterations = get_game_board(game_choices)
        distance = 5 - iterations
        won = isWinner(board, "O") and 17.0 or 0.0

        fit = Fitness((distance * 3.0 + won) / 20.0)
        fit.board = board
        fit.iterations = iterations

        return fit

    def get_random_game(self):
        ind = Individual([random.randint(0, 8) for i in range(5)])
        ind.player_starts = [random.randint(0, 1)] == 0
        return ind

    def get_top_solutions(self, population_size, generations):
        pop = self.toolbox.population(n=population_size)

        # Evaluate the entire population
        fitnesses = map(self.toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness = fit

        for g in xrange(generations):
            offspring = self.toolbox.select(pop, len(pop))
            offspring = map(self.toolbox.clone, offspring)

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < self.CXPB:
                    self.toolbox.mate(child1, child2)
                    child1.fitness = None
                    child2.fitness = None

            for mutant in offspring:
                if random.random() < self.MUTPB:
                    self.toolbox.mutate(mutant)
                    mutant.fitness = None

            invalid_ind = [ind for ind in offspring if ind.fitness is None]
            fitnesses = map(self.toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness = fit

            pop[:] = offspring

        return pop[:10]


def main():
    solver = GameSolver()
    for game in solver.get_top_solutions(population_size=20, generations=2000):
        print
        print
        print game

if __name__ == "__main__":
    main()
