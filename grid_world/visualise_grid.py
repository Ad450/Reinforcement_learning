import pygame
import time
from grid_world_q_learning import Agent, State, START, BOARD_COLS, BOARD_ROWS, BLOCKED_STATES, WIN_STATE

CELL_SIZE = 80  # pixels
MARGIN = 5
WINDOW_ROWS = 5
WINDOW_COLS = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREY = (200, 200, 200)


def visualize_agent(agent, start=START):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_COLS*CELL_SIZE, WINDOW_ROWS*CELL_SIZE))
    pygame.display.set_caption("Grid World Agent Visualization")

    clock = pygame.time.Clock()
    running = True

    state = agent.state = State(state=start)

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        screen.fill(WHITE)

        # Draw grid
        for r in range(1, BOARD_ROWS+1):
            for c in range(1, BOARD_COLS+1):
                rect = pygame.Rect((c-1)*CELL_SIZE, (r-1)*CELL_SIZE, CELL_SIZE-MARGIN, CELL_SIZE-MARGIN)
                color = GREY
                if (r,c) in BLOCKED_STATES:
                    color = BLACK
                elif (r,c) == WIN_STATE:
                    color = GREEN
                elif (r,c) == state.state:
                    color = BLUE
                pygame.draw.rect(screen, color, rect)

        pygame.display.flip()
        clock.tick(1)  # 1 step per second for visualization

        if state.isEnd:
            print("Goal reached!")
            time.sleep(2)
            running = False
            continue

        # Choose action and move
        action = agent.choose_action()
        state.move_to_next_state(action)

    pygame.quit()


if __name__ == '__main__':
    ag = Agent()
    ag.play(50)
    visualize_agent(ag)