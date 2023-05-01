import pygame
import pygame.gfxdraw
import random

def draw_bezier_loop(window_width, window_height, num_points):
    # Generate random control points near the edges of the window
    margin = min(window_width, window_height) * 0.1
    control_points = [(random.uniform(margin, window_width - margin),
                       random.uniform(margin, margin)),
                      (random.uniform(window_width - margin, window_width),
                       random.uniform(margin, window_height - margin)),
                      (random.uniform(margin, window_width - margin),
                       random.uniform(window_height - margin, window_height)),
                      (random.uniform(margin, margin),
                       random.uniform(margin, window_height - margin))]

    # Draw the Bezier curve
    white = (255, 255, 255)
    thickness = 3
    pygame.init()
    window = pygame.display.set_mode((window_width, window_height))
    pygame.display.set_caption("Bezier Loop")
    clock = pygame.time.Clock()
    done = False

    while not done:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                done = True

        # Draw the Bezier curve
        pygame.gfxdraw.bezier(window, control_points, thickness, white)

        # Draw the control points
        for point in control_points:
            pygame.draw.circle(window, white, (int(point[0]), int(point[1])), thickness)

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

def main():
    draw_bezier_loop(800, 600, 10)

if __name__ == "__main__":
    main()
