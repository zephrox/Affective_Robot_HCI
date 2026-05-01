import pygame
import math

class RobotSim:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("TUI Robot Arm Simulator")
        self.clock = pygame.time.Clock()
        
        # Link arm properties
        self.origin = (width // 2, height - 100)
        self.seg_lengths = [100, 120, 80] # Shoulder, Elbow, Wrist segments
        self.current_angles = [90, 90, 90, 90] # Starting position
        self.lerp_speed = 0.05 # How fast the arm moves (0.01 to 1.0)

    def draw_arm(self, target_angles, led_color):
        # 1. Smoothly interpolate current angles toward target angles
        for i in range(len(self.current_angles)):
            diff = target_angles[i] - self.current_angles[i]
            self.current_angles[i] += diff * self.lerp_speed

        # 1. Draw Ambient LED Background Glow
        self.screen.fill((20, 20, 20))
        glow = pygame.Surface((800, 600), pygame.SRCALPHA)
        pygame.draw.circle(glow, (*led_color, 40), self.origin, 400)
        self.screen.blit(glow, (0, 0))

        # Base, Shoulder, Elbow, Gripper (0-180)
        # We simplify 4 servos into a 2D side-view [Base is ignored in 2D height]
        _, s_angle, e_angle, g_angle = self.current_angles

        # Convert servo degrees to radians for math (inverted for Pygame Y-axis)
        s_rad = math.radians(s_angle - 180)
        e_rad = s_rad + math.radians(e_angle - 90)

        # 2. Calculate Joint Positions
        joint1 = (self.origin[0] + self.seg_lengths[0] * math.cos(s_rad),
                  self.origin[1] + self.seg_lengths[0] * math.sin(s_rad))
        
        joint2 = (joint1[0] + self.seg_lengths[1] * math.cos(e_rad),
                  joint1[1] + self.seg_lengths[1] * math.sin(e_rad))

        # 3. Render Arm Segments
        pygame.draw.line(self.screen, (200, 200, 200), self.origin, joint1, 15) # Shoulder
        pygame.draw.line(self.screen, (180, 180, 180), joint1, joint2, 10)     # Elbow
        
        # 4. Render Joints and Gripper
        pygame.draw.circle(self.screen, (255, 255, 255), (int(self.origin[0]), int(self.origin[1])), 20) # Base
        pygame.draw.circle(self.screen, (100, 100, 100), (int(joint1[0]), int(joint1[1])), 12)
        
        # Gripper visual based on angle
        grip_size = int(10 + (g_angle / 180) * 20)
        pygame.draw.rect(self.screen, (255, 255, 0), (joint2[0]-5, joint2[1]-5, grip_size, 10))

        pygame.display.flip()

    def quit(self):
        pygame.quit()