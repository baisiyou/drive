# Simulation for origin version of DWA (Dynamic Window Approach) algorithm
# Simualation for a modified dwa algorithm in dynamic environment
import copy
import math
import os
import random
import time

import pandas as pd
import pygame
from matplotlib import pyplot as plt
from pygame.locals import *


# Set random seed
random.seed(0)


class Environment:
    """ Simulation Environment contains moving obstacles 
        and agent which performs collision aovidance algorithm.
    """
    
    def __init__(self, algorithm, num_obstacles):   
        # Set parameters for drawing 
        self.PLAYFIELDCORNERS = (-4.0, -3.0, 4.0, 3.0)   # The region we will fill with obstacles
        self.WIDTH = 1500  # set the width and height of the screen (pixels)
        self.HEIGHT = 1000
        self.size = [self.WIDTH, self.HEIGHT]
        self.black = (20,20,40)
        self.lightblue = (0,120,255)
        self.darkblue = (0,40,160)
        self.red = (255,100,0)
        self.white = (255,255,255)
        self.blue = (0,0,255)
        self.grey = (70,70,70)
        # Screen center will correspond to (x, y) = (0, 0)
        self.k = 160 # pixels per metre for graphics
        self.u0 = self.WIDTH / 2    # Horizontal screen coordinate:     u = u0 + k * x
        self.v0 = self.HEIGHT / 2   # Vertical screen coordinate:       v = v0 - k * y
        # Configure parameters for visulization in pygame
        pygame.init()
        # Initialise Pygame display screen
        self.screen = pygame.display.set_mode(self.size)
        # This makes the normal mouse pointer invisible/visible (0/1) in graphics window
        pygame.mouse.set_visible(1)

        # Our algorithm
        self.alg = algorithm
        # Simulatation time interval
        self.dt = 0.01
        # Initiate obstacles
        self.OBSTACLE_RADIUS = 0.10
        self.OBSTACLE_MAX_VEL = 0.15
        self.init_obstacle_num = num_obstacles 
        # Initiate robot's goal
        self.goal = (self.PLAYFIELDCORNERS[2]-0.5, self.PLAYFIELDCORNERS[3]-0.5)
        # Initiate pose of robot
        self.init_x = self.PLAYFIELDCORNERS[0] + 0.5
        self.init_y = self.PLAYFIELDCORNERS[1] + 0.5
        self.init_theta = 0.0
        self.init_pose = (self.init_x, self.init_y, self.init_theta)
        self.config = (self.dt, self.k, self.u0, self.v0, self.OBSTACLE_RADIUS)
        # Global parameters for evaluate algorithms
        self.sim_times = 0  # num of simulation times
        self.collision_times = 0  # num of collision times during all simulations
        self.avg_tg = 0
        self.avg_distance_travelled = 0
        self.tg_vec = []
        self.vl_vec = []
        self.vr_vec = []
        self.distance_travelled_vec = []
        # Set moving things
        self.reset()
    

    def reset(self):
        """ Reset parameters for next simulation """
        # Reset obstacles
        self.obstacles = []  # obstalces list
        for i in range(self.init_obstacle_num):
            pos_x = random.uniform(self.PLAYFIELDCORNERS[0], self.PLAYFIELDCORNERS[2])
            pos_y = random.uniform(self.PLAYFIELDCORNERS[1], self.PLAYFIELDCORNERS[3])
            vx = random.gauss(0.0, self.OBSTACLE_MAX_VEL)
            vy = random.gauss(0.0, self.OBSTACLE_MAX_VEL)
            obstacle = [pos_x, pos_y, vx, vy]
            self.obstacles.append(obstacle)
        # Reset algorithm
        self.agent = self.alg(self.init_pose, self.config)
        # Reset sim flag 
        self.sim_over = False  # indicates whether a simulation is over or not
        # Reset collision flag
        self.collision_happened = False
        # Reset time to reach the goal
        self.time_to_goal = 0.0
        # Reset distance_travelled
        self.distance_travelled = 0
        # Record robot's history positions and paths
        self.history_positions = []
        self.history_vl_vec = []
        self.history_vr_vec = []
  


    def move_obstacles(self):
        """ Update locations and velocties of moving obstacles"""
        for idx in range(len(self.obstacles)):
            # Update x coordinate
            self.obstacles[idx][0] += self.obstacles[idx][2] * self.dt
            if self.obstacles[idx][0] < self.PLAYFIELDCORNERS[0]:
                self.obstacles[idx][2] = -self.obstacles[idx][2]
            if self.obstacles[idx][0] > self.PLAYFIELDCORNERS[2]:
                self.obstacles[idx][2] = -self.obstacles[idx][2]
            # Update y coordinate
            self.obstacles[idx][1] += self.obstacles[idx][3] * self.dt  
            if self.obstacles[idx][1] < self.PLAYFIELDCORNERS[1]:
                self.obstacles[idx][3] = -self.obstacles[idx][3]
            if self.obstacles[idx][1] > self.PLAYFIELDCORNERS[3]:
                self.obstacles[idx][3] = -self.obstacles[idx][3]

    def draw_obstacles(self):
        """ Draw small dogs as obstacles on screen """
        for (idx, obstacle) in enumerate(self.obstacles):
            color = self.lightblue

            # Dog dimensions (in meters)
            body_radius = 0.1  # radius of the dog's body
            head_radius = 0.05  # radius of the dog's head

            # Convert obstacle position to screen coordinates
            body_center_x = self.u0 + self.k * obstacle[0]
            body_center_y = self.v0 - self.k * obstacle[1]

            # Draw the dog's body as a circle
            pygame.draw.circle(self.screen, color, (int(body_center_x), int(body_center_y)), int(self.k * body_radius),
                               0)  # Fill body

            # Draw the dog's head as a smaller circle above the body
            head_center_x = body_center_x
            head_center_y = body_center_y - int(self.k * (body_radius * 1.2))  # Position head above the body
            pygame.draw.circle(self.screen, color, (int(head_center_x), int(head_center_y)), int(self.k * head_radius),
                               0)  # Fill head

    def draw_goal(self):
        """ Draw a small house as the goal of the robot on screen """
        color = self.red
        goal = self.goal

        # House dimensions
        house_width = 0.5  # width of the house
        house_height = 0.3  # height of the house
        roof_height = 0.2  # height of the roof

        # Convert goal position to screen coordinates
        house_top_left_x = self.u0 + self.k * (goal[0] - house_width / 2)
        house_top_left_y = self.v0 - self.k * (goal[1] + house_height / 2)

        # Draw the house base as a rectangle
        house_rect = pygame.Rect(int(house_top_left_x), int(house_top_left_y), int(self.k * house_width),
                                 int(self.k * house_height))
        pygame.draw.rect(self.screen, color, house_rect, 0)  # Fill the house rectangle

        # Draw the roof as a triangle
        roof_points = [
            (int(house_top_left_x), int(house_top_left_y)),  # Left bottom corner of the house
            (int(house_top_left_x + house_width / 2 * self.k), int(house_top_left_y - self.k * roof_height)),
            # Top point of the roof
            (int(house_top_left_x + house_width * self.k), int(house_top_left_y)),  # Right bottom corner of the house
        ]
        pygame.draw.polygon(self.screen, color, roof_points)  # Draw the roof

    def draw_robot(self):
        """ Draw humanoid robot on screen with a nurse's hat """
        color = self.white
        position = (self.agent.x, self.agent.y)  # robot's position (x, y)

        # Body dimensions (in meters)
        body_width = 0.2
        body_height = 0.4

        # Head dimensions
        head_radius = 0.1

        # Limb lengths
        arm_length = 0.2
        leg_length = 0.3

        # Convert body position to screen coordinates
        body_top_left_x = self.u0 + self.k * (position[0] - body_width / 2)
        body_top_left_y = self.v0 - self.k * (position[1] + body_height / 2)

        # Draw body as a rectangle
        body_rect = pygame.Rect(int(body_top_left_x), int(body_top_left_y), int(self.k * body_width),
                                int(self.k * body_height))
        pygame.draw.rect(self.screen, color, body_rect, 3)

        # Draw head as a circle above the body
        head_center_x = self.u0 + self.k * position[0]
        head_center_y = body_top_left_y - self.k * head_radius * 2  # Head is above the body
        pygame.draw.circle(self.screen, color, (int(head_center_x), int(head_center_y)), int(self.k * head_radius), 3)

        # Draw nurse hat on top of the head (a small rectangle above the head)
        hat_width = 0.15  # width of the hat
        hat_height = 0.05  # height of the hat
        hat_top_left_x = head_center_x - int(self.k * hat_width / 2)
        hat_top_left_y = head_center_y - int(self.k * head_radius * 2.5)  # Above the head by a small margin
        pygame.draw.rect(self.screen, color,
                         pygame.Rect(hat_top_left_x, hat_top_left_y, int(self.k * hat_width), int(self.k * hat_height)),
                         3)

        # Draw arms (lines from the sides of the body)
        left_arm_start = (body_top_left_x, body_top_left_y + int(self.k * (body_height / 2)))
        right_arm_start = (
        body_top_left_x + int(self.k * body_width), body_top_left_y + int(self.k * (body_height / 2)))
        left_arm_end = (left_arm_start[0] - int(self.k * arm_length), left_arm_start[1])
        right_arm_end = (right_arm_start[0] + int(self.k * arm_length), right_arm_start[1])
        pygame.draw.line(self.screen, color, left_arm_start, left_arm_end, 3)
        pygame.draw.line(self.screen, color, right_arm_start, right_arm_end, 3)

        # Draw legs (lines from the bottom of the body)
        left_leg_start = (body_top_left_x + int(self.k * (body_width / 4)), body_top_left_y + int(self.k * body_height))
        right_leg_start = (
        body_top_left_x + int(self.k * (3 * body_width / 4)), body_top_left_y + int(self.k * body_height))
        left_leg_end = (left_leg_start[0], left_leg_start[1] + int(self.k * leg_length))
        right_leg_end = (right_leg_start[0], right_leg_start[1] + int(self.k * leg_length))
        pygame.draw.line(self.screen, color, left_leg_start, left_leg_end, 3)
        pygame.draw.line(self.screen, color, right_leg_start, right_leg_end, 3)

    def draw_history_trajectory(self):
        """ Draw trajectory of moving robot """
        color = self.red
        # Draw locations
        for pos in self.history_positions:
            pygame.draw.circle(self.screen, color, (int(self.u0 + self.k * pos[0]), int(self.v0 - self.k * pos[1])), 3, 0)
    
    def draw_predicted_tracjetory(self, predicted_path_to_draw):
        # Draw paths(straight lines or arcs)
        for path in predicted_path_to_draw:
            if path[0] == 0:  # Straight line
                straightpath = path[1]
                linestart = (self.u0 + self.k * self.agent.x, self.v0 - self.k * self.agent.y)
                lineend = (self.u0 + self.k * (self.agent.x + straightpath * math.cos(self.agent.theta)), self.v0 - self.k * (self.agent.y + straightpath * math.sin(self.agent.theta)))
                pygame.draw.line(self.screen, (0, 200, 0), linestart, lineend, 1)
            if path[0] == 1:  # Rotation, nothing to draw
                pass
            if path[0] == 2:  # General case: circular arc
                # path[2] and path[3] are start and stop angles for arc but they need to be in the right order to pass
                if (path[3] > path[2]):
                    startangle = path[2]
                    stopangle = path[3]
                else:
                    startangle = path[3]
                    stopangle = path[2]
                # Pygame arc doesn't draw properly unless angles are positive
                if (startangle < 0):
                    startangle += 2*math.pi
                    stopangle += 2*math.pi
                if (path[1][1][0] > 0 and path[1][0][0] > 0 and path[1][1][1] > 1):
                    #print (path[1], startangle, stopangle)
                    pygame.draw.arc(self.screen, (0, 200, 0), path[1], startangle, stopangle, 1)


    def draw_frame(self, predicted_path_to_draw):
        """ Draw each frame of simulation on screen"""
        # Set pygame
        Eventlist = pygame.event.get()
        # Start drawing
        self.screen.fill(self.black)  # Set screen background color
        self.draw_goal()
        self.draw_obstacles()
        self.draw_robot()
        self.draw_history_trajectory()
        self.draw_predicted_tracjetory(predicted_path_to_draw)
        # Update display
        pygame.display.flip()

    
    def check_collsion(self):
        """ Check if collision happened between robot and moving obstacles """
        for obstacle in self.obstacles:
            ox, oy = obstacle[0], obstacle[1]
            dist = math.sqrt((ox-self.agent.x)**2 + (oy-self.agent.y)**2)
            if dist < self.agent.ROBOT_RADIUS + self.OBSTACLE_RADIUS:  # Collision happened
                self.collision_times += 1
                self.collision_happened = True


    def run(self):
        """ Do simulation """
        if self.sim_over == True:
            self.reset()
        while self.sim_over == False:
            # Start simulation
            self.time_to_goal += self.dt
            self.distance_travelled += self.agent.linear_vel * self.dt
            predicted_path_to_draw = []
            # Save robot's locations for display of trail
            self.history_positions.append((self.agent.x, self.agent.y))
            # Planning velocities and path
            predicted_path_to_draw, vLchosen, vRchosen = self.agent.planning(self.goal, self.obstacles)
            self.history_vl_vec.append(vLchosen)
            self.history_vr_vec.append(vRchosen)
            # Visualization
            self.draw_frame(predicted_path_to_draw)
            # Check collison
            self.check_collsion()
            dist_to_goal = math.sqrt((self.agent.x-self.goal[0])**2 + (self.agent.y-self.goal[1])**2)
            if  self.collision_happened == True:
                self.sim_over = True
                self.sim_times += 1
                print('#{} \t Failure \t [tg:  None] \t  [total collision times:{}]'.format(self.sim_times, self.collision_times))
                # time.sleep(1)
                break
            elif round(dist_to_goal, 3) < self.agent.ROBOT_RADIUS:
                self.sim_over = True
                self.sim_times += 1
                self.tg_vec.append(self.time_to_goal)
                self.distance_travelled_vec.append(self.distance_travelled)
                self.vl_vec.extend(self.history_vl_vec)
                self.vr_vec.extend(self.history_vr_vec)
                print( '#{} \t Success \t [tg:  {:.2f}] \t [total collision times:{}]'.format(self.sim_times, self.time_to_goal, self.collision_times))
                break
            else:
                # Continue simualtion
                self.move_obstacles()  # Move obstacles
                self.agent.move_robot()  # Move robot
    

class DWA:
    """ Collision avoidance algorithm """
    def __init__(self, init_pose=(0, 0, 0), config=(0.10, 0, 0, 0)):
        # parameters of robot
        self.ROBOT_RADIUS = 0.1
        # Linear velocity limits
        self.MAX_VEL_LINEAR = 0.8     # ms^(-1) max speed of each wheel
        self.MAX_ACC_LINEAR = 0.3     # ms^(-2) max rate we can change speed of each wheel
        # Angular velocity limits
        self.MAX_VEL_ANGULAR = 0.8
        self.MAX_ACC_ANGULAR = 1.0
        # Current linear velocity and angular velocity
        self.linear_vel = 0.0
        self.angular_vel = 0.0
        # Current positions
        self.x, self.y, self.theta = init_pose
        # Parameters for prediction trajectory
        self.dt, self.k, self.u0, self.v0, self.OBSTACLE_RADIUS = config
        self.STEPS_AHEAD_TO_PLAN = 10
        self.TAU = self.dt * self.STEPS_AHEAD_TO_PLAN
        # Safe distance between robot and closest obstacle after minus robot's radius and obstacle's radius
        self.SAFE_DIST = self.ROBOT_RADIUS   #  
        # Weights for predicted trajectory evaluation
        self.heading_gain = 2  #50
        self.dist_obstacle_gain = 5  #35
        self.vel_gain = 1  #2  


    def predict_position(self, vLpossible, vRpossible, delta_time):
        """ Predict robot's position in delta_time  
            
            @param vLpossible:    possible linear velocity

            @param vRpossible:    possible angular velocity

            @return:    (new_x, new_y, new_theta, path)
        """

        # Go straight line
        if round(vRpossible, 3) == 0:
            new_x = self.x + vLpossible*delta_time*math.cos(self.theta)
            new_y = self.y + vLpossible*delta_time*math.sin(self.theta)
            new_theta = self.theta
            path = (0, vLpossible*delta_time)  # 0 indciates pure translation
        # Pure rotation motion
        elif round(vLpossible, 3) == 0:
            new_x = self.x
            new_y = self.y
            new_theta = self.theta + vRpossible*delta_time
            path = (1, 0)  # 1 indicates pure rotation
        else:
            # Rotation and arc angle of general circular motion
            R = vLpossible / vRpossible
            delta_theta = vRpossible * delta_time
            new_x = self.x + R * (math.sin(delta_theta + self.theta) - math.sin(self.theta))
            new_y = self.y - R * (math.cos(delta_theta + self.theta) - math.cos(self.theta))
            new_theta = self.theta + delta_theta

            # Calculate parameters for drawing arc
            # We need center of circle
            (cx, cy) = (self.x - R * math.sin(self.theta), self.y + R * math.cos(self.theta))
            # Turn this into  Rect
            Rabs = abs(R)
            ((tlx, tly), (Rx, Ry)) = ((int(self.u0 + self.k * (cx - Rabs)), int(self.v0 - self.k * (cy + Rabs))), (int(self.k * (2 * Rabs)), int(self.k * (2 * Rabs))))
            if (R > 0):
                start_angle = self.theta - math.pi/2.0
            else:
                start_angle = self.theta + math.pi/2.0
            stop_angle = start_angle + delta_theta
            path = (2, ((tlx, tly), (Rx, Ry)), start_angle, stop_angle) # 2 indicates general motion
        
        return (new_x, new_y, new_theta, path)


    def calculateClosestObstacleDistance(self, predict_x, predict_y, obstacles):
        """ Calculate  distance to closest obstacle 
            
            @param predict_x: predicted x coordiante of robot

            @param predict_y: predicted y coordiante of robot

            @param obstacles: contains obstacles' information,that is [pos_x, pos_y, vx, vy] 

            @return: distance between robot and closest obstacle
        """
        closestdist = 100000.0  
        for (idx, obstacle) in enumerate(obstacles):
            dx = obstacle[0] - predict_x
            dy = obstacle[1] - predict_y
            d = math.sqrt(dx**2+dy**2)
            # Distance between closest touching point of circular robot and circular obstacle
            dist = d - self.ROBOT_RADIUS - self.OBSTACLE_RADIUS
            if dist < closestdist:
                closestdist = dist

        return closestdist 


    def planning(self, goal, obstacles):
        """ Planning trajectory and select linear and angular velocities for robot
            
            @param goal:  goal postion of robot

            @param obstacles:  [pos_x, pos_y, vx, vy] of each obstacles
        
            @return:  predicted_path_to_draw
        """
        bestBenefit = -100000
        # Range of possible motions: each of vL and vR could go up or down a bit
        sample_num = 10
        vLUpBound = self.linear_vel + self.MAX_ACC_LINEAR * self.dt
        vLDownBound = self.linear_vel - self.MAX_ACC_LINEAR * self.dt
        vLpossiblearray = tuple(vLDownBound + i/(sample_num-1)*(vLUpBound-vLDownBound) for i in range(sample_num))
        # print('vLpossiblearray:', tuple(vLpossiblearray))
        vRUpBound = self.angular_vel + self.MAX_ACC_ANGULAR * self.dt
        vRDownBound = self.angular_vel - self.MAX_ACC_ANGULAR * self.dt
        vRpossiblearray = tuple(vRDownBound + i/(sample_num-1)*(vRUpBound-vRDownBound) for i in range(sample_num))
        # print('vRpossiblearray:', tuple(vRpossiblearray))
        # vLpossiblearray = (self.linear_vel - self.MAX_ACC_LINEAR * self.dt, self.linear_vel, self.linear_vel + self.MAX_ACC_LINEAR * self.dt)
        # vRpossiblearray = (self.angular_vel - self.MAX_ACC_ANGULAR * self.dt, self.angular_vel, self.angular_vel + self.MAX_ACC_ANGULAR * self.dt)
        vLchosen = 0
        vRchosen = 0
        predicted_path_to_draw = []
        # Record for normalize
        costVec = []
        sum_term1 = 0
        sum_term2 = 0
        sum_term3 = 0
        for vLpossible in vLpossiblearray:
            for vRpossible in vRpossiblearray:
                # Check if in veolicties's range
                if vLpossible <= self.MAX_VEL_LINEAR and vRpossible <= self.MAX_VEL_ANGULAR and vLpossible >= 0 and vRpossible >= -self.MAX_VEL_ANGULAR:
                    # Predict robot's new position in TAU seconds
                    predict_x, predict_y, predict_theta, path = self.predict_position(vLpossible, vRpossible, self.TAU)
                    predicted_path_to_draw.append(path)
                    # Calculate how much close we've moved to target location
                    # previousTargetDistance = math.sqrt((self.x - goal[0])**2 + (self.y - goal[1])**2)
                    # newTargetDistance = math.sqrt((predict_x - goal[0])**2 + (predict_y - goal[1])**2)
                    # distanceForward = previousTargetDistance - newTargetDistance
                    # Cost term about  heading angle between predicted robot location and goal point
                    alpha = math.atan2(goal[1]-predict_y, goal[0]-predict_x)
                    alpha = alpha+2*math.pi if alpha < 0 else alpha
                    k = int(abs(predict_theta) / (2*math.pi))
                    beta = predict_theta+(k+1)*(2*math.pi) if predict_theta < 0 else predict_theta-k*(2*math.pi)
                    if beta > math.pi:
                        beta = beta - 2*math.pi
                    
                    assert alpha >= 0, "ERROR:  alpha angle is negative [alpha={:.4f}]".format(alpha)
                    # assert beta >= 0, "ERROR:  beta angle is negative [beta={:.4f}]".format(beta)
                    if beta > 0:
                        heading_angle = abs(alpha - beta)
                    else:
                        heading_angle = alpha + abs(beta)
                    # print('[Heading angle is {:.2f} degrees]\t[alpha={:.2f}]\t[beta={:.2f}]\t[theta={:.2f}]'.format(heading_angle/math.pi*180, alpha/math.pi*180, beta/math.pi*180, predict_theta/math.pi*180))
                    heading_angle_term = math.pi - heading_angle
                    sum_term1 += heading_angle_term
                    # Cost term about distance to goal for evaluation
                    # dist_goal_cost  =  self.forward_gain * distanceForward
                    # Cost term about distance to closest obstacle for evaluation
                    dist_obstacle_term = self.calculateClosestObstacleDistance(predict_x, predict_y, obstacles)
                    sum_term2 += dist_obstacle_term
                    # if distanceToObstacle < self.SAFE_DIST:
                    #     dist_obstacle_cost = self.obstacle_gain * (self.SAFE_DIST - distanceToObstacle)
                    # else:
                    #     dist_obstacle_cost = 0
                    sum_term3 += vLpossible
                    costVec.append((heading_angle_term, dist_obstacle_term, vLpossible, vRpossible))        
                    # Total cost
                    # benefit = dist_goal_cost - dist_obstacle_cost
                    # if benefit > bestBenefit:
                    #     vLchosen = vLpossible
                    #     vRchosen = vRpossible
                    #     bestBenefit = benefit
        # Normalize cost terms
        for ele in costVec:
            if ele[1] < 0:
                continue
            if round(ele[2], 3) > round(math.sqrt(2*self.MAX_ACC_LINEAR*ele[1]), 3):
                # Cannot reduce to zero before collision
                continue
            heading_angle_term = ele[0] / sum_term1
            dist_obstacle_term = ele[1] / sum_term2
            vel_term = ele[2] / sum_term3
            benefit  =  self.heading_gain*heading_angle_term + self.dist_obstacle_gain*dist_obstacle_term + self.vel_gain*vel_term
            if benefit > bestBenefit:
                bestBenefit = benefit
                vLchosen = ele[2]
                vRchosen = ele[3]
        # Update velocities
        self.linear_vel = vLchosen
        self.angular_vel = vRchosen
        # print('[vLchosen:\t{:.3f}]\t[vRchosen:\t{:.3f}]'.format(vLchosen, vRchosen))


        # Return path to draw
        return predicted_path_to_draw, vLchosen, vRchosen


    def move_robot(self):
        """ Move robot based on chosen velocities in dt time"""
        self.x, self.y, self.theta, tmp_path = self.predict_position(self.linear_vel, self.angular_vel, self.dt) 

    

       
if __name__ == '__main__':
    env = Environment(DWA, 20)
    while env.sim_times < 100:
        env.run()
        
    env.avg_tg = sum(env.tg_vec)/len(env.tg_vec)
    env.avg_distance_travelled = sum(env.distance_travelled_vec)/len(env.distance_travelled_vec)
  
    # Save tg_vec into csv file
    tg_file = pd.DataFrame(data=env.tg_vec, columns=["tg"])
    tg_file.to_csv("static_goal_orgin_dwa_tg{}.csv".format(env.init_obstacle_num))
    vel_vec = [[env.vl_vec[idx], env.vr_vec[idx]] for idx in range(len(env.vl_vec))]
    vel_file = pd.DataFrame(data=vel_vec, columns=["vl", "vr"])
    vel_file.to_csv("static_goal_origin_dwa_vel{}.csv".format(env.init_obstacle_num))
    # sort tg_vec
    tg_vec_sorted = sorted(env.tg_vec)
    tg_75th = tg_vec_sorted[int(0.75*len(tg_vec_sorted))-1]
    tg_90th = tg_vec_sorted[int(0.90*len(tg_vec_sorted))-1]

    # Save printed varibles into txt
    res_str1 = '[Collision Rate: {}/{}={:.2f}%] \n'.format(env.collision_times, env.sim_times, env.collision_times/env.sim_times*100)
    res_str2 = '[Average time to goal: {:.2f} secs] \t [tg_75th: {:.2f} secs] \t [tg_90th: {:.2f} secs] \n'.format(env.avg_tg, tg_75th, tg_90th)
    res_str3 = '[Average distance travelled to goal: \t {:.2f} meters] \n'.format(env.avg_distance_travelled)
    to_txt = res_str1 + res_str2 + res_str3
    print("\n" + "* "*30 + "\n")
    print(to_txt)
    print("\n" + "* "*30 + "\n")
    with open('static_orgin_dwa_result{}.txt'.format(env.init_obstacle_num), 'w') as f:
        f.write(to_txt)
    # Plot
    plt.figure()
    plt.plot(list(range(len(env.tg_vec))), env.tg_vec, 'bo', list(range(len(env.tg_vec))), env.tg_vec, 'k')
    plt.title("time to goal")
    plt.xlabel("simulation times")
    plt.ylabel("tg(sec)")
    plt.figure()
    plt.subplot(211)
    plt.plot(list(range(len(env.vl_vec))), env.vl_vec, 'r', label="linear velocity")
    plt.xlabel("simulation steps")
    plt.ylabel("vl(m/s)")
    plt.legend(loc="upper right")
    plt.subplot(212)
    plt.plot(list(range(len(env.vr_vec))), env.vr_vec, 'g', label="angular velocity")
    plt.xlabel("simulation steps")
    plt.ylabel("vr(rad/s)")
    plt.legend(loc="upper right")
    plt.show()