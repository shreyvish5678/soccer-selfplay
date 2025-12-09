import torch
import pygame

class SoccerGameEnv:
    def __init__(self, num_envs=1000, render=False, max_steps=2000, device="cpu", stagnation_on=True):
        self.num_envs = num_envs # number of parallel environments
        self.render_mode = render # whether to render the game with pygame
        self.max_steps = max_steps # number of maximum updates
        self.device = device 
        self.stagnation_on = stagnation_on # whether to enable stagnation detection and reset, only keep for training

        self.WIDTH = 600 # field width
        self.HEIGHT = 400 # field height
        self.GOAL_WIDTH = 10 # goal width
        self.GOAL_HEIGHT = 150 # goal height

        self.PLAYER_SIZE = 15 # radius of player
        self.BALL_SIZE = 10 # radius of ball

        self.OFFSET = 150 # distance from edge to player

        self.PLAYER_ACCELERATION = 0.5 # acceleration applied to player movement
        self.PLAYER_MAX_SPEED = 6.0 # maximum speed of player
        self.PLAYER_FRICTION = 0.9 # friction applied to player movement

        self.BALL_FRICTION = 0.98 # friction applied to ball movement
        self.BALL_MAX_SPEED = 15.0 # maximum speed of ball
        self.WALL_BOUNCE = 0.8 # ball velocity retained after hitting wall

        self.PLAYER_RESTITUTION = 0.5 # how bouncy the player is
        self.BALL_RESTITUTION = 0.7 # how bouncy the ball is
        self.RANDOM_IMPULSE_NOISE_BALL = 1.5 # magnitude of random noise added to ball velocity after collisions

        self.PLAYER_MASS = 2.0 # mass of player
        self.BALL_MASS = 1.0 # mass of ball

        self.STAGNATION_THRESHOLD = 180 # steps the game has to be stagnant to reset positions
        self.STAGNATION_VELOCITY = 0.5 # ball velocity to determine whether current step is stagnant

        self.CORNER_RADIUS = 30.0 # radius of corner arc
        self.CORNER_FORCE = 1.5 # magnitude of corner arc force applied to ball (prevent ball getting stuck in corner)

        self.SIZE = torch.sqrt(torch.tensor(self.WIDTH**2 + self.HEIGHT**2, device=self.device)) # diagonal size of field

        self.GOAL_REWARD = 100.0 # reward for scoring a goal
        self.BALL_TOWARDS_GOAL_REWARD = 1.0 # reward for moving ball towards opponent's goal
        self.MOVE_CLOSE_TO_BALL_REWARD = 0.5 # reward for moving closer to the ball
        self.COLLISION_BALL_REWARD = 1.0 # reward for colliding with the ball
        self.DISTANCE_PENALTY = 0.5 # penalty for being far from the ball
        self.STAGNATION_RATIO = 0.15 # ratio of max speed below which the ball is considered stagnant
        self.STAGNANT_PENALTY = 0.1 # penalty for each stagnant step

        # each action corresponds to a direction vector and since actions represent applied forces,
        # we scale each direction by player acceleration to convert them into acceleration vectors
        self.action_vectors = torch.tensor([
            [0, 0],   # no movement
            [0, -1],  # up
            [0, 1],   # down
            [-1, 0],  # left
            [1, 0],   # right
            [-1, -1], # up-left
            [1, -1],  # up-right
            [-1, 1],  # down-left
            [1, 1],   # down-right
        ], dtype=torch.float32, device=self.device) * self.PLAYER_ACCELERATION
        self.action_dim = self.action_vectors.shape[0] # number of possible actions

        # if rendering is enabled, initialize pygame and rendering components
        if self.render_mode: 
            self.init_rendering()
            
        self.initialize_state_tensors() # initialize the state tensors (player positions, ball position, velocities, scores, etc.)
        self.reset() # reset the environment

    def init_rendering(self):
        pygame.init() # initialize pygame
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT)) # create pygame window size of the field
        pygame.display.set_caption("Soccer Physics") # set window title
        self.clock = pygame.time.Clock() # clock to control frame rate
        self.font = pygame.font.Font(None, 74) # font for rendering scores

    def initialize_state_tensors(self):
        # x,y positions of player 1 for all environments, initialized to zeros
        self.player_1_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32) # shape: (num_envs, 2)
        self.player_2_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32) # same for player 2
        self.ball_pos = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32) # same for the ball

        # velocity vectors of player 1 for all environments
        self.player_1_vel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32) # shape: (num_envs, 2)
        self.player_2_vel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32) # same for player 2
        self.ball_vel = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float32) # same for the ball

        # scores of player 1 for all environments
        self.score_p1 = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # shape: (num_envs,)
        self.score_p2 = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # same for player 2

        # collision flags for player 1 for all environments
        self.collision_p1 = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # shape: (num_envs,)
        self.collision_p2 = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool) # same for player 2

        self.current_step = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # current step count for all environments, shape: (num_envs,)
        self.stagnation_steps = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # stagnation step count for all environments, shape: (num_envs,)

        # vector for the distances between player 1 and the ball in the previous step for all environments
        self.prev_dist_p1 = None # shape: (num_envs,), currently None set on first reset
        self.prev_dist_p2 = None # same for player 2

        # vector for the distances between the ball and the left goal in the previous step for all environments
        self.prev_ball_dist_left = None # shape: (num_envs,), currently None set on first reset
        self.prev_ball_dist_right = None # same for the right goal, shape: (num_envs,)

    def reset(self, env_ids=None):
        # here env_ids can be inputted if we want to reset only specific environments
        # this is generally used if we are training an agent with multiple environments in parallel
        # and only want to reset the environments that have reached terminal states
        if env_ids is None:
            # this means we want to reset all environments, so we create a tensor of all environment ids
            env_ids = torch.arange(self.num_envs, device=self.device) 

        self.reset_positions(env_ids) # reset positions of players and ball for specified environments
        
        self.score_p1[env_ids] = 0 # reset scores of player 1
        self.score_p2[env_ids] = 0 # same for player 2
        
        self.collision_p1[env_ids] = False # reset collision flags for player 1
        self.collision_p2[env_ids] = False # same for player 2

        self.current_step[env_ids] = 0 # reset current step count for each environment

        return self.get_state() # return the states of all environments
    
    def reset_positions(self, env_ids):
        self.player_1_pos[env_ids, 0] = self.OFFSET # set x position(s) of player 1 offset from left edge
        self.player_2_pos[env_ids, 0] = self.WIDTH - self.OFFSET # same for player 2, but offset from right edge

        self.player_1_pos[env_ids, 1] = self.HEIGHT // 2 # set y position(s) of player 1 to center height
        self.player_2_pos[env_ids, 1] = self.HEIGHT // 2 # same for player 2
        
        # this sets the the ball to the center of the field
        self.ball_pos[env_ids, 0] = self.WIDTH // 2 # set x position(s) of the ball to center width
        self.ball_pos[env_ids, 1] = self.HEIGHT // 2 # set y position(s) of the ball to center height

        self.player_1_vel[env_ids] = 0.0 # reset velocity vectors of player 1
        self.player_2_vel[env_ids] = 0.0 # same for player 2
        self.ball_vel[env_ids] = 0.0 # same for the ball

        self.stagnation_steps[env_ids] = 0 # reset stagnation step count for each environment

        if self.prev_dist_p1 is not None:
            # dim=1 here for the calculations means do across the 2nd dimension in the (num_envs, 2) tensor, 
            # which is the x,y positions, updating all the environments at once but independently
            # this is something we will do a lot to leverage parallel computations for efficiency
            self.prev_dist_p1[env_ids] = torch.norm(self.ball_pos[env_ids] - self.player_1_pos[env_ids], dim=1) # reset previous distances between player 1 and ball
            self.prev_dist_p2[env_ids] = torch.norm(self.ball_pos[env_ids] - self.player_2_pos[env_ids], dim=1) # same for player 2

            goal_left = torch.tensor([self.GOAL_WIDTH, self.HEIGHT / 2], device=self.device) # left goal center position
            goal_right = torch.tensor([self.WIDTH - self.GOAL_WIDTH, self.HEIGHT / 2], device=self.device) # right goal center position
            
            self.prev_ball_dist_left[env_ids] = torch.norm(self.ball_pos[env_ids] - goal_left, dim=1) # reset previous distances between ball and left goal
            self.prev_ball_dist_right[env_ids] = torch.norm(self.ball_pos[env_ids] - goal_right, dim=1) # same for right goal
    
    def get_state(self):
        # here first we have [ball_x, ball_y] and then [p1_x, p1_y] and [p2_x, p2_y], so after subtracting we get [ball_x - p1_x, ball_y - p1_y]
        # then we compute the distance by getting the length of this vector with: sqrt[(ball_x - p1_x) ^ 2 + (ball_y - p1_y)^2] done by torch.norm
        # then we divide by the size to make sure this stays between [0, 1] with the shape being (num_envs,) for every environment
        dist_p1_ball = torch.norm(self.ball_pos - self.player_1_pos, dim=1, keepdim=True) / self.SIZE 
        dist_p2_ball = torch.norm(self.ball_pos - self.player_2_pos, dim=1, keepdim=True) / self.SIZE # same for player 2
        
        state = torch.cat([
            # position of player 1, with the width normalized by the field width and same with height to ensure values between [0, 1]
            self.player_1_pos / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device), # shape: (num_envs, 2)

            # similar with velocity but divide both x and y velocities by max speed
            self.player_1_vel / self.PLAYER_MAX_SPEED, # shape: (num_envs, 2)

            # now same for player 2
            self.player_2_pos / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device), 
            self.player_2_vel / self.PLAYER_MAX_SPEED,

            # now same for the ball
            self.ball_pos / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device),
            self.ball_vel / self.BALL_MAX_SPEED,

            # the vectors from the players to the ball, normalized by field dimensions to ensure values between [0, 1]
            (self.ball_pos - self.player_1_pos) / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device), # shape: (num_envs, 2)
            (self.ball_pos - self.player_2_pos) / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device), # shape: (num_envs, 2)

            # now the vectors from player 1 to player 2
            (self.player_2_pos - self.player_1_pos) / torch.tensor([self.WIDTH, self.HEIGHT], device=self.device),

            # ball x position relative to left goal, normalized
            # we use slicing here to ensure the shape is (num_envs, 1) instead of (num_envs,)
            (self.ball_pos[:, 0:1] - self.GOAL_WIDTH) / self.WIDTH, # shape: (num_envs, 1)

            # ball x position relative to right goal, normalized
            (self.WIDTH - self.ball_pos[:, 0:1] - self.GOAL_WIDTH) / self.WIDTH, # shape: (num_envs, 1)

            # seems redundant but it works so ill keep it, ball y position centered and normalized between [-0.5, 0.5]
            self.ball_pos[:, 1:2] / self.HEIGHT - 0.5, # shape: (num_envs, 1)

            (dist_p1_ball > dist_p2_ball).float() # shape: (num_envs, 1) a boolean indicating if player 1 is farther from the ball than player 2
        ], dim=1)

        # so concatenating all these together along dim=1 gives us a final shape of (num_envs, 22) so our state dimension is 22
        return state
    
    def step(self, actions):
        p1_actions, p2_actions = actions # unpack the actions for both players
        # shape of each: (num_envs,) and each entry is an integer representing the action index, between [0, self.action_dim - 1]

        # picks the acceleration vectors for player 1 based on the action indices for every environment
        p1_accelerations = self.action_vectors[p1_actions] # shape: (num_envs, 2)
        p2_accelerations = self.action_vectors[p2_actions] # shape: (num_envs, 2)

        # update player physics based on the selected accelerations, along with positions, velocities, max speed, and friction
        self.update_physics(self.player_1_pos, self.player_1_vel, p1_accelerations, self.PLAYER_MAX_SPEED, self.PLAYER_FRICTION)
        self.update_physics(self.player_2_pos, self.player_2_vel, p2_accelerations, self.PLAYER_MAX_SPEED, self.PLAYER_FRICTION)

        # clamp player positions to ensure they stay within field boundaries based on their velocities and sizes
        self.clamp_pos(self.player_1_pos, self.player_1_vel, self.PLAYER_SIZE)
        self.clamp_pos(self.player_2_pos, self.player_2_vel, self.PLAYER_SIZE)

        # set the collision flags to false before checking for collisions
        # essentially resets collisions every step before recomputing
        # we use [:] instead of = to modify in place, as this avoids changing the reference, for better performance
        self.collision_p1[:] = False
        self.collision_p2[:] = False

        for _ in range(2):
            # resolve collisions between players and ball using the positions, velocities, masses, sizes, and restitution coefficients
            c1 = self.resolve_collisions(
                self.player_1_pos, self.player_1_vel, self.PLAYER_MASS, self.PLAYER_SIZE,
                self.ball_pos, self.ball_vel, self.BALL_MASS, self.BALL_SIZE, self.PLAYER_RESTITUTION
            ) # shape: (num_envs,)
            c2 = self.resolve_collisions(
                self.player_2_pos, self.player_2_vel, self.PLAYER_MASS, self.PLAYER_SIZE,
                self.ball_pos, self.ball_vel, self.BALL_MASS, self.BALL_SIZE, self.PLAYER_RESTITUTION
            ) # shape: (num_envs,)

            # update collision flags, set to true if there was a collision detected this iteration or in previous ones
            # if detected in atleast one iteration, it remains true
            self.collision_p1 = self.collision_p1 | c1
            self.collision_p2 = self.collision_p2 | c2

        self.ball_pos += self.ball_vel # update ball positions based on velocities: x_new = x_old + v * dt (dt=1 here, represents time step)

        self.handle_wall_collisions() # handle collisions between ball and field walls
        # apply corner repulsion forces to ball to prevent it from getting stuck in corners
        # makes the game more interesting and dynamic
        self.apply_corner_repulsion() 

        # compute ball speed by using speed = sqrt(vel_x^2 + vel_y^2), done by torch.norm
        ball_speed = torch.norm(self.ball_vel, dim=1, keepdim=True) # shape: (num_envs, 1)

        # check which environments have ball speed exceeding max speed
        over_limit = ball_speed > self.BALL_MAX_SPEED # shape: (num_envs, 1) boolean tensor

        # scale down velocities that exceed max speed to ensure they stay within limits
        # this doesn't change the direction of the velocity, just its magnitude to the max speed
        # torch.where makes sure that this only applies to environments where the speed is over the limit
        self.ball_vel = torch.where(over_limit, (self.ball_vel / ball_speed) * self.BALL_MAX_SPEED, self.ball_vel)
        self.ball_vel *= self.BALL_FRICTION # apply friction to ball velocities to simulate slowing down over time

        goal_codes = self.check_goal() # shape: (num_envs,), 0 for no goal, 1 for player 1 goal, 2 for player 2 goal
        reset_mask_env_ids = goal_codes > 0 # environments that need to be reset (goal scored), shape: (num_envs,)
        
        # compute rewards based on goal codes and other factors
        # first dimension is for every environment, second dimension is for both players
        rewards = self.compute_rewards(goal_codes) # shape: (num_envs, 2)

        if reset_mask_env_ids.any():
            self.reset_positions(reset_mask_env_ids) # reset positions for environments where a goal was scored

        current_ball_speed = ball_speed.squeeze(1) # shape: (num_envs,)
        is_stagnant = current_ball_speed < self.STAGNATION_VELOCITY # shape: (num_envs,) boolean tensor indicating stagnant environments

        self.stagnation_steps[is_stagnant] += 1 # increment stagnation
        self.stagnation_steps[~is_stagnant] = 0 # reset stagnation count for non-stagnant environments

        stuck_mask_env_ids = self.stagnation_steps > self.STAGNATION_THRESHOLD # environments that are stuck, shape: (num_envs,) boolean tensor
        if stuck_mask_env_ids.any() and self.stagnation_on:
            self.reset_positions(stuck_mask_env_ids) # reset positions for stuck environments

        self.current_step += 1 # increment current step count for all environments

        player_1_winner = self.score_p1 >= 10 # shape: (num_envs,) boolean tensor indicating which envs player 1 has won in
        player_2_winner = self.score_p2 >= 10 # same for player 2

        # boolean indicating if max steps reached, shape: (num_envs,) 
        # same for all environments as the environments run in parallel, step together
        terminal_state = self.current_step >= self.max_steps 

        # checks the 3 ways that an environment can be done, either player 1 wins, player 2 wins, or max steps reached
        dones = player_1_winner | player_2_winner | terminal_state # shape: (num_envs,)

        if self.render_mode:
            self.render() # render the game if in render mode
        
        # info that contains the scores of both players for all environments
        info = {
            "player_1_score": self.score_p1.clone(),
            "player_2_score": self.score_p2.clone()
        } 

        # return the state, rewards, done flags, and info dictionary (similar to OpenAI Gym API)
        return self.get_state(), rewards, dones, info
    
    def update_physics(self, pos, vel, acc, max_speed, friction):
        vel += acc # update velocities based on accelerations: v_new = v_old + a * dt (dt=1 here, represents time step)
        speed = torch.norm(vel, dim=1, keepdim=True) # compute speeds

        over_limit = speed > max_speed # boolean tensor indicating which environments exceed max speed
        vel[:] = torch.where(over_limit, (vel / speed) * max_speed, vel) # clamp velocities to max speed, similar to ball speed clamping

        # boolean tensor indicating which environments have no x acceleration for given player
        no_acc_x = (acc[:, 0:1] == 0) # shape: (num_envs, 1)
        no_acc_y = (acc[:, 1:2] == 0) # same for y acceleration

        # apply friction only to x velocity if no x acceleration is applied
        vel[:, 0:1] = torch.where(no_acc_x, vel[:, 0:1] * friction, vel[:, 0:1])
        vel[:, 1:2] = torch.where(no_acc_y, vel[:, 1:2] * friction, vel[:, 1:2]) # same for y velocity

        pos += vel # update positions based on velocities: x_new = x_old + v * dt (dt=1 here, represents time step)

    def clamp_pos(self, pos, vel, radius):
        # ensure positions stay within field boundaries
        # checks if the center position minus radius is less than 0 (left/top wall)
        # we do radius checks to ensure the entire player/ball stays within boundaries, not just the center point
        at_left = pos[:, 0] < radius # boolean tensor for left wall, shape: (num_envs,)
        at_right = pos[:, 0] > self.WIDTH - radius # same for right wall
        
        # move x positions back within boundaries if they exceed limits, clamping them between the boundaries
        pos[:, 0] = torch.clamp(pos[:, 0], radius, self.WIDTH - radius)
        vel[at_left | at_right, 0] = 0 # set x velocity to 0 if at left or right wall, so it doesn't keep trying to go out of bounds

        # same logic for y positions and velocities, but use 
        at_top = pos[:, 1] < radius
        at_bottom = pos[:, 1] > self.HEIGHT - radius
        pos[:, 1] = torch.clamp(pos[:, 1], radius, self.HEIGHT - radius)
        vel[at_top | at_bottom, 1] = 0

    def resolve_collisions(self, pos1, vel1, mass1, size1, pos2, vel2, mass2, size2, restitution):
        # vector from object 1 to object 2
        delta_pos = pos2 - pos1 # shape: (num_envs, 2)

        # compute distances between object centers
        distances = torch.norm(delta_pos, dim=1, keepdim=True) # shape: (num_envs, 1)

        # compute collision mask where distance is less than sum of sizes (indicating overlap)
        collision_mask = (distances < (size1 + size2)).squeeze(1) # shape: (num_envs,) boolean tensor

        if not collision_mask.any():
            return collision_mask # no collisions detected, return early
        
        dist_safe = torch.clamp(distances, min=1e-6) # prevent division by zero

        # unit collision normal vectors, normalized vector from object 1 to object 2
        # here we use vector normalization to get direction only
        # v = [x, y] -> ||v|| = sqrt(x^2 + y^2) --> unit vector = v / ||v||
        collision_normal = delta_pos / dist_safe # shape: (num_envs, 2)
        
        overlap = (size1 + size2) - distances # shape: (num_envs,) overlap distances where collisions occur

        mass1_ratio = mass2 / (mass1 + mass2) # shape: (num_envs,) mass ratio for object 1
        mass2_ratio = mass1 / (mass1 + mass2) # same for object 2

        # for the environments where collisions occur, adjust positions to resolve overlap
        # move each object away from the collision point proportionally to their mass
        # move the object along the collision normal vector by the overlap distance scaled by mass ratio
        # the first object moves opposite to the normal, the second moves along the normal
        # this essentially means that the overlap is now 0, and the objects are just touching
        pos1[collision_mask] -= collision_normal[collision_mask] * overlap[collision_mask] * mass1_ratio
        pos2[collision_mask] += collision_normal[collision_mask] * overlap[collision_mask] * mass2_ratio # same for object 2
        
        # compute relative velocities between the two objects
        # basically the velocity of object 2 from the point of view of object 1 
        rel_vel = vel2 - vel1 # shape: (num_envs, 2)

        # compute velocity along the collision normal
        # dot product of the relative velocity and the collision normal 
        # if positive, objects are moving apart, if negative, they are moving towards each other
        vel_along_normal = (rel_vel * collision_normal).sum(dim=1, keepdim=True)

        # determine which collisions need impulse applied (objects moving towards each other and colliding)
        apply_impulse = collision_mask & (vel_along_normal.squeeze(1) < 0)

        if not apply_impulse.any():
            return collision_mask # no impulses to apply, return early
        
        # compute impulse scalar
        # restitution affects how bouncy the collision is (0: inelastic or sticky, 1: perfectly elastic or bouncy)
        impulse_magnitude = -(1 + restitution) * vel_along_normal[apply_impulse]

        # adjust impulse magnitude based on masses
        impulse_magnitude /= (1 / mass1 + 1 / mass2)

        # the impulse of the collision, applied along the collision normal to move the objects apart
        impulse = impulse_magnitude * collision_normal[apply_impulse]

        # move each object away from the collision point by applying the impulse scaled by their mass
        # so now instead of just adjusting positions to resolve overlap
        # we are also adjusting velocities to simulate realistic collision response by using conservation of momentum
        vel1[apply_impulse] -= impulse / mass1
        vel2[apply_impulse] += impulse / mass2

        if mass2 == self.BALL_MASS:
            # if the second object is the ball, add some random noise to its velocity after collision
            # this adds unpredictability to ball movement after collisions, making the game more dynamic
            noise = torch.randn_like(vel2[apply_impulse]) * self.RANDOM_IMPULSE_NOISE_BALL
            vel2[apply_impulse] += noise

        return collision_mask # return the collision mask indicating where collisions occurred
    
    def handle_wall_collisions(self):
        top_goal = (self.HEIGHT - self.GOAL_HEIGHT) / 2 # y position of top of goal
        bottom_goal = top_goal + self.GOAL_HEIGHT # y position of bottom of goal

        # check if ball is within goal y boundaries
        in_goal_y = (self.ball_pos[:, 1] > top_goal) & (self.ball_pos[:, 1] < bottom_goal) # shape: (num_envs,)

        # check if ball hit the left side considering the ball radius
        hit_left_side = self.ball_pos[:, 0] - self.BALL_SIZE < self.GOAL_WIDTH # shape: (num_envs,)

        # check if ball hit the right side, outside the goal area
        hit_left = hit_left_side & ~in_goal_y # shape: (num_envs,)

        self.ball_pos[hit_left, 0] = self.GOAL_WIDTH + self.BALL_SIZE # reposition the x position of the ball to be just inside the left wall
        self.ball_vel[hit_left, 0] = torch.abs(self.ball_vel[hit_left, 0]) * self.WALL_BOUNCE # reverse x velocity and apply bounce factor to move ball away from wall

        # same for right
        hit_right_side = self.ball_pos[:, 0] + self.BALL_SIZE > self.WIDTH - self.GOAL_WIDTH
        hit_right = hit_right_side & ~in_goal_y 
        self.ball_pos[hit_right, 0] = self.WIDTH - self.GOAL_WIDTH - self.BALL_SIZE
        self.ball_vel[hit_right, 0] = -torch.abs(self.ball_vel[hit_right, 0]) * self.WALL_BOUNCE

        hit_top = self.ball_pos[:, 1] - self.BALL_SIZE < 0 # check if ball hit top wall
        self.ball_pos[hit_top, 1] = self.BALL_SIZE # reposition y position to be just inside top wall
        self.ball_vel[hit_top, 1] = torch.abs(self.ball_vel[hit_top, 1]) * self.WALL_BOUNCE # reverse y velocity and apply bounce factor to move ball away from wall

        # same for bottom wall
        hit_bottom = self.ball_pos[:, 1] + self.BALL_SIZE > self.HEIGHT
        self.ball_pos[hit_bottom, 1] = self.HEIGHT - self.BALL_SIZE
        self.ball_vel[hit_bottom, 1] = -torch.abs(self.ball_vel[hit_bottom, 1]) * self.WALL_BOUNCE

    def apply_corner_repulsion(self):
        # define the four corners of the field
        corners = torch.tensor([
            [0, 0], # top-left corner
            [self.WIDTH, 0], # top-right corner
            [0, self.HEIGHT], # bottom-left corner
            [self.WIDTH, self.HEIGHT] # bottom-right corner
        ], device=self.device)

        # compute distances from ball to each corner, the first dimension is for each environment, second for each corner, third for x,y
        diff = self.ball_pos.unsqueeze(1) - corners.unsqueeze(0) # shape: (num_envs, 4, 2)
        dist = torch.norm(diff, dim=2) # shape: (num_envs, 4), distances to each corner

        # check which environments are within corner radius for each corner
        in_corner = dist < self.CORNER_RADIUS # shape: (num_envs, 4) boolean tensor
        if in_corner.any():
            safe_dist = torch.clamp(dist, min=1e-5).unsqueeze(2) # prevent division by zero with clamping, shape: (num_envs, 4, 1)
            direction = diff / safe_dist # unit vectors from corners to ball, shape: (num_envs, 4, 2) by using vector normalization

            # sum the unit direction vectors for the corners the ball is within radius of, used for force direction
            force = (direction * in_corner.unsqueeze(2).float()).sum(dim=1) 

            self.ball_vel += force * self.CORNER_FORCE # apply repulsion force to ball velocity scale by corner force magnitude
    
    def check_goal(self):
        top_goal = (self.HEIGHT - self.GOAL_HEIGHT) / 2 # y position of top of goal
        bottom_goal = top_goal + self.GOAL_HEIGHT # y position of bottom of goal

        # check if ball is within goal y boundaries
        in_goal_y = (self.ball_pos[:, 1] > top_goal) & (self.ball_pos[:, 1] < bottom_goal) # shape: (num_envs,)

        # check if ball is within goal x boundaries on the left side, where player 2 scores
        in_goal_x_left = (self.ball_pos[:, 0] - self.BALL_SIZE <= self.GOAL_WIDTH) # shape: (num_envs,)

        # check if ball is within goal x boundaries on the right side, where player 1 scores
        in_goal_x_right = (self.ball_pos[:, 0] + self.BALL_SIZE >= self.WIDTH - self.GOAL_WIDTH) # shape: (num_envs,)

        # codes for goals: 0 = no goal, 1 = player 1 goal, 2 = player 2 goal for all environments
        goal_codes = torch.zeros(self.num_envs, device=self.device, dtype=torch.int32) # shape: (num_envs,)

        p2_scores = in_goal_x_left & in_goal_y # check which environments player 2 scored in
        goal_codes[p2_scores] = 2 # set goal code to 2 for player 2 goals
        self.score_p2[p2_scores] += 1 # increment player 2 scores
        
        # same for player 1
        p1_scores = in_goal_x_right & in_goal_y 
        goal_codes[p1_scores] = 1
        self.score_p1[p1_scores] += 1

        return goal_codes
    
    def compute_rewards(self, goal_codes):
        # initialize rewards tensor for both players in all environments, the 2 columns represent player 1 and player 2 respectively
        rewards = torch.zeros((self.num_envs, 2), device=self.device) # shape: (num_envs, 2)
        
        p1_scored = (goal_codes == 1) # boolean tensor indicating which environments player 1 scored in, shape: (num_envs,)
        p2_scored = (goal_codes == 2) # same for player 2

        rewards[p1_scored, 0] += self.GOAL_REWARD # give player 1 a reward of +100 for scoring in the given environments
        rewards[p1_scored, 1] -= self.GOAL_REWARD # give player 2 a penalty of -100 for conceding a goal in the given environments

        rewards[p2_scored, 1] += self.GOAL_REWARD # give player 2 a reward of +100 for scoring in the given environments
        rewards[p2_scored, 0] -= self.GOAL_REWARD # give player 1 a penalty of -100 for conceding a goal in the given environments

        goal_left = torch.tensor([self.GOAL_WIDTH, self.HEIGHT / 2], device=self.device) # left goal center position
        goal_right = torch.tensor([self.WIDTH - self.GOAL_WIDTH, self.HEIGHT / 2], device=self.device) # right goal center position

        dist_p1_ball = torch.norm(self.ball_pos - self.player_1_pos, dim=1) # distance between player 1 and ball
        dist_p2_ball = torch.norm(self.ball_pos - self.player_2_pos, dim=1) # distance between player 2 and ball

        dist_ball_left = torch.norm(self.ball_pos - goal_left, dim=1) # distance between ball and left goal
        dist_ball_right = torch.norm(self.ball_pos - goal_right, dim=1) # distance between ball and right goal

        if self.prev_dist_p1 is None:
            self.prev_dist_p1 = dist_p1_ball.clone() # store previous distance between player 1 and ball
            self.prev_dist_p2 = dist_p2_ball.clone() # store previous distance between player 2 and ball
            self.prev_ball_dist_left = dist_ball_left.clone() # store previous distance between ball and left goal
            self.prev_ball_dist_right = dist_ball_right.clone() # store previous distance between ball and right goal

        # reward players for moving the ball towards opponent's goal, normalize to control the max reward magnitude
        rewards[:, 0] += ((self.prev_ball_dist_right - dist_ball_right) / self.BALL_MAX_SPEED) * self.BALL_TOWARDS_GOAL_REWARD 
        rewards[:, 1] += ((self.prev_ball_dist_left - dist_ball_left) / self.BALL_MAX_SPEED) * self.BALL_TOWARDS_GOAL_REWARD 

        # reward players for getting closer to the ball, normalize to control the max reward magnitude
        rewards[:, 0] += ((self.prev_dist_p1 - dist_p1_ball) / self.PLAYER_MAX_SPEED) * self.MOVE_CLOSE_TO_BALL_REWARD
        rewards[:, 1] += ((self.prev_dist_p2 - dist_p2_ball) / self.PLAYER_MAX_SPEED) * self.MOVE_CLOSE_TO_BALL_REWARD
        
        # reward players for colliding with the ball to encourage interaction
        rewards[self.collision_p1, 0] += self.COLLISION_BALL_REWARD
        rewards[self.collision_p2, 1] += self.COLLISION_BALL_REWARD

        # punish players for being far from the ball to encourage engagement, normalize to control the max penalty magnitude
        rewards[:, 0] -= self.DISTANCE_PENALTY * (dist_p1_ball / self.SIZE)
        rewards[:, 1] -= self.DISTANCE_PENALTY * (dist_p2_ball / self.SIZE)

        # boolean tensors indicating which players are moving slowly for all environments
        p1_slow = torch.norm(self.player_1_vel, dim=1) < self.STAGNATION_RATIO * self.PLAYER_MAX_SPEED
        p2_slow = torch.norm(self.player_2_vel, dim=1) < self.STAGNATION_RATIO * self.PLAYER_MAX_SPEED

        # punish players for moving slowly to encourage active play
        rewards[p1_slow, 0] -= self.STAGNANT_PENALTY
        rewards[p2_slow, 1] -= self.STAGNANT_PENALTY

        # update previous distances for next step's reward calculations
        self.prev_dist_p1 = dist_p1_ball.clone()
        self.prev_dist_p2 = dist_p2_ball.clone()
        self.prev_ball_dist_left = dist_ball_left.clone()
        self.prev_ball_dist_right = dist_ball_right.clone()

        return rewards

    def render(self):
        env_idx = 0 # only render the first environment for visualization

        p1 = self.player_1_pos[env_idx].cpu().int().numpy() # get player 1 position as numpy array for rendering, shape: (2,)
        p2 = self.player_2_pos[env_idx].cpu().int().numpy() # same for player 2
        b = self.ball_pos[env_idx].cpu().int().numpy() # same for the ball

        self.screen.fill((70, 140, 70)) # fill field with green color
        top_goal = (self.HEIGHT - self.GOAL_HEIGHT) // 2 # y position of top of goal

        # draw goals and field markings
        pygame.draw.rect(self.screen, (200, 0, 0), (0, top_goal, self.GOAL_WIDTH, self.GOAL_HEIGHT)) # draw left goal
        pygame.draw.rect(self.screen, (0, 0, 200), (self.WIDTH - self.GOAL_WIDTH, top_goal, self.GOAL_WIDTH, self.GOAL_HEIGHT)) # draw right goal
        pygame.draw.line(self.screen, (255, 255, 255), (self.WIDTH // 2, 0), (self.WIDTH // 2, self.HEIGHT), 5) # draw center line
        pygame.draw.circle(self.screen, (255, 255, 255), (self.WIDTH // 2, self.HEIGHT // 2), 70, 5) # draw center circle

        # draw corner circles to visualize corner repulsion areas
        pygame.draw.circle(self.screen, (255, 255, 0), (0, 0), self.CORNER_RADIUS // 2) # top-left corner
        pygame.draw.circle(self.screen, (255, 255, 0), (self.WIDTH, 0), self.CORNER_RADIUS // 2) # top-right corner  
        pygame.draw.circle(self.screen, (255, 255, 0), (0, self.HEIGHT), self.CORNER_RADIUS // 2) # bottom-left corner
        pygame.draw.circle(self.screen, (255, 255, 0), (self.WIDTH, self.HEIGHT), self.CORNER_RADIUS // 2) # bottom-right corner

        pygame.draw.circle(self.screen, (200, 0, 0), p1, self.PLAYER_SIZE) # draw player 1 as red circle
        pygame.draw.circle(self.screen, (0, 0, 200), p2, self.PLAYER_SIZE) # draw player 2 as blue circles
        pygame.draw.circle(self.screen, (0, 0, 0), b, self.BALL_SIZE) # draw ball as black circle

        # get scores for both players to display
        s1 = self.score_p1[env_idx].item() 
        s2 = self.score_p2[env_idx].item()

        # render the score text at the top center of the screen
        s_text = self.font.render(f"{s1}  -  {s2}", True, (255, 255, 255))
        self.screen.blit(s_text, (self.WIDTH // 2 - s_text.get_width() // 2, 20))

        # update the display and control the frame rate
        pygame.display.flip()
        self.clock.tick(60)

        # handle pygame events such as window close
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False