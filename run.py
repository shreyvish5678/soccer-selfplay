import torch
import pygame
import sys
from game import SoccerGameEnv
from ppo import ActorCriticNetwork

PLAYER_1_TYPE = "AI" # left side control, can be "HUMAN" or "AI"
PLAYER_2_TYPE = "AI" # right side control, can be "HUMAN" or "AI"  
PLAYER_1_MODEL_PATH = "soccer_final.pth" # the model path for player 1 if AI
PLAYER_2_MODEL_PATH = "soccer_final.pth" # the model path for player 2 if AI
MAX_STEPS = 5000 # maximum steps of the game

def get_action_from_key_state(keys, is_p1=True):
    # keymaps for each player
    keymap_p1 = { "up": pygame.K_w, "down": pygame.K_s, "left": pygame.K_a, "right": pygame.K_d }
    keymap_p2 = { "up": pygame.K_UP, "down": pygame.K_DOWN, "left": pygame.K_LEFT, "right": pygame.K_RIGHT }
    keymap = keymap_p1 if is_p1 else keymap_p2
    
    x = keys[keymap["right"]] - keys[keymap["left"]] # horizontal direction (-1 left, +1 right so right - left)
    y = keys[keymap["down"]] - keys[keymap["up"]] # vertical direction (-1 up, +1 down so down - up)

    # direction to action lookup
    dir_to_action = {
        (0, -1): 1,  # up
        (0,  1): 2,  # down
        (-1, 0): 3,  # left
        (1,  0): 4,  # right
        (-1, -1): 5, # up-left
        (1, -1): 6,  # up-right
        (-1, 1): 7,  # down-left
        (1, 1): 8,   # down-right
    }
    return dir_to_action.get((x, y), 0) # default to 0 (no-op) if no direction

def load_agent(filepath, env, device):
    state_dim = env.get_state().shape[1] + 1 # +1 for player side indicator
    action_dim = env.action_dim # number of discrete actions
    agent = ActorCriticNetwork(state_dim, action_dim).to(device) # initialize agent
    
    try:
        # load the model checkpoint safely
        checkpoint = torch.load(filepath, map_location=device)            
        if 'agent_state_dict' in checkpoint:
            agent.load_state_dict(checkpoint['agent_state_dict'])
        else:
            agent.load_state_dict(checkpoint)
    except Exception as e:
        print(f"Failed to load model: {e}")
        sys.exit(1)
        
    agent.eval() # set agent to evaluation mode
    return agent

def get_ai_action(agent, state, player_side_idx, device):
    indicator = torch.full((state.shape[0], 1), player_side_idx, device=device, dtype=torch.float32) # player side indicator
    # select action using the agent by inputting state + indicator
    if state.dim() == 1:
        state = state.unsqueeze(0)
    augmented_state = torch.cat([state, indicator], dim=1)
    with torch.no_grad():
        action, _, _ = agent.select_action(augmented_state)
    return action

def main():
    # set the device
    device = 'cpu' 
    if torch.cuda.is_available(): 
        device = 'cuda'
    elif torch.backends.mps.is_available():
        device = 'mps'
    
    # initialize the environment
    print(f"\nInitializing Game on {device}...")
    env = SoccerGameEnv(num_envs=1, render=True, max_steps=MAX_STEPS, device=device, stagnation_on=False)
    
    # initialize the agents based on parameters
    p1_agent = None
    p2_agent = None
    
    if PLAYER_1_TYPE == "AI":
        p1_agent = load_agent(PLAYER_1_MODEL_PATH, env, device)
        print(f"Player 1 (Red): AI from {PLAYER_1_MODEL_PATH}")
    else:
        print("Player 1 (Red): HUMAN - WASD keys")
    
    if PLAYER_2_TYPE == "AI":
        p2_agent = load_agent(PLAYER_2_MODEL_PATH, env, device)
        print(f"Player 2 (Blue): AI from {PLAYER_2_MODEL_PATH}")
    else:
        print("Player 2 (Blue): HUMAN - Arrow keys")
    
    print("Press ESC to quit.\n")

    # run the game loop
    state = env.reset()
    running = True
    
    while running:
        # set up quitting events in pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

        keys = pygame.key.get_pressed() # get current key states
        
        # if human, get action from key states; if AI, get action from agent
        if PLAYER_1_TYPE == "HUMAN":
            a1 = get_action_from_key_state(keys, is_p1=True)
            action_p1 = torch.tensor([a1], device=device)
        else:
            action_p1 = get_ai_action(p1_agent, state, 0.0, device)
        
        # same for player 2
        if PLAYER_2_TYPE == "HUMAN":
            a2 = get_action_from_key_state(keys, is_p1=False)
            action_p2 = torch.tensor([a2], device=device)
        else:
            action_p2 = get_ai_action(p2_agent, state, 1.0, device)

        next_state, _, dones, _ = env.step((action_p1, action_p2)) # take a step in the environment with both actions
        
        # check for done signals to reset the environment
        if dones.any():
            print(f"Game Over! Score: Red {env.score_p1.item()} - Blue {env.score_p2.item()}")
            state = env.reset()
        else:
            state = next_state
    
    # clean up and quit pygame
    pygame.quit()

if __name__ == "__main__":
    main()