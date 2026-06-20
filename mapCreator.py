
import numpy as np

# Cell type constants
EMPTY   = 0
TOXIC   = 1
HEALTHY = 2
START   = 3
GOAL    = 4
 

def create_map(  toxic_pct, healthy_pct, seed=None):
    if seed is not None:
        np.random.seed(seed)


    """     grid_height, grid_width = np.random.randint(5, 10), np.random.randint(7, 12) """
    grid_height, grid_width = 10, 12
   
    grid = np.zeros((grid_height, grid_width), dtype=int)

    # Define start and goal positions on the fly 
    start_pos = (grid_height - 1, 0)  
    goal_pos = (grid_height - 1, grid_width - 1)
 
    # Test for a bug 
    start_idx = start_pos[0] * grid_width + start_pos[1]
    goal_idx = goal_pos[0] * grid_width + goal_pos[1]
    reserved = {start_idx, goal_idx}
    all_idx = [i for i in range(grid_height * grid_width) if i not in reserved]

    # Calculate the number of toxic and healthy cells based on the specified percentages
    # The percentage of toxic cells is not possibly to exactly match the number of cells as in healthy cells, 
    # as we make a random choise of the cells, so we can have some overlap between toxic and healthy cells
    # But this is good enough for us, lets say that this way we create a toxic swamp to traverse in the fog ;)
    num_toxic_cells = int(grid_height * grid_width * toxic_pct)
    
    toxic_cells = np.random.choice(all_idx, size=num_toxic_cells, replace=False)
    toxic_cells = [(cell // grid_width, cell % grid_width) for cell in toxic_cells]

    #Healthy cells 
    num_healthy_cells = int(grid_height * grid_width * healthy_pct)
    healthy_cells = np.random.choice(all_idx, size=num_healthy_cells, replace=False)
    healthy_cells = [(cell // grid_width, cell % grid_width) for cell in healthy_cells]

    for cell in toxic_cells:
        grid[cell] = TOXIC

    for cell in healthy_cells:
        grid[cell] = HEALTHY

# Mark the start and goal positions
    #start
    grid[start_pos] = START
    #goal
    grid[goal_pos] = GOAL

    return grid, toxic_cells, healthy_cells, start_pos, goal_pos