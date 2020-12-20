from wumpus import Hunter
from .linear_space import SmartCoordinate, SmartVector

# utility funcs for the logger
def get_pretty_log_representation_of(*, locations):
    """
    make a concise and sorted string representation of a list of SmartCoordinate
    """
    return "[" + ", ".join([f"(x: {location.x}, y: {location.y})" for location in sorted(locations)]) + "]\n"

def get_final_location(*, from_location, with_orientation, with_plan_actions):
    """
    execute a plan simulation to calculate the final of the agent if it was to execute the actions
    """
    location = from_location
    orientation = with_orientation
    plan_actions = with_plan_actions

    agent_location = location
    agent_orientation = orientation

    for action in plan_actions:
        if action == Hunter.Actions.MOVE:
            agent_location = agent_location + agent_orientation
        elif action == Hunter.Actions.LEFT:
            agent_orientation = -agent_orientation.get_perpendicular_vector_clockwise()
        elif action == Hunter.Actions.RIGHT:
            agent_orientation = agent_orientation.get_perpendicular_vector_clockwise()

    return (agent_location, len(plan_actions) + 9 if Hunter.Actions.SHOOT in plan_actions 
            else len(plan_actions))

def manhattan_distance_between(start, destination):
    """
    the traditional manhattan distance
    """
    return abs(destination.x - start.x) + abs(destination.y - start.y)

# suppose start being oriented in destination direction (either one of them)
def smart_manhattan_distance(start, *, destination, safe_locations):
    """
    checks if there is a solution to reach the destination in exactly manhattan distance steps:
    - if there is one, it will return manhattan distance + orientation overhead of the best 
      available path (with value 0-3)
    - otherwise it will return manhattan_distance + 4
    more info about these values is available in the documentation (they are not random values)
    """
    map_size = (abs(destination.x - start.x) + 1, abs(destination.y - start.y) + 1)

    #early exit if start and destination in the same point
    if start == destination:
        return 0

    #filtering out all outer blocks from the smallest grid containing the start and destination location
    block_max_y = max(start.y, destination.y)
    block_max_x = max(start.x, destination.x)
    block_min_y = min(start.y, destination.y)
    block_min_x = min(start.x, destination.x)

    block_locations = []

    for x in range(block_min_x, block_max_x + 1):
        for y in range(block_min_y, block_max_y + 1):
            if (current_location := SmartCoordinate(x, y)) not in safe_locations:
                block_locations.append(current_location)

    filtered_blocks = list(filter(lambda block: (block_min_x <= block.x <= block_max_x) \
                                                and (block_min_y <= block.y <= block_max_y), block_locations))

    ax_orientation = SmartVector(start.x - destination.x, start.y - destination.y)

    # we are in the \ situation and we have to consider top_left_location and bottom_right_location
    if ax_orientation == SmartVector(1, -1) or ax_orientation == SmartVector(-1, 1):
        top_left_location = SmartCoordinate()
        bottom_right_location = SmartCoordinate()

        if start.y > destination.y:
            top_left_location = start
            bottom_right_location = destination
        else:
            top_left_location = destination
            bottom_right_location = start

        translation_vector = -bottom_right_location
        start = start + translation_vector
        destination = destination + translation_vector
        filtered_blocks = list(map(lambda block: block + translation_vector, filtered_blocks))
        filtered_blocks = list(map(lambda block: SmartCoordinate(-block.x, block.y), filtered_blocks))

    #we are in the / situation on the positive quadrant and can consider top_right_location and bottom_left_location
    else:
        top_right_location = SmartCoordinate()
        bottom_left_location = SmartCoordinate()

        if start.y > destination.y:
            top_right_location = start
            bottom_left_location = destination
        else:
            top_right_location = destination
            bottom_left_location = start

        translation_vector = -bottom_left_location
        start = start + translation_vector
        destination = destination + translation_vector
        filtered_blocks = list(map(lambda block: block + translation_vector, filtered_blocks))
        

    # mapping the grid world into a binary matrix to performing some calculations 
    # (read documentation for better explanation )
    grid_map = [[0 for i in range(map_size[0])] for j in range(map_size[1])]    

    for block_location in filtered_blocks:
        grid_map[block_location.y][block_location.x] = 1

    base_manhattan_distance = manhattan_distance_between(start, destination)

    result_indexes = []

    previous_bitmap_row = grid_map[0]
    previous_mapping_indexes_right = []

    for index, value in enumerate(previous_bitmap_row):
        if value == 0:
            previous_mapping_indexes_right.append(index)
        else:
            break

    result_indexes.append(previous_mapping_indexes_right)

    for bitmap_row_index, bitmap_row in enumerate(grid_map[1:]):
        legal_row = [value if index in previous_mapping_indexes_right 
                     else 1 
                     for index, value in enumerate(bitmap_row)] 
        
        indexes = [index for index, value in enumerate(legal_row) 
                   if value == 0 and previous_bitmap_row[index] == 0]

        mapping_right = set()

        for index in indexes:
            for row_index, value in enumerate(bitmap_row[index:]):
                if value == 0:
                    mapping_right.add(row_index + index)
                else:
                    break

        if not mapping_right:
            # no solution with manhattan distance
            return base_manhattan_distance + 1 + 3
            # 1 because there is no way to reach destination with a straight manhattan path, 
            #   so also manhattan needs at least one rotation
            # 2 because it need to one step farther and one step to recover that step distance 
            # 1 for minimum orientation overhead (manhattan from farther location)
        
        result_indexes.append(sorted(list(mapping_right)))
        previous_bitmap_row = bitmap_row
        previous_mapping_indexes_right = mapping_right
    

    if not (map_size[0] - 1 in result_indexes[-1]):
        # no solution with manhattan distance
        return base_manhattan_distance + 1 + 3
        # 1 because there is no way to reach destination with a straight manhattan path, 
        #   so also manhattan needs at least one rotation
        # 2 because it need to one step farther and one step to recover that step distance 
        # 1 for minimum orientation overhead (manhattan from farther location)

    else:
        # there is a straight path to the goal with no blocks
        if len(result_indexes) == 1 or all([len(mapping_right) == 1 for mapping_right in result_indexes]):
            return base_manhattan_distance

        # straight right then up is available
        elif (len(result_indexes[0]) == map_size[0]
             and all([map_size[0] - 1 in mapping_right for mapping_right in result_indexes])):
            return base_manhattan_distance + 1 # only one orientation needed  EEEEE {LEFT} NNNNN
        # straight up then right is available
        elif (len(result_indexes[-1]) == map_size[0]
             and all([0 in mapping_right for mapping_right in result_indexes])):
            return base_manhattan_distance + 1 # only one orientation needed  NNNNN {RIGHT} EEEEE
        
        # one direction straight, the other can be split up in two
        elif (list(range(map_size[0])) in result_indexes 
             or any([False not in boolean_list for boolean_list in [[i in mapping_right for mapping_right in result_indexes] for i in range(map_size[0])]])):
            return base_manhattan_distance + 2 # maybe only two orientation needed  EE {LEFT} NNNNN {RIGHT} EEE 
                                               # at least an entire segment must be followed straight

        else:
            return base_manhattan_distance + 3 # all other cases
