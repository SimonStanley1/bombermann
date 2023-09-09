def state_to_features(self, game_state: dict) -> np.array:
    """
    *This is not a required function, but an idea to structure your code.*

    Converts the game state to the input of your model, i.e.
    a feature vector.

    You can find out about the state of the game environment via game_state,
    which is a dictionary. Consult 'get_state_for_agent' in environment.py to see
    what it contains.

    :param game_state:  A dictionary describing the current game board.
    :return: np.array
    """
    # This is the dict before the game begins and after it ends
    #if game_state is None:
    #    return None
    #print(game_state['self'])
    # For example, you could construct several channels of equal shape, ...
    #channels = []
    #channels.append(...)
    # concatenate them as a feature tensor (they must have the same shape), ...
    #stacked_channels = np.stack(channels)
    # and return them as a vector

    if game_state is None:
        return None
    
    feat = np.array([])
    # agents position 
    pos_agent = np.array(game_state['self'][3])

    moves_to_coin = np.zeros(4)

    if game_state['coins']:
        #find nearest coin
        dist_to_coins = np.sum((np.array(game_state['coins'])-pos_agent)**2,axis=-1)
        nearest_coin_index = np.argmin(dist_to_coins)
        nearest_coin = game_state['coins'][nearest_coin_index]

        m = -1
        for move in pos_agent + MOVES:
            m += 1
            if game_state['field'][move[0],move[1]] == 0:
                    dist_after_move = np.sqrt(np.sum((move-nearest_coin)**2))
                    if dist_after_move == 0:
                        inv_dist_after_move = 2
                    else:
                        inv_dist_after_move = 1/dist_after_move
                    moves_to_coin[m] = inv_dist_after_move
    feat = np.append(feat, moves_to_coin)

    #number of destroyed crates, if the agent would drop a bomb
    destroyed_crates = 0
    for move in MOVES:
        field = pos_agent
        for i in range(3):
            field = field + move
            value_on_field = game_state['field'][field[0],field[1]]
            if value_on_field == -1:
                break
            if value_on_field:
                destroyed_crates += 1
    #print(destroyed_crates)
    feat = np.append(feat, [destroyed_crates])
    #bomb feature
    moves_from_bomb = np.zeros(8)
    #find nearest bomb
    if game_state['bombs']:
        nearest_bomb = np.array(game_state['bombs'][0][0])
        dist_to_bomb = np.sum((nearest_bomb - pos_agent)**2,axis=-1)
        for bomb in game_state['bombs']:
            pos_bomb = np.array(bomb[0])
            if dist_to_bomb > np.sum((pos_bomb - pos_agent)**2,axis=-1):
                nearest_bomb = pos_bomb
                dist_to_bomb = np.sum((pos_bomb - pos_agent)**2,axis=-1)
        m = -2
        for move in pos_agent + MOVES:
            m += 2
            #print(game_state['field'][move[0],move[1]], move)
            if game_state['field'][move[0],move[1]] == 0:
                    distX_after_move = abs(move[0]-nearest_bomb[0])
                    if distX_after_move == 0:
                        inv_distX_after_move = 2
                    else:
                        inv_distX_after_move = 1/distX_after_move
                    moves_from_bomb[m] = inv_distX_after_move
                    #print(m, inv_distX_after_move)

                    distY_after_move = abs(move[1]-nearest_bomb[1])
                    if distY_after_move == 0:
                        inv_distY_after_move = 2
                    else:
                        inv_distY_after_move = 1/distY_after_move
                    moves_from_bomb[m+1] = inv_distY_after_move

                    #print(m+1, inv_distY_after_move)

    feat = np.append(feat, moves_from_bomb)

    #number of bombs
    number_bombs = len(game_state['bombs'])
    feat = np.append(feat, [number_bombs])
    #print(game_state['explosion_map'])
    #print(feat)



    
    features = torch.from_numpy(feat).float()
    #print(features)
    return features
