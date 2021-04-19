import pprint
import random
from math import log, sqrt

random.seed(0)

TwoXTwoMDP = {
    # Key (state id): Value (adjacent state via action L,R,U,D)
    'stategraph': {1: [1, 4, 2, 1],  # Connections in order L,R,U,D
                   2: [2, 3, 2, 1],
                   3: None,
                   4: None},
    # Probability of traversing each edge type (L,R,U,D) given each action
    'paction': {'L': [.8, 0, .1, .1],
                'R': [0, .8, .1, .1],
                'U': [.1, .1, .8, 0],
                'D': [.1, .1, 0, .8]},

    'actions': ['L', 'R', 'U', 'D']

}

FourXThreeMDP = {
    # Key (state id): Value (adjacent state via action L,R,U,D)
    'stategraph': {(1, 1): [(1, 1), (2, 1), (1, 2), (1, 1)],  # Connections in order L,R,U,D
                   (1, 2): [(1, 2), (1, 2), (1, 3), (1, 1)],
                   (1, 3): [(1, 3), (2, 3), (1, 3), (1, 2)],
                   (2, 1): [(1, 1), (3, 1), (2, 1), (2, 1)],
                   (2, 2): None,
                   (2, 3): [(1, 3), (3, 3), (2, 3), (2, 3)],
                   (3, 1): [(2, 1), (4, 1), (3, 2), (3, 1)],
                   (3, 2): [(3, 2), (4, 2), (3, 3), (3, 1)],
                   (3, 3): [(2, 3), (4, 3), (3, 3), (3, 2)],
                   (4, 1): [(3, 1), (4, 1), (4, 2), (4, 1)],
                   (4, 2): None,
                   (4, 3): None},

    # Probability of traversing each edge type (L,R,U,D) given each action
    'paction': {'L': [.8, 0, .1, .1],
                'R': [0, .8, .1, .1],
                'U': [.1, .1, .8, 0],
                'D': [.1, .1, 0, .8]},

    'actions': ['L', 'R', 'U', 'D']

}


def next_state(mdp, current_state):
    return mdp.get('stategraph').get(current_state)


def paction_list(mdp, move):
    return mdp.get('paction').get(move)


def get_action_list(mdp):
    return mdp.get('actions')


def get_max_cost(mdp, state_list, u):
    if state_list is None:
        return 0
    cost_list = [0]
    for each_action in get_action_list(mdp):
        my_zip = zip(state_list, paction_list(mdp, each_action))
        cost_list += [sum([u[s] * p for (s, p) in my_zip])]
    return max(cost_list)


def value_iteration(mdp, gamma, r_fn, quiet=True, delta=.0001, n=-1):
    """
    __ Part 1: Implement this __

    Perform Value Iteration:
    
     mdp - A Markov Decision Process represented as a dictionary with keys:
         'stategraph' : a map between states and transition vectors 
                        (next states reachable via a the action with 'index' i)
         'paction'    : a map between intended action and a distribution overal
                         *actual* actions, the actual action vector should have the
                         same length as the transition vectors in the stategraph, and
                         the sum of this vector should be 1.0
         'actions'    : the 'name' of each action in the action vector.

     gamma - a discount rate
     r_fn  - a reward function that takes a state and returns an immediate reward.
     quiet - if True, supress all output in this function
     delta - a stopping criteria: stops when utility changes by <= delta
     n     - a stopping criteria: stops when n iterations have occurred 
              (if n==-1, only the delta test applies)
     
     the stopping criteria for value iteration should have the following semantics:

       if (utility_change <= delta or (n > 0 and iterations == n)) then stop

     returns:
      a map of {state -> utilities}
    """

    states = mdp.get('stategraph').keys()

    # initialize utilities
    u_1 = dict([(s, r_fn(s)) for s in states])
    iterations = 1

    while 1:
        u = u_1.copy()
        utility_change = 0

        for each_state in states:

            # get the final value of -> maxΣP(s’|s,a)U[s’]
            max_cost = get_max_cost(mdp, next_state(mdp, each_state), u)

            # the form of -> U’[s] = R(s) + γmaxΣP(s’|s,a)U[s’]
            u_1[each_state] = r_fn(each_state) + (gamma * max_cost)

            new_utility_change = abs(u_1[each_state] - u[each_state])
            utility_change = max(new_utility_change, utility_change)

        if utility_change <= delta or 0 < n == iterations:  # stopping criteria
            if not quiet:
                print(u_1)
            return u_1

        iterations += 1


def policy_evaluation(policy, r_fn, gamma, u, mdp, iterations):
    states = mdp.get('stategraph').keys()

    for i in range(iterations):
        for s in states:
            next_state_list = next_state(mdp, s)

            if next_state_list is not None:
                total_cost = 0

                my_zip = zip(next_state_list, mdp.get('paction')[policy[s]])
                cost = sum((u[ns] * p) for (ns, p) in my_zip)

                u[s] = r_fn(s) + (gamma * cost)
                total_cost += cost
    return u


def policy_iteration(mdp, gamma, r_fn, policy, quiet=True, viterations=5):
    """
    __ Part 2: Implement this __

    Perform Policy Iteration:
    
     mdp - A Markov Decision Process represented as a dictionary with keys:
         'stategraph' : a map between states and transition vectors 
                        (next states reachable via a the action with 'index' i)
         'paction'    : a map between intended action and a distribution overal
                         *actual* actions, the actual action vector should have the
                         same length as the transition vectors in the stategraph, and
                         the sum of this vector should be 1.0
         'actions'    : the 'name' of each action in the action vector.

     gamma - a discount rate
     r_fn  - a reward function that maps states -> immediate rewards
     quiet - if True, supress all output in this function
     vit   - the number of iterations for the value update step
     n     - the number of iterations for policy update

     the stopping criteria for policy iteration should have the following semantics:

       if (not policy_changed or (n > 0 and iterations == n)) then stop

     returns:
      a map of {state -> actions}
    """

    states = mdp.get('stategraph').keys()

    # initialize utilities
    u = dict([(s, r_fn(s)) for s in states])
    iterations = 1

    while 1:
        u = policy_evaluation(policy, r_fn, gamma, u, mdp, viterations)
        policy_changed = False

        for each_state in states:
            next_state_list = next_state(mdp, each_state)
            if next_state_list is not None:
                # get this value -> maxΣP(s’|s,a)U[s’]
                action_utility = dict((e, 0) for e in get_action_list(mdp))
                action_max_cost = 0
                for action in get_action_list(mdp):
                    my_zip = zip(next_state_list, paction_list(mdp, action))
                    cost = sum((u[ns] * p) for (ns, p) in my_zip)

                    action_utility[action] = cost
                    action_max_cost = max(action_max_cost, cost)

                # get this -> P(s’|s, π[s])U[s’]
                my_policy = zip(next_state_list, mdp.get('paction')[policy[each_state]])
                policy_cost = sum((u[ns] * p) for (ns, p) in my_policy)

                if action_max_cost > policy_cost:
                    policy_changed = True
                    for (key, value) in action_utility.items():
                        if value >= action_max_cost:
                            policy[each_state] = key

        # stopping criteria
        if not policy_changed or (0 < viterations == iterations):
            if not quiet:
                print(policy)
            return policy

        iterations += 1


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('method', choices=['policy', 'value'])
    parser.add_argument('environment', choices=['4x3', '2x2'])
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--gamma', type=float, default=1.0)

    args = parser.parse_args()

    if args.environment == '4x3':
        env = FourXThreeMDP
        gamma = args.gamma
        rfn = lambda s: {(4, 2): -1, (4, 3): 1}.get(s, -0.04)
        pi = {s: 'L' for s in env['stategraph']}
        s0 = (1, 1)  # start state
    elif args.environment == '2x2':
        env = TwoXTwoMDP
        gamma = args.gamma
        # rfn is a reference to the function that performs the indexing
        # look at the documentation (for python's built-in list) for info
        rfn = [None, -0.04, -0.04, 1, -1].__getitem__
        pi = {1: 'R', 2: 'D'}
        s0 = 1  # start state

    if args.method == 'policy':
        policy_iteration(env, gamma, rfn, pi, quiet=args.quiet)
    elif args.method == 'value':
        value_iteration(env, gamma, rfn, quiet=args.quiet)
