import numpy as np
import random

def metropolis_hastings(initial_state, proposal_function, log_density, iters=100, print_every=10, 
                        tolerance=0.02, error_function=None, pretty_state=None):
    """
    Runs a metropolis hastings algorithm given the settings
    
    Arguments:
    
    initial_state: state from where we should start moving
    
    proposal_function: proposal function for next state, it takes the current state
                       and returns the next state
                       
    log_density: log probability(upto an unknown normalization constant) function, takes a 
                 state as input, and gives the log(probability*some constant) of the state.
    
    iters: number of iters to continue
    
    print_every: print every $ iterations the current statistics. For diagnostics purposes.
    
    tolerance: if acceptance rate drops below this, we stop the simulation
    
    error_function: computes the error for current state. Printed every print_every iterations.
                    Just for your diagnostics.
    
    pretty_state: A function from your side to print the current state in a pretty format.
    
    Returns:
    
    states: List of states generated during simulation
    
    cross_entropies: list of negative log probabilites during the simulation.
    
    errors: lists of errors generated if given error_function, none otherwise.
    
    """
    
    p1 = log_density(initial_state)
    errors = []
    cross_entropies = []
    
    state = initial_state
    cnt = 0
    accept_cnt = 0
    error = -1
    states = [initial_state]
    it = 0
    
    while it < iters:
        #propose a move
        new_state = proposal_function(state)
        p2 = log_density(new_state)
        
        u = random.random()
        cnt += 1
        
        #accept the new move with probability p2-p1
        if p2-p1 > np.log(u):
            
            #update the state
            state = new_state
            
            #increment the iteration counter
            it += 1
            
            #increment the acceptance counter
            accept_cnt += 1
            
            #update the current state probability
            p1 = p2
            
            #append errors and states
            cross_entropies.append(p1)
            states.append(state)
            if error_function is not None:
                error = error_function(state)
                errors.append(error)
                
            #print if required
            if it%print_every == 0:
                acceptance = float(accept_cnt)/float(cnt)
                s = ""
                if pretty_state is not None:
                    s = "Current state : " + pretty_state(state)
                    
                print "Entropy : ", -p1, ", Error : ", error, ", Acceptance : ", acceptance, s
                
                if acceptance < tolerance:
                    break
                
                cnt = 0
                accept_cnt = 0
    
    if error_function is None:
        errors = None
    
    return states, cross_entropies, errors