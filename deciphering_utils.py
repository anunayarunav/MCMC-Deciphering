import numpy as np
import random
from utils import *

def compute_log_probability(text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the 
    charcter c from permutation_map[c]), given the text statistics
    
    Note: This is quite slow, as it goes through the whole text to compute the probability,
    if you need to compute the probabilities frequently, see compute_log_probability_by_counts.
    
    Arguments:
    text: text, list of characters
    
    permutation_map[c]: gives the character to replace 'c' by
    
    char_to_ix: characters to index mapping
    
    frequency_statistics: frequency of character i is stored in frequency_statistics[i]
    
    transition_matrix: probability of j following i
    
    Returns:
    p: log likelihood of the given text
    """
    t = text
    p_map = permutation_map
    cix = char_to_ix
    fr = frequency_statistics
    tm = transition_matrix
    
    i0 = cix[p_map[t[0]]]
    p = np.log(fr[i0])
    i = 0
    while i < len(t)-1:
        subst = p_map[t[i+1]]
        i1 = cix[subst]
        p += np.log(tm[i0, i1])
        i0 = i1
        i += 1
        
    return p

def compute_transition_counts(text, char_to_ix):
    """
    Computes transition counts for a given text, useful to compute if you want to compute 
    the probabilities again and again, using compute_log_probability_by_counts.
    
    Arguments:
    text: Text as a list of characters
    
    char_to_ix: character to index mapping
    
    Returns:
    transition_counts: transition_counts[i, j] gives number of times character j follows i
    """
    N = len(char_to_ix)
    transition_counts = np.zeros((N, N))
    c1 = text[0]
    i = 0
    while i < len(text)-1:
        c2 = text[i+1]
        transition_counts[char_to_ix[c1],char_to_ix[c2]] += 1
        c1 = c2
        i += 1
    
    return transition_counts

def compute_log_probability_by_counts(transition_counts, text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the 
    charcter c from permutation_map[c]), given the transition counts and the text
    
    Arguments:
    
    transition_counts: a matrix such that transition_counts[i, j] gives the counts of times j follows i,
                       see compute_transition_counts
    
    text: text to compute probability of, should be list of characters
    
    permutation_map[c]: gives the character to replace 'c' by
    
    char_to_ix: characters to index mapping
    
    frequency_statistics: frequency of character i is stored in frequency_statistics[i]
    
    transition_matrix: probability of j following i stored at [i, j] in this matrix
    
    Returns:
    
    p: log likelihood of the given text
    """
    c0 = char_to_ix[permutation_map[text[0]]]
    p = np.log(frequency_statistics[c0])
    
    p_map_indices = {}
    for c1, c2 in permutation_map.iteritems():
        p_map_indices[char_to_ix[c1]] = char_to_ix[c2]
    
    indices = [value for (key, value) in sorted(p_map_indices.items())]
    
    p += np.sum(transition_counts*np.log(transition_matrix[indices,:][:, indices]))
    
    return p

def compute_difference(text_1, text_2):
    """
    Compute the number of times to text differ in character at same positions
    
    Arguments:
    
    text_1: first text list of characters
    text_2: second text, should have same length as text_1
    
    Returns
    cnt: number of times the texts differ in character at same positions
    """
    
    cnt = 0
    for x, y in zip(text_1, text_2):
        if y != x:
            cnt += 1
            
    return cnt

def get_state(text, transition_matrix, frequency_statistics, char_to_ix):
    """
    Generates a default state of given text statistics
    
    Arguments:
    pretty obvious
    
    Returns:
    state: A state that can be used along with,
           compute_probability_of_state, propose_a_move,
           and pretty_state for metropolis_hastings
    
    """
    transition_counts = compute_transition_counts(text, char_to_ix)
    p_map = generate_identity_p_map(char_to_ix.keys())
    
    state = {"text" : text, "transition_matrix" : transition_matrix, 
             "frequency_statistics" : frequency_statistics, "char_to_ix" : char_to_ix,
            "permutation_map" : p_map, "transition_counts" : transition_counts}
    
    return state

def compute_probability_of_state(state):
    """
    Computes the probability of given state using compute_log_probability_by_counts
    """
    
    p = compute_log_probability_by_counts(state["transition_counts"], state["text"], state["permutation_map"], 
                                          state["char_to_ix"], state["frequency_statistics"], state["transition_matrix"])
    
    return p

def propose_a_move(state):
    """
    Proposes a new move for the given state, 
    by moving one step (randomly swapping two characters)
    """
    new_state = {}
    for key, value in state.iteritems():
        new_state[key] = value
    new_state["permutation_map"] = move_one_step(state["permutation_map"])
    return new_state

def pretty_state(state, full=False):
    """
    Returns the state in a pretty format
    """
    if not full:
        return pretty_string(scramble_text(state["text"][1:200], state["permutation_map"]), full)
    else:
        return pretty_string(scramble_text(state["text"], state["permutation_map"]), full)
