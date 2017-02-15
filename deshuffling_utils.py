import numpy as np
import random
from utils import *

def compute_log_probability_of_text(text, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text
    
    Note: This is quite slow, as it goes through the whole text to compute the probability.
    
    Arguments:
    text: text, list of characters
    
    char_to_ix: characters to index mapping
    
    frequency_statistics: frequency of character i is stored in frequency_statistics[i]
    
    transition_matrix: probability of j following i
    
    Returns:
    p: log likelihood of the given text
    """
    t = text
    cix = char_to_ix
    fr = frequency_statistics
    tm = transition_matrix
    
    i0 = cix[t[0]]
    p = np.log(fr[i0])
    i = 0
    while i < len(t)-1:
        i1 = cix[t[i+1]]
        p += np.log(tm[i0, i1])
        i0 = i1
        i += 1
        
    return p

def compute_probability_of_state(state):
    """
    Computes the probability of given state using compute_log_probability_by_counts
    """
    p = compute_log_probability_of_text(state["text"], state["char_to_ix"], 
                                        state["frequency_statistics"], state["transition_matrix"])
    
    return p

def get_state(text, transition_matrix, frequency_statistics, char_to_ix, max_len=2):
    state = {}
    state["text"] = text
    state["transition_matrix"] = transition_matrix
    state["frequency_statistics"] = frequency_statistics
    state["char_to_ix"] = char_to_ix
    state["max_len"] = max_len
    return state

def propose_a_move(state):
    text = state["text"]
    l = len(state["text"])
    max_len = state["max_len"]
    i1 = random.randint(0, l)
    i2 = random.randint(i1+1, i1+1+max_len)
    t2 = shuffle_text(text, i1, i2)
    new_state = {}
    for key, value in state.iteritems():
        new_state[key] = value
        
    new_state["text"] = t2
    return new_state

def pretty_state(state):
    return pretty_string(state["text"])