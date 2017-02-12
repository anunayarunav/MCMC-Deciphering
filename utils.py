import numpy as np
import random
from copy import deepcopy

def az_list():
    """
    Returns a default a-zA-Z characters list
    """
    cx = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    return cx

def generate_random_permutation_map(chars):
    """
    Generate a random permutation map for given character list. Only allowed permutations
    are alphabetical ones. Helpful for debugging
    
    Arguments:
    chars: list of characters
    
    Returns:
    p_map: a randomly generated permutation map for each character
    """
    cx = az_list()
    cx2 = az_list()
    random.shuffle(cx2)
    p_map = generate_identity_p_map(chars)
    for i in xrange(len(cx)):
        p_map[cx[i]] = cx2[i]
        
    return p_map
    
def generate_identity_p_map(chars):
    """
    Generates an identity permutation map for given list of characters
    
    Arguments:
    chars: list of characters
    
    Returns:
    p_map: an identity permutation map
    
    """
    p_map = {}
    for c in chars:
        p_map[c] = c
    
    return p_map
    
def scramble_text(text, p_map):
    """
    Scrambles a text given a permutation map
    
    Arguments:
    text: text to scramble, list of characters
    
    p_map: permutation map to scramble text based upon
    
    Returns:
    text_2: the scrambled text
    """
    text_2 = []
    for c in text:
        text_2.append(p_map[c])
        
    return text_2
    
def move_one_step(p_map):
    """
    Swaps two characters in the given p_map
    
    Arguments:
    p_map: A p_map
    
    Return:
    p_map_2: new p_map, after swapping the characters
    """
    
    keys = az_list()
    sample = random.sample(keys, 2)
    
    p_map_2 = deepcopy(p_map)
    p_map_2[sample[1]] = p_map[sample[0]]
    p_map_2[sample[0]] = p_map[sample[1]]
    
    return p_map_2

def pretty_string(text, full=False):
    """
    Pretty formatted string
    """
    if not full:
        return ''.join(text[1:200]) + '...'
    else:
        return ''.join(text) + '...'
    
def compute_statistics(filename):
    """
    Returns the statistics for a text file.
    
    Arguments:
    filename: name of the file
    
    Returns:
    char_to_ix: mapping from character to index
    
    ix_to_char: mapping from index to character
    
    transition_probabilities[i,j]: gives the probability of j following i, smoothed by laplace smoothing
    
    frequency_statistics[i]: gives number of times character i appears in the document
    """
    data = open(filename, 'r').read() # should be simple plain text file
    chars = list(set(data))
    N = len(chars)
    char_to_ix = {c : i for i, c in enumerate(chars)}
    ix_to_char = {i : c for i, c in enumerate(chars)}
    
    transition_matrix = np.ones((N, N))
    frequency_statistics = np.zeros(N)
    i = 0
    while i < len(data)-1:
        c1 = char_to_ix[data[i]]
        c2 = char_to_ix[data[i+1]]
        transition_matrix[c1, c2] += 1
        frequency_statistics[c1] += 1
        i += 1
        
    frequency_statistics[c2] += 1
    transition_matrix /= np.sum(transition_matrix, axis=1, keepdims=True)
    
    return char_to_ix, ix_to_char, transition_matrix, frequency_statistics
