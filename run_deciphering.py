import matplotlib.pyplot as plt
from metropolis_hastings import *
from deciphering_utils import *

#!/usr/bin/python

import sys
from optparse import OptionParser


def main(argv):
   inputfile = None
   decodefile = None
   parser = OptionParser()

   parser.add_option("-i", "--input", dest="inputfile", 
                     help="input file to train the code on")
   
   parser.add_option("-d", "--decode", dest="decode", 
                     help="file that needs to be decoded")
   
   parser.add_option("-e", "--iters", dest="iterations", 
                     help="number of iterations to run the algorithm for", default=5000)
    
   parser.add_option("-t", "--tolerance", dest="tolerance", 
                     help="percentate acceptance tolerance, before we should stop", default=0.02)
   
   parser.add_option("-p", "--print_every", dest="print_every", 
                     help="number of steps after which diagnostics should be printed", default=500)

   (options, args) = parser.parse_args(argv)

   filename = options.inputfile
   decode = options.decode
   
   if filename is None:
      print "Input file is not specified. Type -h for help."
      sys.exit(2)

   if decode is None:
      print "Decoding file is not specified. Type -h for help."
      sys.exit(2)

   char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)
   
   s = list(open(decode, 'r').read())
   scrambled_text = list(s)
   i = 0
   initial_state = get_state(scrambled_text, tr, fr, char_to_ix)
   states = []
   entropies = []
   while i < 3:
      iters = options.iterations
      print_every = options.print_every
      tolerance = options.tolerance
      state, lps, _ = metropolis_hastings(initial_state, propose_a_move, compute_probability_of_state, 
                                            iters=iters, print_every=print_every, tolerance=tolerance, pretty_state=pretty_state)
      states.extend(state)
      entropies.extend(lps)
      i += 1
   
   p = zip(states, entropies)
   p.sort(key=lambda x:x[1])
   
   print "Best Guesses : "
   
   for j in xrange(1,6):
      print pretty_state(p[-j][0], full=True)
   
if __name__ == "__main__":
   main(sys.argv)