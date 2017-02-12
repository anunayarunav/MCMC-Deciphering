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
                     help="file to scramble", default=None)
   
   (options, args) = parser.parse_args(argv)
    
   filename = options.inputfile
   
   if options.inputfile is None:
        print "File name not specified. Type -h for help."
        sys.exit(2)
        
   char_to_ix, ix_to_char, tr, fr = compute_statistics(filename)
   text = list(open(filename, 'r').read())
   p_map = generate_random_permutation_map(char_to_ix.keys())
   scrambled_t = scramble_text(text, p_map)
   print ''.join(scrambled_t)
   
if __name__ == "__main__":
   main(sys.argv)