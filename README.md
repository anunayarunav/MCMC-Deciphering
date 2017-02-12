MCMC Deciphering
================

Okay, so this deciphering looks fancy, right? Well actually in California Prison, prisoners were trying to transfer messages in an encoded format. Using a substituition cipher. And a Stanford Statistics professor thought of solving the code. How you wonder? Well, of course by using 'Markov Chain Monte Carlo'! Cool ri... "Uh what? Isn't MCMC a sampling method?"

Yep it is. And that is precisely what helped him solving the code. He actually used MCMC to sampling from a valid english language model by substituting different ciphers. A valid english language sample, because having a large probability of would have a higher probability of being sampled, and thus would be sampled more times than an invalid one... There by breaking the code!

Okay. Well if you'd like to know more, I'd highly recommend you to read [this paper](http://www-users.york.ac.uk/~sbc502/decode.pdf)

How to run the code
===================

Well, if you already happen to have an encoded text and wish to decode it, all you need to be concerned with `run_deciphering.py`.
Typing `python run_deciphering.py -h` would show

Usage: run_deciphering.py [options]\n

Options:
  -h, --help            show this help message and exit
  -i INPUTFILE, --input=INPUTFILE
                        input file to train the code on
  -d DECODE, --decode=DECODE
                        file that needs to be decoded
  -e ITERATIONS, --iters=ITERATIONS
                        number of iterations to run the algorithm for
  -t TOLERANCE, --tolerance=TOLERANCE
                        percentate acceptance tolerance, before we should stop
  -p PRINT_EVERY, --print_every=PRINT_EVERY
                        number of steps after which diagnostics should be
                        printed

If you don't have an inputfile or a file to be decoded, and just want to play around, you can use default files provided in data folder. For example `python run_deciphering.py -i data/warpeace_input.txt -d data/shakespeare_scrambled.txt` would use `warpeace_input.txt` as input (courtesy Andrej Karphaty, @karpathy), and `shakespeare_scrambled.txt` a scrambled version of partial shakespeare book as a file to decode.

You have your own training data and you'd like that as a basis, you're recommended to use a scrambled version of a data of similar model as a file to decode, so that language model does not appear too foreign. Right now it works only on english, however you can modify the function the alphabets in `az_list.py` in `util.py`, to use your own model in a foreign language. 

If you wish to scramble a piece of text, use `python scramble_text.py -i filename > output`. It'd lessen your loads by tons! 

Also, you are welcome!

TODO
====

I have a few ideas to test by modifying and trying different language probability models, as basis. Also since right now it works only on codes based on substituition ciphers I'd like to see if I can figure out a way to use the same model for 
