from collections import Counter
from re import sub, compile
import matplotlib.pyplot as plt
import numpy as np

cleaner = compile(r'[^a-zA-Z\s]')

class UnimplementedFunctionError(Exception):
	pass

class Vocabulary:

	def __init__(self, corpus):
		self.word2idx, self.idx2word, self.freq = self.build_vocab(corpus)
		self.size = len(self.word2idx)

	def most_common(self, k):
		freq = sorted(self.freq.items(), key=lambda x: x[1], reverse=True)
		return [t for t,f in freq[:k]]

	def text2idx(self, text):
		tokens = self.tokenize(text)
		return [self.word2idx[t] if t in self.word2idx.keys() else self.word2idx['UNK'] for t in tokens]

	def idx2text(self, idxs):
		return [self.idx2word[i] if i in self.idx2word.keys() else 'UNK' for i in idxs]


	###########################
	## TASK 1.1           	 ##
	###########################
	def tokenize(self, text):
		"""
	    
	    tokenize takes in a string of text and returns an array of strings splitting the text into discrete tokens.

	    :params: 
	    - text: a string to be tokenize, e.g. "The blue dog jumped, but not high."

	    :returns:
	    - tokens: a list of strings derived from the text, e.g. ["the", "blue", "dog", "jumped", "but", "not", "high"] for word-level tokenization
	    
	    """
		# Make texts lower cases
		text = text.lower()

		# Remove Number, Symbols
		text = cleaner.sub("", text)

		# Split for each token
		tokens = text.split()

		return tokens

	###########################
	## TASK 1.2            	 ##
	###########################
	def build_vocab(self, corpus):
		"""
	    
	    build_vocab takes in list of strings corresponding to a text corpus, tokenizes the strings, and builds a finite vocabulary

	    :params:
	    - corpus: a list string to build a vocabulary over

	    :returns: 
	    - word2idx: a dictionary mapping token strings to their numerical index in the dictionary e.g. { "dog": 0, "but":1, ..., "UNK":129}
	    - idx2word: the inverse of word2idx mapping an index in the vocabulary to its word e.g. {0: "dog", 1:"but", ..., 129:"UNK"}
	    - freq: a dictionary of words and frequency counts over the corpus (including words not in the dictionary), e.g. {"dog":102, "the": 18023, ...}

	    """
		# Collect all tokens in corpus
		all_tokens = []
		for sentence in corpus:
			tokens = self.tokenize(sentence)
			all_tokens.extend(tokens)

		# Count frequency
		freq = Counter(all_tokens)
		total_count = sum(freq.values())

		# Sorting tokens and Calculating cumulative fraction
		sorted_tokens = sorted(freq.items(), key=lambda x:x[1], reverse=True)
		cumulative = np.cumsum([count for token, count in sorted_tokens])
		self.cumulative_fraction = cumulative / total_count

		# Check the index of 92% cumulative coverage
		self.threshold_idx = np.argmax(self.cumulative_fraction >= 0.92)
		self.freq_threshold = sorted_tokens[self.threshold_idx][1]

		# Build the components of vocab
		word2idx, idx2word = {}, {}
		cumulative_count = 0
		idx = 0
		for token, count in sorted_tokens:
			if count < self.freq_threshold:
				continue
			if (cumulative_count / total_count) >= 0.92:
				break
			word2idx[token] = idx
			idx2word[idx] = token
			cumulative_count += count
			idx += 1

		# UNK
		word2idx["UNK"] = idx
		idx2word[idx] = "UNK"

		return word2idx, idx2word, freq

	###########################
	## TASK 1.3              ##
	###########################
	def make_vocab_charts(self):
		"""
	    
	    make_vocab_charts plots word frequency and cumulative coverage charts for this vocabulary. See handout for more details

	    
	    """
		# Sorting based on the frequency
		sorted_freq = sorted(self.freq.values(), reverse=True)

		# Set cut-off
		# freq_cutoff_y = 50
		freq_cutoff = self.freq_threshold

		plt.figure(figsize=(14, 5))

		# Token Frequency Distribution
		plt.subplot(1, 2, 1)
		plt.plot(sorted_freq)
		plt.axhline(freq_cutoff, color="red")
		plt.text(80000, freq_cutoff + 5, f"freq={freq_cutoff}", color="red")
		plt.yscale("log")
		plt.xlabel("Token ID (sorted by frequency)")
		plt.ylabel("Frequency")
		plt.title("Token Frequency Distribution")

		# Cumulative Fraction Covered
		plt.subplot(1, 2, 2)
		plt.plot(self.cumulative_fraction)
		plt.axvline(self.threshold_idx, color="red")
		plt.text(self.threshold_idx + 500, self.cumulative_fraction[self.threshold_idx] - 0.05, f"{self.cumulative_fraction[self.threshold_idx]:.2f}", color="red")
		plt.xlabel("Token ID (sorted by frequency)")
		plt.ylabel("Fraction of Token Occurrences Covered")
		plt.title("Cumulative Fraction Covered")

		plt.tight_layout()
		plt.show()
