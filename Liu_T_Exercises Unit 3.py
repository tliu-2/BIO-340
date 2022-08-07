# Representing ... with Python Strings:
# Exercise 2.
# Write python code that converts an RNA sequence to a DNA sequence
# by replacing all characters representing Uracil with characters representing Thymidine
def transcribe_dna(dna):
    rna_sequence = ''
    for i in dna:
        if i == 'a':
            rna_sequence += 'U'
        elif i == 'c':
            rna_sequence += 'G'
        elif i == 't':
            rna_sequence += 'A'
        else:  # nt == 'g'
            rna_sequence += 'C'
    return rna_sequence

# Representing ... with Python Strings: Exercise 5 Aligned DNA sequences often have gap (-) characters in them.
# Sometimes you aren't comparing sequences and so want to remove these gaps. Write code that removes gaps from a
# sequence. HINT: replacing a character with an empty string ('') is equivalent to removing it from a sequence.

dna_seq2 = 'TTA---ACGGCA--AGC-AATTTGGCGCC-'
dna_seq2_no_gap = dna_seq2.replace('-', '')
print(f'Starting Sequence: {dna_seq2} \n'
      f'Resultant Sequence: {dna_seq2_no_gap}')
print()

# Representing ... with Python Strings: Exercise 6 Write code to calculate the percentage of a sequence that is
# gaps. HINT: you might count the number of gaps directly using the count method , or you might use your answer
# to number 5, generate an ungapped sequence, and infer the percentage of gaps by the change in sequence length
# when converting to ungapped.

dna_seq3 = 'AATCGAATC-ATC----CTGAAAATTTGG-C--T-A'
total_length = len(dna_seq3)
gap_count = dna_seq3.count('-')
percent_gap = (gap_count / total_length) * 100

print(f'Starting Sequence: {dna_seq3} \n'
      f'Total Length: {total_length} \n'
      f'Gap Count: {gap_count} \n'
      f'Percent Gap: {percent_gap}%')

# Using For Loops to Analyze Biological Sequences
# Exercise 1. Write code to calculate the frequency of each nucleotide in an RNA sequence.
# Keep these things in mind:
# Be sure that the code can be easily run on new sequences.
# Use DRY coding methods and a for loop to avoid lots of repeated code
# Be sure to check your code using a sequence where you know the right answer. For example,
# on the sequence: "AAUUGGGG", your code should return frequencies A: 25%, U:25%, G:50%, C:0%.

rna_seq = "AAACCCUUGGAUGUCUAAAUG"
counts_dict = {'A': 0, 'U': 0, 'C': 0, 'G': 0}
for nt in rna_seq:
    if nt == 'A':
        counts_dict[nt] += 1
    elif nt == 'U':
        counts_dict[nt] += 1
    elif nt == 'C':
        counts_dict[nt] += 1
    else:
        counts_dict[nt] += 1

for key, value in counts_dict.items():
    counts_dict[key] = value / len(rna_seq)

print(counts_dict)

# Using For Loops to Analyze Biological Sequences
# Exercise 2. Using the code for calculating sequence similarity,
# calculate the similarity of these two DNA sequences:
sequence_1 = "AAGTC"
sequence_2 = "AGTCG"

len_short = min(len(sequence_1), len(sequence_2))  # in this case len are the same.
same = 0
diff = 0
for x in range(len_short):
    if sequence_1[x] == sequence_2[x]:
        same += 1
    else:
        diff += 1

seq_similarity = same / len_short
print(seq_similarity)



#%%
