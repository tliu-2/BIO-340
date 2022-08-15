import pprint  # pretty print for printing dictionaries
from random import choice

# Lecture Exercises:
# FASTA Parser:
def fasta_parser(file):
    seqs = {}
    curr_seq_id = None
    curr_line = []

    for line in file:
        line = line.strip()  # remove any leading / trailing spaces
        if line.startswith('>'):  # If on identifier line execute:
            if curr_seq_id is not None:  # If curr_seq has a value update dictionary value.
                seqs[curr_seq_id] = ''.join(curr_line)
            curr_seq_id = line  # get the identifier line.
            curr_seq_id = curr_seq_id.replace('>', '')  # remove the '>'; could also do line[1:] instead.
            curr_line = []  # reset to blank.
            continue

        # Add seq to the current line.
        curr_line.append(line)

    seqs[curr_seq_id] = ''.join(curr_line)  # join the value in dict with the list.
    print(seqs)
    return seqs


def calc_dinucleotide(seqs):
    # Dinucleotide Content:
    dinucleotides = ['AT', 'AA', 'AC', 'AG',
                     'TT', 'TA', 'TC', 'TG',
                     'GT', 'GA', 'GC', 'GG', 'CT', 'CA', 'CC', 'CG']

    seq_dict = {}  # Dictionary for storing dinucleotide content for each sequence.
    for key, value in seqs.items():  # Iterate through the dictionary from above.
        print(f'Current Seq: {key} \n')
        dinucleotide_content_dict = {}  # Dictionary for dinucleotide content for each dinucleotide
        for x in dinucleotides:
            print(f'{x} Content: {value.count(x.lower())}')
            dinucleotide_content_dict[x] = value.count(x.lower())

        seq_dict[key] = dinucleotide_content_dict
        print()

    print("Printing seq_dict:")
    print(seq_dict)  # Dictionary of dictionaries.
    return seq_dict


def transcribe_dna(seqs):
    rna_seqs = {}
    for key, value in seqs.items():
        rna_seq = ''
        for nt in value:
            if nt == 'a':
                rna_seq += 'U'
            elif nt == 'c':
                rna_seq += 'G'
            elif nt == 't':
                rna_seq += 'A'
            else:  # nt == 'g'
                rna_seq += 'C'
        rna_seqs[key] = rna_seq
    return rna_seqs


# Codon Usage Bias:
def calc_codon_usage(rna_seq):
    codons = ['UUU', 'UUC', 'UUA', 'UUG', 'CUU', 'CUC', 'CUA', 'CUG',
              'AUU', 'AUC', 'AUA', 'AUG', 'GUU', 'GUC', 'GUA', 'GUG',
              'UCU', 'UCC', 'UCA', 'UCG', 'CCU', 'CCC', 'CCA', 'CCG',
              'ACU', 'ACC', 'ACA', 'ACG', 'GCU', 'GCC', 'GCA', 'GCG',
              'UAU', 'UAC', 'UAA', 'UAG', 'CAU', 'CAC', 'CAA', 'CAG',
              'AAU', 'AAC', 'AAA', 'AAG', 'GAU', 'GAC', 'GAA', 'GAG',
              'UGU', 'UGC', 'UGA', 'UGG', 'CGU', 'CGC', 'CGA', 'CGG',
              'AGU', 'AGC', 'AGA', 'AGG', 'GGU', 'GGC', 'GGA', 'GGG']

    # Dictionary for storing sequences with associated codon usage.
    seq_dict = {}
    for key, value in rna_seq.items():
        # Create a dictionary from codon list where values are all 0.
        codon_dict = dict.fromkeys(codons, 0)

        # Iterate through the sequence.
        # Step by 3 to get codons.
        # Assumes valid sequence divisible by 3.
        for pos in range(0, len(value), 3):
            for codon in codons:  # Iterate through codon list.
                # If codon match, increment value in codon_dict.
                if value[pos:pos + 3] == codon:
                    codon_dict[codon] += 1

        # Store codon usages with associated sequence id (key).
        seq_dict[key] = codon_dict
    return seq_dict


# Monte Carlo
def roll_die(sides=[1, 2, 3, 4, 5, 6]):
    result = choice(sides)
    return result


if __name__ == '__main__':
    # f = open("C:/Users/TNYae/Desktop/BIO-340/test.ffn")
    f = open("Z:/School/BIO-340/test.ffn")
    sequences = fasta_parser(f)
    inucleotides = calc_dinucleotide(sequences)
    rna = transcribe_dna(sequences)
    print('RNA Seqs:')
    pprint.pprint(rna)
    print()

    print('Codon Usage:')
    codon_usage = calc_codon_usage(rna)
    pprint.pprint(codon_usage)  # Pretty print the dictionary

    # Monte Carlo


# %%
