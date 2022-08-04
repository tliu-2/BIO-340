# f = open("C:/Users/TNYae/Desktop/BIO-340/test.ffn")

# Lecture Exercises:
# FASTA Parser:
f = open("Z:/School/BIO-340/test.ffn")
seqs = {}
curr_seq_id = None
curr_line = []

for line in f:
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

# Dinucleotide Content:
dinucleotides = ['AT', 'AA', 'AC', 'AG',
                 'TT', 'TA', 'TC', 'TG',
                 'GT', 'GA', 'GC', 'GG', 'CT', 'CA', 'CC', 'CG']

seq_dict = {}  # Dictionary for storing dinucleotide content for each sequence.
for key, value in seqs.items(): # Iterate through the dictionary from above.
    print(f'Current Seq: {key} \n')
    dinucleotide_content_dict = {}  # Dictionary for dinucleotide content for each dinucleotide
    for x in dinucleotides:
        print(f'{x} Content: {value.count(x.lower())}')
        dinucleotide_content_dict[x] = value.count(x.lower())

    seq_dict[key] = dinucleotide_content_dict
    print()

print("Printing seq_dict:")
print(seq_dict)  # Dictionary of dictionaries.

#%%
