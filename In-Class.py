f = open("C:/Users/TNYae/Desktop/BIO-340/test.ffn")

parsed_seqs = {}
curr_seq_id = None
curr_seq = []

for line in f:
    line = line.strip()
    if line.startswith('>'):
        if curr_seq_id is not None:
            parsed_seqs[curr_seq_id] = ''.join(curr_seq)
        curr_seq_id = line[1:]
        curr_seq = []
        continue
    curr_seq.append(line)

parsed_seqs[curr_seq_id] = ''.join(curr_seq)
print(parsed_seqs)

dinucleotides = ['AT', 'AA', 'AC', 'AG',
                 'TT', 'TA', 'TC', 'TG',
                 'GT', 'GA', 'GC', 'GG', 'CT', 'CA', 'CC', 'CG']

for key, value in parsed_seqs.items():
    print(f'Current Seq: {key} \n')
    for x in dinucleotides:
        print(f'{x} Content: {value.count(x.lower())}')
    print()
