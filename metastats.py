from collections import defaultdict

from Bio import SeqIO

reg_counts = defaultdict(int)

for record in SeqIO.parse("./data/balanced_seqs.fasta", "fasta"):
    sep_idx = record.description.rfind('|')
    region = record.description[sep_idx+1:]
    reg_counts[region] += 1

print(reg_counts)
