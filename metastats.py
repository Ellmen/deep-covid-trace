from collections import defaultdict

from Bio import SeqIO

regions = set()
reg_counts = defaultdict(int)

for record in SeqIO.parse("./data/msa.fasta", "fasta"):
    sep_idx = record.id.rfind('|')
    region = record.id[sep_idx+1:]
    reg_counts[region] += 1
    # regions.add(region)

print(reg_counts)
