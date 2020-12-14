from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq

regions = set()
reg_counts = defaultdict(int)

records = list(SeqIO.parse("../data/balanced_seqs.fasta", "fasta"))
new_records = list(SeqIO.parse("../data/balanced_seqs.fasta", "fasta"))

for record in new_records:
    record.seq = Seq("")


def is_conserved(base_counts, num_records):
    cutoff = 0.99
    for base in base_counts:
        if base_counts[base] >= cutoff*num_records:
            return True
    return False


for i in range(len(records[0].seq)):
    base_counts = defaultdict(int)
    bases = [r.seq[i] for r in records]
    for base in bases:
        base_counts[base] += 1
    if not is_conserved(base_counts, len(records)):
        for j in range(len(records)):
            new_records[j].seq += bases[j]
            
SeqIO.write(new_records, "../data/balanced_variable_seqs.fasta", "fasta")
