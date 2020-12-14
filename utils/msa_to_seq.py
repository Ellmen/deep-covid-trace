from collections import defaultdict

from Bio import SeqIO
from Bio.Seq import Seq

regions = set()
reg_counts = defaultdict(int)

records = []

for record in SeqIO.parse("../data/msa.fasta", "fasta"):
    record.seq = Seq(str(record.seq).replace('-', ''))
    records.append(record)

SeqIO.write(records, "../data/seqs.fasta", "fasta")
