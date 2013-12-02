import os
import sys
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from gpu_pssm import gpu_pssm
from mg_perm_analysis import metagenomics
import numpy as np
import math

bases = "ACGT"

def main():
    motif_filename = "Gamma_collection.txt"
    genome_filename = "NC_000913.fna" # Escherichia coli str. K-12 substr. MG1655
    top_sites = ["TACTGTATAAATAAACAGTA", "TACTGTATATAAAAACAGTA", "TACTGTATATAAAAACAGTA", "AACTGTATATAAATACAGTT", "TACTGTATATAAAACCAGTT", "TACTGGATAAAAAAACAGTT", "TACTGTATAAAATCACAGTT", "AACTGGATAATCATACAGTA", "CACTGTATAAATAAACAGCT", "TACTGTACATCCATACAGTA", "ACCTGTATAAATAACCAGTA", "TACTGTATGAGCATACAGTA", "TACTGGATATTTAAACAGGT", "AACTGTATATACACCCAGGG", "TGCTGTATATACTCACAGCA", "AACTGGATAAAATTACAGGG", "TACTGTATATTCATTCAGGT", "TACTGTACACAATAACAGTA", "AGCTGAATAAATATACAGCA", "ATCTGTATATATACCCAGCT", "CACTGGATAGATAACCAGCA", "TACTGCATATACAACCAGAA", "CACTGTATACTTTACCAGTG", "AACTGTCGATACGTACAGTA", "TGCTGTACAAACGTCCAGTT", "CGCTGGATATCTATCCAGCA", "TGCTGTTTATTAAACCAGAA", "CATTGTTTATAAAAACAGCA", "TACTGGAGACAAATACAGCT", "ACCTGTATATATCATCAGTA", "TACTGTATAAACAGCCAATA", "TACTGTTTATCTTCCCAGCG", "GACTGTATAAAACCACAGCC", "TATTGTATATATTCACATTA", "AAGTGTATTTACACACAGCG", "AACTGGATAATCATACCGTT", "TGCTGTATGGATTAACAGGA", "TCCTGTATGAAAAACCATTA", "GCCTGTCTGAACAAACAGTA", "AAGTGTATATATATCCATCG", "AATTGTTTAAAAAACCAGAA", "TATTGTATTTATAAACATTA", "AACTGATTAAAAACCCAGCG", "GAGTGTATATAAAGCCAGAA", "TTCTGGATAAGCATCCAGAA", "ACCTGAATATTCAAACAGCG", "TACTGTCTACCAAAACAGAG", "TCCTGTATAAATTAACCGTT", "TAATGAATATAAAACCAGGA", "AACTGTAAATAATTACATGA", "TACTGAATAAAAAAGCAGAA", "TGCTGTACAATCAGCCAGCA", "TGCTGGATTTACGACCAGAA", "TACTGTTTATAAACCGAGCG", "TCTTGTATATCCAACCAGTT", "CACTGTATTAAAAACCATTC", "AACTGTATTACCTTCCAGCC", "TACTGTTTCCATTTACAGCC", "TAGTGGATGTAAAAACATTT", "CACTGTCTATACTTACATGT", "TACTGTTTGTGCAAATAGTA", "AACTGGTTATCAACCCAGAC", "CTCTGTATCTAATTACAGGT", "CACTGAATGCTAAAACAGCA", "TGCTGGATGTGAAACCAGCG", "TGCTGATTAAAAAACCAGCG", "AACTGGATATCTATCCGGAA", "CACTGTCTGATACAACAGTT", "AATTGTTAATATATCCAGAA", "AACTTTATGTACAGCCAGTG", "AACTGGTTATTCCACCAGAA", "GAATGGATAAAAAAACAGCC", "AGCTGTATAAAAATCCTGAG", "TCCTGGCTATTTTGCCAGTA", "AGCTGGCTATCTGAACAGTT", "CACTGGATATTCCTTCAGGT", "GGCTGGATAAAGAACCAGAA", "AACTGAATAAAAACAAAGGA", "TGCTGTACATCAGCACAGAT", "CAGTGGATTAACTTCCAGTT", "TACTGGATACAAAAACGGAT", "AACTATATAAATAAACATAA", "TGCTGCATGAAGAAACAGTA", "GTCTGAATGAATACCCAGTA", "TACTTTATTTACTCCCAGTG", "TACTGGAAACTAACACAGGC", "TACTGGATATCAAACCTGAA", "TACTGTCGGAAAAATCAGTG", "AACTGTATATGTCGCCAGGC", "TACTCTTCATTAAAACAGTG", "GACTGGCTTAATACACAGCC", "CAGTGTAAGAATAGACAGTG", "CGCTGGATGAGCGTACAGCA", "AACGGGATAATAAAACAGCC", "TACCGTTTGTATTTCCAGCT", "CTCTGGACATTAAACCAGGA", "TAGTTTATATAAATTCAGTC", "TACTGTATATTCCTCAAGCG", "AACTGGATATATAAATAATG", "AACGGTTTAAACACCCAGCG", "GCCTGGTTAAATGACCAGCA", "TACTGTTTAGCCAGCCAGTC", "TGCTGAACATTCTTCCAGCA", "TACTGGAAATAAGATCAGCC", "AACTGGATCTTAAAAAAGTA", "GGCTGGACAATTTTACAGCT", "AACTGTCTCTTATGACAGTT", "AACTGGATACAGAAACAATA", "TAGTGTTCAGTTTTACAGTA", "TAGTGTTTGTTCATCCATTA", "CAGTGTTGAAAAGTCCAGTA", "ATCTGGCTGAAATTACAGAA", "GATTGAATGAATATACAGGG", "CACTGGTTTTCCACCCAGCA", "AACTGGTTGAAAAACCATCA", "AACTGGTTTAACTCCCAGGG", "AGCTGGATAAACAAAAAGCG", "AACTGGATAAATTACCGGAT", "AACTGGATTTAAATCCTGGT", "AACTGTATTTACAATCATCT", "TACTCTCTGTTCATCCAGCA", "TACTCTATGCAATAACAGAA", "TGCTGAATGCATTAACAGCG", "CACTGGTTATCTTTACCGTA", "TGGTGGATATTTTTTCAGGA", "AACTGGATATCAATCCGGAT", "AAGTGTAAAATTCTACAGAA", "AACCGGACATAAAGCCAGTA", "CACTGTATAAAAATCCTATA", "AACTCTATATTACCCCAGTT", "ACCAGAATAAACATCCAGTA", "TACTGGATGCATTACCGGTA", "CACTCTTTATAAAACCAGGC", "CAGTGAAAATAAAAACAGGA", "AACGGTTTATCTAGCCAGTA", "TCATGTATGATCATACAGAC", "TGCTGAATATATAAAAAGAG", "ACCTGGATGTCACCACAGTT", "TACTAAATGAAAAAACAGCG", "AACTGGATGAACAACCGGCG", "TTCTTTATACATATCCAGCG", "AAGTGTTTGCTAACACAGCA", "AAGTGTAAAAAATCCCAGCG", "AACTGGATAAAGACACCGCT", "TGCTGTATGGTAAATCAGAA", "AACTGTATGATTTAAAAGAT", "TACGGTATAAAAAGACCGTA", "TGCTGGATATTATCCCATCA", "TACTGTCTGAAGAAGCAGTG", "AACTGAATAAATACCCCGGT", "TGCTGTATGAGTAACCGGTA", "CACTGGAAAAATGCGCAGTA", "AACTGGATAGCTATGCAGAA", "AATTGTAAAAAACAACAGCA", "AACTGTTTATCAACACCGCT", "ACCTGTACCTTAAACCAGGA", "TACTTTATAGTTTCCCAGTT", "ATCTGCATAAAGAACCAGTA", "CCCTCTTTATATTTCCAGTG", "TACTGTTTATTAATGTAGCA", "ATGTGAATGAATATCCAGTT", "AACTGGTTAAAATTAGAGAT", "TACTGTAAGAAAAACCCGCA", "ACCTGGAGAAAGAAACAGCG", "CACTGTTTACCCTGACAGTC", "AACTGGCTCATAACCCAGAA"]
    top_seqs = [nt2int(seq) for seq in top_sites]
    top_seqs = top_seqs[0:3] # Truncate to first three for the sake of less output spam
    
    # Load genome (validated)
    genome, __ = load_scaffolds(genome_filename)
    _genome, __ = metagenomics.load_scaffolds(genome_filename)
    print "Genomes loaded equally:", np.array_equal(genome, _genome)
    print

    # Ensure the sites are in the genome (validated)
    print "Finding the sites in the genome:"
    positions = []
    sites = []
    strands = []
    for seq in top_seqs:
        matches = np.where(np.all(sliding_window(genome, seq.size) == seq, axis=1))
        matches_r = np.where(np.all(sliding_window(genome, seq.size) == wc(seq), axis=1))
        
        # forward strand
        for match in matches:
            if match.size > 0 and match[0] not in positions:
                pos = match[0]
                seq = genome[pos:pos + 20]
                print int2nt(seq), "->", match[0] 
                positions.append(pos)
                sites.append(seq)
                strands.append(0) # forward
        
        # reverse strand
        for match in matches_r:
            if match.size > 0 and match[0] not in positions:
                pos = match[0]
                seq = wc(genome[match[0]:match[0] + 20])
                print int2nt(seq), "<-", match[0]
                positions.append(pos)
                sites.append(seq)
                strands.append(1) # reverse
    print
    
    # Load PSSM (validated)
    genome_frequencies = np.bincount(genome).astype(np.float) / genome.size
    _pssm = create_pssm(motif_filename, genome_frequencies=genome_frequencies)
    _pssm2 = gpu_pssm.create_pssm(motif_filename, genome_frequencies=genome_frequencies)
    print "PSSMs created equally:", np.array_equal(_pssm, _pssm2)
    print
    
    # Score sites (validated!!)
    print "Scoring sites:"
    for n, site in enumerate(sites):
        print "True:", score_site(site, _pssm), int2nt(site), strands[n], positions[n]
        print "CPU: ", cpu_score_site(site, _pssm), int2nt(site), strands[n], positions[n]
        print "GPU: ", gpu_score_site(site, _pssm), int2nt(site), strands[n], positions[n]
        print

    # motif_seqs = load_fasta(motif_filename)
    # pscm = count_matrix([nt2int(seq) for seq in motif_seqs])
    # psfm = freq_matrix(pscm)
    # print psfm
    # genome_frequencies = [0.25, 0.25, 0.25, 0.25]
    # _pssm2 = pssm(psfm, genome_frequencies)
    # #print _pssm2
    # print score_site(top_site, _pssm2)


def pssm(psfm, genome_frequencies):
    epsilon = 1e-50
    return np.array([[math.log(f / genome_frequencies[b] + epsilon, 2) for b, f in enumerate(pos)] for pos in psfm]).flatten()


def consensus(matrix):
    return [pos.index(max(pos)) for pos in matrix]


def count_matrix(seqs):
    return [[sum([1 for seq in seqs if seq[i] == b]) for b in range(len(bases))] for i in range(len(seqs[0]))]


def freq_matrix(pscm):
    return [[float(base_count) / sum(pos) for base_count in pos] for pos in pscm]


def score_site(site, pssm):
    return sum([pssm[i * 4 + j] for i, j in enumerate(site)])


def gpu_score_site(site, pssm):
    scores = gpu_pssm.score_sequence(site, pssm, keep_strands = False)
    return scores[0]


def cpu_score_site(site, pssm):
    scores = gpu_pssm.score_sequence_with_cpu(site, pssm, keep_strands = False)
    return scores[0]


def sliding_window(a, size):
    shape = a.shape[:-1] + (a.shape[-1] - size + 1, size)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def load_fasta(filename):
    with open(filename) as f:
        lines = f.readlines()
        lines = [line.strip() for line in lines if ">" not in line]
    return lines


def load_scaffolds(filename):
    scaffolds = []
    
    with open(filename, "r+") as f:
        # Pre-allocate sequence to size of file
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0, os.SEEK_SET)
        sequence = np.tile(-1, size)
        
        # Process the lines in the file
        pos = 0
        next_line_starts_scaffold = False
        for line in f:
            if ">" in line:
                next_line_starts_scaffold = True
            else:
                # Keep sequence as the charcode integers
                scaffold_seq = np.array(line.strip(), "c").view(np.uint8)
                
                # Replace the sequence chunk in memory
                sequence[pos:pos + len(scaffold_seq)] = scaffold_seq
                
                if next_line_starts_scaffold:
                    # Keep track of the beginning of scaffolds
                    scaffolds.append(pos)
                    next_line_starts_scaffold = False
                
                # Update position in the sequence array
                pos += len(scaffold_seq)
                
    # Truncate the sequence array to fit the data
    sequence = sequence[0:np.where(sequence == -1)[0][0]]
    
    # Replace character codes with 0-3 int representation of the nucleotides
    for value, base in enumerate(bases):
        sequence[np.where(sequence == ord(base))] = value
        
    return sequence, scaffolds


def create_pssm(motif_filename, genome_frequencies = [0.25] * 4, epsilon = 1e-50):
    # Initialize language and counts
    bases = "ACGT"
    counts = np.array([])
    
    # Read in file "without slurping"
    with open(motif_filename, "r+") as f:
        for line in f:
            if ">" not in line:
                # Initialize counts array
                if counts.size is 0:
                    counts = np.zeros(len(line.strip()) * 4)
                
                # Add to nucleotide count per position
                for pos, base in enumerate(line.strip()):
                    counts[pos * 4 + bases.index(base)] += 1
    
    # Intialize PSSM array
    pssm = np.zeros(counts.size)
    
    # Calculate PSSM
    for pos in range(counts.size / 4):
        total_count = sum(counts[pos * 4:pos * 4 + 4])
        genomic_entropy = -sum([p_i * math.log(p_i, 2) for p_i in genome_frequencies])
        for b in range(4):
            # p = frequency of this base in the genome
            p = genome_frequencies[b]
            
            ### Laplacian
            # pseudo_freq = frequency of this base at this position using a
            # Laplacian pseudocount
            pseudo_freq = (counts[pos * 4 + b] + p) / (total_count + 1)
            
            # Each entry in the PSSM is the log-likelihood of that base at that
            # position.
            pssm[pos * 4 + b] = math.log(pseudo_freq / p, 2)
            

            ### Epsilon pseudocount
            # f = frequency of this base at this position
            #f = counts[pos * 4 + b] / total_count
            
            # We add a very small epsilon to avoid log(0) situations, which
            # should yield -Inf, but is not supported by Python. This is known
            # as a computational pseudocount
            #pssm[pos * 4 + b] = math.log(f / p + epsilon, 2)


            ### Information score
            #f = counts[pos * 4 + b] / total_count
            #pssm[pos * 4 + b] = math.log(f + epsilon, 2) + genomic_entropy
    return pssm


def wc(int_seq):
    complements = [np.array([])] * len(bases)
    for b in range(len(bases)):
        complements[b] = np.where(int_seq == b)
    _int_seq = np.copy(int_seq)
    for b, indices in enumerate(complements):
        _int_seq[indices] = len(bases) - b - 1
    return _int_seq[::-1]


def int2nt(int_seq):
    return "".join([bases[b] for b in int_seq])


def nt2int(nt_seq):
    sequence = np.array(nt_seq, "c").view(np.uint8)
    for value, base in enumerate(bases):
        sequence[np.where(sequence == ord(base))] = value
    return sequence


if __name__ == "__main__":
    main()