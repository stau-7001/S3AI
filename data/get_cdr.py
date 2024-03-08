import os
import subprocess
import re
import csv

input_file = './processed_data.csv'
output_file = './processed_data_cdr.csv'

# Function to run AbRSA and return the output as a string
def run_abrsa(fasta_file):
    command = ['./AbRSA', '-i', fasta_file, '-c', '-o', 'ab_numbering.txt']
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout

# Function to extract CDR sequences from AbRSA output
def extract_cdrs(output):
    cdr_regex = re.compile(r'(H|L)_CDR(\d)\s*:\s*([A-Za-z]+)')
    cdrs = cdr_regex.findall(output)
    return {f'{chain}_CDR{number}': sequence for chain, number, sequence in cdrs}

with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
    reader = csv.reader(infile)
    writer = csv.writer(outfile)

    header = next(reader)
    header.extend(['H_CDR1', 'H_CDR2', 'H_CDR3', 'L_CDR1', 'L_CDR2', 'L_CDR3'])
    writer.writerow(header)

    for row in reader:
        vh_seq = row[-2]
        vl_seq = row[-1]

        with open("temp.fasta", "w") as temp_fasta:
            temp_fasta.write(f">temp_vh\n{vh_seq}\n>temp_vl\n{vl_seq}\n")

        abrsa_output = run_abrsa("temp.fasta")
        cdrs = extract_cdrs(abrsa_output)

        row.extend([cdrs['H_CDR1'], cdrs['H_CDR2'], cdrs['H_CDR3'], cdrs['L_CDR1'], cdrs['L_CDR2'], cdrs['L_CDR3']])
        writer.writerow(row)

    os.remove("temp.fasta")
