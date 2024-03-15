# Read 4gh.txt and extract 1,2,3grams from it given the input files
# 4gh.txt Format: <count> <4gram>
import sys
import os
import glob
import collections
import gzip
import json

def generate_grams(s):
    for i in range(len(s) - 3):
        yield s[i:i+4]
    for i in range(len(s) - 2):
        yield s[i:i+3]
    for i in range(len(s) - 1):
        yield s[i:i+2]
    for i in range(len(s)):
        yield s[i]

def main():
    counts = collections.Counter()

    wanted_grams = set()
    with open('4gh.txt', 'r') as f:
        for line in f:
            count, four_gram = line.split('\t', 1)
            four_gram = four_gram.strip()
            if len(four_gram) != 4:
                continue
            counts[four_gram] = int(count)

            wanted_grams.add(four_gram[:3])
            wanted_grams.add(four_gram[1:])
            wanted_grams.add(four_gram[1:3])
            wanted_grams.add(four_gram[:2])
            wanted_grams.add(four_gram[2:])
            wanted_grams.add(four_gram[0])
            wanted_grams.add(four_gram[1])
            wanted_grams.add(four_gram[2])
            wanted_grams.add(four_gram[3])

    files = glob.glob(sys.argv[1])

    for file in files:
        print(file)
        if file.endswith('.gz'):
            f = gzip.open(file, 'rt')
        else:
            f = open(file, 'rt')

        while line := f.readline():
            line = line.strip()
            # input is in the form YYYY-mm-dd,<text>
            s = line.strip()
            _, line = line.split(',', 1)

            for gram in generate_grams(line):
                if gram in wanted_grams:
                    counts[gram] += 1
        f.close()

    open('subgrams_from_4gh.json', 'w').write(json.dumps(counts, indent=1))

if __name__ == '__main__':
    main()
