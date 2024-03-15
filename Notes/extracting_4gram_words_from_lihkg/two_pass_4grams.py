# Find common 4-grams in two pass
# The problem is that the 4-grams are too large to fit in memory
# The solution is to use two pass, first pass to count the 4-grams and their
# frequencies using a limited size hash table. There are collisions, so but
# that is OK, since we just want to know the most frequent 4-grams. The second
# pass is to find the most frequent 4-grams using the hash table.

import collections
import glob
import gzip
import sys

hash_table = collections.Counter()
real_counter = collections.Counter()
above_threshold = set()

def hash_4gram(s):
    assert len(s) == 4
    if ord(s[0]) < 255: return None
    result =  (ord(s[0]) % 256) << 24
    result += (ord(s[1]) % 256) << 16
    result += (ord(s[2]) % 256) << 8
    result += (ord(s[3]) % 256)
    result ^= ((ord(s[0]) + ord(s[1]) + ord(s[2]) + ord(s[3])) % 32768) << 16
    result ^= ((ord(s[0]) + ord(s[1]) + ord(s[2]) + ord(s[3])) % 32768)
    # assert result <= 2**32, f'{s} {result} {ord(s[0]) % 256}'
    return result % (2**30)


def main():
    total_4grams = 0
    # read from glob
    files = glob.glob(sys.argv[1])
    print(files)
    max_h = 0

    for file in files:
        print(len(hash_table))
        print(file)
        if file.endswith('.gz'):
            f = gzip.open(file, 'rt')
        else:
            f = open(file, 'rt')

        while line := f.readline():
            # input is in the form YYYY-mm-dd,<text>
            s = line.strip()
            _, line = line.split(',', 1)
            for i in range(0, len(line) - 4):
                s = line[i:i+4]
                h = hash_4gram(s)
                if h is None: continue
                if h > max_h: max_h = h
                # print(s, h)
                hash_table[h] += 1
                total_4grams += 1
        f.close()

    threshold = 100
    print(max_h, total_4grams)

    for file in files:
        print(len(real_counter))
        print(file)
        if file.endswith('.gz'):
            f = gzip.open(file, 'rt')
        else:
            f = open(file, 'rt')

        while line := f.readline():
            # input is in the form YYYY-mm-dd,<text>
            s = line.strip()
            _, line = line.split(',', 1)
            for i in range(0, len(line) - 4):
                s = line[i:i+4]
                h = hash_4gram(s)
                if h is None: continue

                if hash_table[h] > threshold:
                    real_counter[s] += 1
                    above_threshold.add((hash_table[h], h, s))
        f.close()

    with open('4gh.txt', 'w') as f:
        for x in sorted(real_counter.items(), key=lambda x: x[1], reverse=True):
            if x[1] < threshold:
                break
            f.write(f'{x[1]}\t{x[0]}\n')

    for x in sorted(above_threshold, reverse=True):
        print(x)


if __name__ == '__main__':
    main()
