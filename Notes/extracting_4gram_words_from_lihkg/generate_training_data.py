# Read ../shared/moedict.txt.bz2 for the moe-dict word list (line separated)

import bz2
import json
import math

def read_moe_dict():
    with bz2.open('../shared/moedict.txt.bz2', 'rt') as f:
        return set(x.strip() for x in f)

# Read ../shared/wordslist.csv.bz2 for the words.hk word list (csv format, no header, first column is the word)
def read_words_hk():
    with bz2.open('../shared/wordslist.csv.bz2', 'rt') as f:
        return set(x.split(',')[0].strip() for x in f)

# Read subgrams_from_4gh.json.bz2 for the n-gram counts
def read_subgrams():
    with bz2.open('subgrams_from_4gh.json.bz2', 'rt') as f:
        return json.load(f)

# Read 4gh.txt.bz2, the format is <count><tab><4-gram word>
def read_4gh():
    with bz2.open('4gh.txt.bz2', 'rt') as f:
        return [x.strip().split('\t')[1] for x in f]

def wordshk_convert():
        return json.loads(open("wordshk_autoconvert.json", "r").read())

def wordshk_3grams(wordshk_set):
    result = set()
    for word in wordshk_set:
        n = len(word)
        if n == 3:
            result.add((word, 0))
        elif n > 3:
            result.add((word[:3], 1))
            result.add((word[-3:], -1))

    return result

def wordshk_4subgrams(wordshk_set):
    result = set()
    for word in wordshk_set:
        n = len(word)
        for i in range(n-3):
            result.add(word[i:i+4])
    return result

def data_for_4gram(fourgram, gram_counts, is_known, three_grams):
    data = []
    # print(fourgram)
    data.append(math.log(1+gram_counts.get(fourgram, 0)))
    data.append(math.log(1+gram_counts.get(fourgram[:-1], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[1:], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[:-2], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[1:-1], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[2:], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[0], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[1], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[2], 0)))
    data.append(math.log(1+gram_counts.get(fourgram[3], 0)))
    data.append(float(is_known(fourgram[0:2])))
    data.append(float(is_known(fourgram[2:4])))
    data.append(float((fourgram[:-1], 0) in three_grams or (fourgram[:-1], -1) in three_grams))
    data.append(float((fourgram[1:], 0) in three_grams or (fourgram[1:], 1) in three_grams))
    return data


def is_cjk_cp(cp):
    return (0x3400 <= cp <= 0x4DBF) or (0x4E00 <= cp <= 0x9FFF) or (0xF900 <= cp <= 0xFAFF) or (0x20000 <= cp <= 0x2FFFF)

def is_cjk(s):
    try:
        return sum([not is_cjk_cp(ord(c)) for c in s]) == 0
    except ValueError:
        return False

def main():
    moe = read_moe_dict()
    wordshk = read_words_hk()
    threegrams = wordshk_3grams(wordshk)
    foursubgrams = wordshk_4subgrams(wordshk)
    gram_counts = read_subgrams()
    convert = wordshk_convert()
    convert_word = lambda word: "".join([convert.get(c, c) for c in word])
    is_known = lambda word: word in moe or convert_word(word) in wordshk

    # Generate training data
    # Parameters:
    # - log(4-gram count)
    # - log(3-gram count)
    # - log(3-gram count)
    # - log(2-gram count)
    # - log(2-gram count)
    # - log(2-gram count)
    # - log(1-gram count)
    # - log(1-gram count)
    # - log(1-gram count)
    # - log(1-gram count)
    # - whether left 2-gram is known
    # - whether right 2-gram is known

    with open('not_known_as_words.jsonl', 'w') as not_known, open('known_words.jsonl', 'w') as known:

        for fourgram in read_4gh():
            if len(fourgram) != 4: # Not sure why there are non-4-gram words here but we can just ignore them
                continue
            if not is_cjk(fourgram):
                continue

            data = data_for_4gram(fourgram, gram_counts, is_known, threegrams)

            line_data = {
                "data": data,
                "entity": fourgram,
            }

            if is_known(fourgram) or convert_word(fourgram) in foursubgrams:
                # Do not ensure ASCII
                known.write(json.dumps(line_data, ensure_ascii=False) + '\n')
            else:
                not_known.write(json.dumps(line_data, ensure_ascii=False) + '\n')

if __name__ == '__main__':
    main()
