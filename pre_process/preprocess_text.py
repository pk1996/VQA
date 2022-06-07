'''
Script to generate the vocab
'''
import json
import os
from collections import Counter
import argparse
import itertools


def main(path):
    questions = get_path(path, train=True, question=True)
    answers = get_path(path, train=True, answer=True)

    print('Reading data.....')
    with open(questions, 'r') as fd:
        questions = json.load(fd)
    with open(answers, 'r') as fd:
        answers = json.load(fd)
    
    print('Processing Vocab.....')
    questions = get_questions(questions)
    answers = get_answers(answers)
    questions = extract_vocab_2(questions)
    answers = extract_vocab_2(answers,end_idx=3000)
    vocabs={'questions':questions,'answers':answers}
    
    print('Dumping as json file.....')
    with open('../data/vocabulary_alternate.json', 'w') as fd:
        json.dump(vocabs, fd)

def extract_vocab_2(gen,start_idx=0,end_idx=None):
    import itertools
    from collections import Counter
    all_tokens = itertools.chain.from_iterable(gen)
    counter = Counter(all_tokens)
    if end_idx:
        most_common = counter.most_common(end_idx)
        most_common = (t for t, c in most_common)
    else:
        most_common = counter.keys()
    tokens = sorted(most_common, key=lambda x: (counter[x], x), reverse=True)
    vocab = {t: i for i, t in enumerate(tokens, start=start_idx)}
    return vocab

def get_answers(json_file):
    import re
    _period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')
    _comma_strip = re.compile(r'(\d)(,)(\d)')
    _punctuation_chars = re.escape(r';/[]"{}()=+\_-><@`,?!')
    _punctuation = re.compile(r'([{}])'.format(re.escape(_punctuation_chars)))
    _punctuation_with_a_space = re.compile(r'(?<= )([{0}])|([{0}])(?= )'.format(_punctuation_chars))
    
    def process_punctuation(s):
        if _punctuation.search(s) is None:
            return s
        s = _punctuation_with_a_space.sub('', s)
        if re.search(_comma_strip, s) is not None:
            s = s.replace(',', '')
        s = _punctuation.sub(' ', s)
        s = _period_strip.sub('', s)
        return s.strip()
    answers = [[a['answer'] for a in d['answers']] for d in json_file['annotations']]
    for answer_list in answers:
        yield list(map(process_punctuation, answer_list))

def get_questions(json_file):
    questions = [a['question'] for a in json_file['questions']]
    for q in questions:
        q = q.lower()[:-1]
        yield q.split(' ')

def get_path(path, train=False,val=False,test=False,question=False,answer=False):
    import os
    assert train==True or val==True or test==True
    assert question==True or answer==True
    assert answer*test == 0
    
    if train:
        add1='train/'
    elif val:
        add1='val/'
    else:
        add1='test/'
    path = path + add1
    
    if question:
        addpath = 'questions.json'
    elif answer:
        addpath = 'annotations.json'
    return os.path.join(path, addpath)

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataPath', default = '../data/', help = 'path to data dir')
    opt = parser.parse_args()

    main(opt.dataPath)
    