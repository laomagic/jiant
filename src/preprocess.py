'''Preprocessing functions and pipeline'''
import os
import logging as log
from collections import defaultdict
import ipdb as pdb # pylint disable=unused-import
import _pickle as pkl
import numpy as np
import torch

from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp_mods.numeric_field import NumericField

from tasks import SingleClassificationTask, PairClassificationTask, \
                  PairRegressionTask, SequenceGenerationTask, RankingTask, \
                  CoLATask, MRPCTask, MultiNLITask, QQPTask, RTETask, \
                  QNLITask, SNLITask, SSTTask, STSBTask, WNLITask

if "cs.nyu.edu" in os.uname()[1] or "dgx" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'
PATH_PREFIX = PATH_PREFIX + 'processed_data/mtl-sentence-representations/'

ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnli', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
NAME2INFO = {'sst': (SSTTask, 'SST-2/'),
             'cola': (CoLATask, 'CoLA/'),
             'mrpc': (MRPCTask, 'MRPC/'),
             'qqp': (QQPTask, 'QQP'),
             'sts-b': (STSBTask, 'STS-B/'),
             'mnli': (MultiNLITask, 'MNLI/'),
             'qnli': (QNLITask, 'QNLI/'),
             'rte': (RTETask, 'RTE/'),
             'snli': (SNLITask, 'SNLI/'),
             'wnli': (WNLITask, 'WNLI/')
            }
for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], PATH_PREFIX + v[1])

def build_tasks(args):
    '''Prepare tasks'''

    def parse_tasks(task_list):
        '''parse string of tasks'''
        if task_list == 'all':
            tasks = ALL_TASKS
        elif task_list == 'none':
            tasks = []
        else:
            tasks = task_list.split(',')
        return tasks

    train_task_names = parse_tasks(args.train_tasks)
    eval_task_names = parse_tasks(args.eval_tasks)
    all_task_names = list(set(train_task_names + eval_task_names))
    tasks = get_tasks(all_task_names, args.max_seq_len, args.load_tasks)

    max_v_sizes = {'word': args.max_word_v_size}
    token_indexer = {}
    if args.elmo:
        token_indexer["elmo"] = ELMoTokenCharactersIndexer("elmo")
        if not args.elmo_no_glove:
            token_indexer["words"] = SingleIdTokenIndexer()
    else:
        token_indexer["words"] = SingleIdTokenIndexer()

    vocab_path = os.path.join(args.exp_dir, 'vocab')
    preproc_file = os.path.join(args.exp_dir, args.preproc_file)
    if args.load_preproc and os.path.exists(preproc_file):
        preproc = pkl.load(open(preproc_file, 'rb'))
        vocab = Vocabulary.from_files(vocab_path)
        word_embs = preproc['word_embs']
        for task in tasks:
            train, val, test = preproc[task.name]
            task.train_data = train
            task.val_data = val
            task.test_data = test
        log.info("\tFinished building vocab. Using %d words",
                 vocab.get_vocab_size('tokens'))
        log.info("\tLoaded data from %s", preproc_file)
    else:
        log.info("\tProcessing tasks from scratch")
        word2freq = get_words(tasks)
        vocab = get_vocab(word2freq, max_v_sizes)
        word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)
        preproc = {'word_embs': word_embs}
        for task in tasks:
            process_task(task, token_indexer, vocab)
            del_field_tokens(task)
            preproc[task.name] = (task.train_data, task.val_data, task.test_data)
        log.info("\tFinished indexing tasks")
        pkl.dump(preproc, open(preproc_file, 'wb'))
        vocab.save_to_files(vocab_path)
        log.info("\tSaved data to %s", preproc_file)
        del word2freq
    del preproc

    train_tasks = [task for task in tasks if task.name in train_task_names]
    eval_tasks = [task for task in tasks if task.name in eval_task_names]
    log.info('\t  Training on %s', ', '.join([task.name for task in train_tasks]))
    log.info('\t  Evaluating on %s', ', '.join([task.name for task in eval_tasks]))
    return train_tasks, eval_tasks, vocab, word_embs

def del_field_tokens(task):
    ''' Save memory by deleting the tokens that will no longer be used '''
    all_instances = task.train_data + task.val_data + task.test_data
    for instance in all_instances:
        if 'input1' in instance.fields:
            field = instance.fields['input1']
            del field.tokens
        if 'input2' in instance.fields:
            field = instance.fields['input2']
            del field.tokens

def get_tasks(task_names, max_seq_len, load):
    '''
    Load tasks
    '''
    tasks = []
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        pkl_path = NAME2INFO[name][1] + "%s_task.pkl" % name
        if os.path.isfile(pkl_path) and load:
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            task = NAME2INFO[name][0](NAME2INFO[name][1], max_seq_len, name)
            pkl.dump(task, open(pkl_path, 'wb'))
        tasks.append(task)
    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks

def get_words(tasks):
    '''
    Get all words for all tasks for all splits for all sentences
    Return dictionary mapping words to frequencies.
    '''
    word2freq = defaultdict(int)

    def count_sentence(sentence):
        '''Update counts for words in the sentence'''
        for word in sentence:
            word2freq[word] += 1
        return

    for task in tasks:
        for sentence in task.sentences:
            count_sentence(sentence)

    log.info("\tFinished counting words")
    return word2freq

def get_vocab(word2freq, max_v_sizes):
    '''Build vocabulary'''
    vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes['word'])
    words_by_freq = [(word, freq) for word, freq in word2freq.items()]
    words_by_freq.sort(key=lambda x: x[1], reverse=True)
    for word, _ in words_by_freq[:max_v_sizes['word']]:
        vocab.add_token_to_namespace(word, 'tokens')
    log.info("\tFinished building vocab. Using %d words", vocab.get_vocab_size('tokens'))
    return vocab

def get_embeddings(vocab, vec_file, d_word):
    '''Get embeddings for the words in vocab'''
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, d_word)
    with open(vec_file) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                idx = vocab.get_token_index(word)
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index('@@PADDING@@')] = 0.
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading embeddings")
    return embeddings

def process_task(task, token_indexer, vocab):
    '''
    Convert a task's splits into AllenNLP fields then index the splits using vocab.
    Different tasks have different formats and fields, so process_task routes tasks
    to the corresponding processing based on the task type. These task specific processing
    functions should return three splits, which are lists (possibly empty) of AllenNLP instances.
    These instances are then indexed using the vocab
    '''
    for split_name in ['train', 'val', 'test']:
        split_text = getattr(task, '%s_data_text' % split_name)
        if isinstance(task, SingleClassificationTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=False)
        elif isinstance(task, PairClassificationTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=True)
        elif isinstance(task, PairRegressionTask):
            split = process_single_pair_task_split(split_text, token_indexer, is_pair=True,
                                                   classification=False)
        elif isinstance(task, SequenceGenerationTask):
            pass
        elif isinstance(task, RankingTask):
            pass
        else:
            raise ValueError("Preprocessing procedure not found for %s" % task.name)
        for instance in split:
            instance.index_fields(vocab)
        setattr(task, '%s_data' % split_name, split)
    return

def process_single_pair_task_split(split, indexers, is_pair=True, classification=True):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    if is_pair:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        inputs2 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[1]]
        if classification:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[2]]
        else:
            labels = [NumericField(l) for l in split[-1]]

        if len(split) == 4: # numbered test examples
            idxs = [LabelField(l, label_namespace="idxs", skip_indexing=True) for l in split[3]]
            instances = [Instance({"input1": input1, "input2": input2, "labels": label, "idx": idx}) \
                          for (input1, input2, label, idx) in zip(inputs1, inputs2, labels, idxs)]

        else:
            instances = [Instance({"input1": input1, "input2": input2, "labels": label}) for \
                            (input1, input2, label) in zip(inputs1, inputs2, labels)]

    else:
        inputs1 = [TextField(list(map(Token, sent)), token_indexers=indexers) for sent in split[0]]
        if classification:
            labels = [LabelField(l, label_namespace="labels", skip_indexing=True) for l in split[2]]
        else:
            labels = [NumericField(l) for l in split[2]]

        if len(split) == 4:
            idxs = [LabelField(l, label_namespace="idxs", skip_indexing=True) for l in split[3]]
            instances = [Instance({"input1": input1, "labels": label, "idx": idx}) for \
                         (input1, label, idx) in zip(inputs1, labels, idxs)]
        else:
            instances = [Instance({"input1": input1, "labels": label}) for (input1, label) in
                         zip(inputs1, labels)]
    return instances #DatasetReader(instances) #Batch(instances) #Dataset(instances)
