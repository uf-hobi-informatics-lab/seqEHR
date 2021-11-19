import sys

import numpy as np

from common_utils.utils import load_text, save_text

sys.path.append("../")


def code2index(codes, c2i):
    """
     the function input codes should be a list of list with all real code
     the first list is time steps; the second list is features
     the c2i is the code to index mappings
     function will return a list of list with all mapped indexes
    """
    return [[c2i[c] for c in temp] for temp in codes]


def load_embeddings(embedding_file):
    """
    function used to load pre-trained embeddings
    input file should be organized as word2vec/fasttext pre-trained embedding format (a txt file):
    each line is for a unique code
    each line start with the code followed by its embedding vectors
    we have pad at index 0 with all values set to 0
    we have unk at index -1 with all values random initialized ~N(0, 1)
    pad is for padding
    unk is for code that is not in the code vocab

    :param embedding_file: the pretrained embedding files
    :return: numpy embedding matrix, code2index, index2code
    """
    raw_embeddings = load_text(embedding_file).strip()
    lines = raw_embeddings.split("\n")

    emb_dim = -1
    code2index = dict()
    code2index['pad'] = 0
    code2index['unk'] = len(lines) + 1

    embeddings = []
    for idx, line in enumerate(lines):
        info = line.split(" ")
        tok = info[0]
        code2index[tok] = idx + 1
        vector = [float(each) for each in info[1:]]
        embeddings.append(vector)

        if idx == 0:
            emb_dim = len(vector)
        else:
            assert emb_dim == len(vector), \
                "expect embeddings have same dim but get {} and {}".format(emb_dim, len(vector))

    embeddings.insert(0, list(np.zeros(emb_dim)))
    np.random.seed(13)
    embeddings.append(list(np.random.normal(0, 1, size=emb_dim)))
    index2code = {v: k for k, v in code2index.items()}

    return embeddings, code2index, index2code


def random_generate_embeddings(vocab, emb_dim=50):
    """
    The function is used to create a random initialized embeddings based on a pre-defined vocab
    :param emb_dim: embedding dimension
    :param vocab: a list of medical codes (ICD or RXCUI)
    :return:
    """
    vocab = sorted(list(set(vocab)))

    code2index = dict()
    code2index['pad'] = 0
    code2index['unk'] = len(vocab) + 1

    embeddings = np.zeros(emb_dim).reshape(1, -1)

    for idx, code in enumerate(vocab):
        code2index[code] = idx + 1

    np.random.seed(2)
    embeddings = np.concatenate([embeddings, np.random.rand(len(vocab)+1, emb_dim)], axis=0)

    index2code = {v: k for k, v in code2index.items()}

    return embeddings, code2index, index2code


def main(vocab_file, dim, output_file):
    """
        in vocab file, each line should be a unique medical code
    """
    codes = load_text(vocab_file).strip().split("\n")
    embeddings, code2index, index2code = random_generate_embeddings(codes, emb_dim=dim)

    outputs = []
    for code, index in code2index.items():
        vector = embeddings[index]
        str_vec = " ".join([str(each) for each in vector])
        line = "{} {}".format(code, str_vec)
        outputs.append(line)

    outputs = "\n".join(outputs)
    save_text(outputs, output_file)


if __name__ == '__main__':
    import sys
    vocab_file, emb_dim, emb_file = sys.argv[1:]
    emb_dim = int(emb_dim)

    main(vocab_file, emb_dim, emb_file)