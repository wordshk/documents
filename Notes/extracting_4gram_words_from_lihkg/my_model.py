# model based off https://towardsdatascience.com/going-beyond-99-mnist-handwritten-digits-recognition-cfff96337392
# Code adopted from https://github.com/tinygrad/tinygrad/blob/641f347232101f5df13a37a3a1b155b2eea7beb5/examples/beautiful_mnist.py
# Copyright Notice: https://github.com/tinygrad/tinygrad/blob/641f347232101f5df13a37a3a1b155b2eea7beb5/LICENSE

from tinygrad import Tensor, nn, GlobalCounters
from tqdm import trange
import os, sys
import numpy as np
from tinygrad.nn.state import safe_save, safe_load, get_state_dict, load_state_dict

def read_json_into_np(file):
    """Convert a JSON (line) file into a numpy array"""
    import json
    with open(file) as f:
        data = []
        entities = []
        for line in f:
            data.append(json.loads(line)["data"])
            entities.append(json.loads(line)["entity"])
        return (np.array(data), entities)

def read_json_into_np_generator(file, batch_size=512):
    """Convert a JSON (line) file into a numpy array"""
    import json
    with open(file) as f:
        data = []
        entities = []
        for line in f:
            data.append(json.loads(line)["data"])
            entities.append(json.loads(line)["entity"])
            if len(data) >= batch_size:
                yield (np.array(data), entities)
                data = []
                entities = []
        yield (np.array(data), entities)


class Model:
    def __init__(self):
        self.layers = [
            nn.Linear(14, 50),
            nn.Linear(50, 50),
        ]

    def __call__(self, x:Tensor) -> Tensor: return x.sequential(self.layers)

def train():
    # X_train, Y_train, X_test, Y_test = fetch_mnist(tensors=True)
    # print(X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)
    # print(Y_train.data)
    X_train_known = read_json_into_np("known_words.jsonl")[0]
    X_train_unknown = read_json_into_np("not_known_as_words_picked.jsonl")[0]

    # Concatenate the two datasets
    X_train = np.concatenate((X_train_known, X_train_unknown))
    Y_train = np.array([1.0]*X_train_known.shape[0] + [0.0]*X_train_unknown.shape[0])

    X_train = Tensor(X_train.astype(np.float32))
    Y_train = Tensor(Y_train.astype(np.float32))

    X_test = X_train
    Y_test = Y_train

    model = Model()
    opt = nn.optim.Adam(nn.state.get_parameters(model), lr=0.008)


    def get_test_acc() -> Tensor: return (model(X_test).argmax(axis=1) == Y_test).mean()*100

    test_acc = float('nan')
    for i in (t:=trange(X_train.shape[0] // 512 + 3)):
        GlobalCounters.reset()     # NOTE: this makes it nice for DEBUG=2 timing
        with Tensor.train():
            opt.zero_grad()
            samples = Tensor.randint(512, high=X_train.shape[0])
            # TODO: this "gather" of samples is very slow. will be under 5s when this is fixed
            # loss = model(X_train[samples]).binary_crossentropy_logits(Y_train[samples]).backward()
            # loss = model(X_train[samples]).binary_crossentropy_logits(Y_train[samples]).backward()
            loss = model(X_train[samples]).sparse_categorical_crossentropy(Y_train[samples]).backward()
            opt.step()
        test_acc = get_test_acc().item()
        t.set_description(f"loss: {loss.item():6.2f} test_accuracy: {test_acc:5.2f}%")

    safe_save(get_state_dict(model), "model.safetensors")

def run():
    model = Model()
    state_dict = safe_load("model.safetensors")
    load_state_dict(model, state_dict)
    # Check whether stdout is a terminal
    if os.isatty(sys.stdin.fileno()) and os.isatty(sys.stdout.fileno()):
        import generate_training_data as gtd
        moe = gtd.read_moe_dict()
        wordshk = gtd.read_words_hk()
        gram_counts = gtd.read_subgrams()
        convert = gtd.wordshk_convert()
        convert_word = lambda word: "".join([convert.get(c, c) for c in word])
        is_known = lambda word: word in moe or convert_word(word) in wordshk

        while user_input := input("Enter a 4-gram: "):
            data = gtd.data_for_4gram(user_input, gram_counts, is_known)
            print(model(Tensor([data,])).argmax(axis=1).item())

    else:
        for inputs, item_labels in read_json_into_np_generator("not_known_as_words.jsonl"):
            X_run = Tensor(inputs.astype(np.float32), requires_grad=False)
            data = model(X_run).argmax(axis=1).data()
            for idx, entry in enumerate(data):
                print(item_labels[idx], entry)

if __name__ == "__main__":
    if not os.path.exists("model.safetensors"):
        train()
    run()
