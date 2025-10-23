import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense
from keras.utils import to_categorical

# === Leer texto ===
text = open("input.txt", "r", encoding="utf-8").read()
print(f"Longitud del texto: {len(text)} caracteres")

# Crear vocabulario
chars = sorted(list(set(text)))
vocab_size = len(chars)
print(f"Vocabulario: {vocab_size} caracteres únicos")

char_to_idx = {ch: i for i, ch in enumerate(chars)}
idx_to_char = {i: ch for i, ch in enumerate(chars)}

# === Preprocesamiento ===
seq_length = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - seq_length, step):
    sentences.append(text[i: i + seq_length])
    next_chars.append(text[i + seq_length])

print(f"Cantidad de secuencias: {len(sentences)}")

# Convertir a índices y one-hot
X = np.zeros((len(sentences), seq_length, vocab_size), dtype=np.bool_)
y = np.zeros((len(sentences), vocab_size), dtype=np.bool_)
for i, sentence in enumerate(sentences):
    for t, ch in enumerate(sentence):
        X[i, t, char_to_idx[ch]] = 1
    y[i, char_to_idx[next_chars[i]]] = 1

# === Modelo ===
model = Sequential()
model.add(SimpleRNN(128, input_shape=(seq_length, vocab_size)))
model.add(Dense(vocab_size, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam")
model.summary()

# === Entrenar ===
model.fit(X, y, batch_size=128, epochs=20)

# === Función para generar texto ===
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(range(len(preds)), p=preds)

def generate_text(model, seed_text, length=400, temperature=0.5):
    generated = seed_text
    for _ in range(length):
        x_pred = np.zeros((1, seq_length, vocab_size))
        for t, ch in enumerate(seed_text):
            x_pred[0, t, char_to_idx[ch]] = 1
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = sample(preds, temperature)
        next_char = idx_to_char[next_index]
        generated += next_char
        seed_text = seed_text[1:] + next_char
    return generated

# === Generar texto ===
start_index = np.random.randint(0, len(text) - seq_length - 1)
seed = text[start_index:start_index + seq_length]
print("Texto inicial:")
print(seed)
print("\nTexto generado:")
print(generate_text(model, seed, length=400, temperature=0.5))


"""
Model: "sequential"
┌──────────────────────────────────────┬─────────────────────────────┬─────────────────┐
│ Layer (type)                         │ Output Shape                │         Param # │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ simple_rnn (SimpleRNN)               │ (None, 128)                 │          19,840 │
├──────────────────────────────────────┼─────────────────────────────┼─────────────────┤
│ dense (Dense)                        │ (None, 26)                  │           3,354 │
└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘
 Total params: 23,194 (90.60 KB)
 Trainable params: 23,194 (90.60 KB)
 Non-trainable params: 0 (0.00 B)
Epoch 1/20
2025-10-23 20:10:14.061667: E tensorflow/core/util/util.cc:131] oneDNN supports DT_BOOL only on platforms with AVX-512. Falling back to the default Eigen-based implementation if present.
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m1s←[0m 1s/step - loss: 3.3834
Epoch 2/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 3.2172
Epoch 3/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 3.0678
Epoch 4/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 2.9390
Epoch 5/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 2.8343
Epoch 6/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 2.7324
Epoch 7/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 2.6228
Epoch 8/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 2.5208
Epoch 9/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 2.4285
Epoch 10/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 2.3371
Epoch 11/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 2.2439
Epoch 12/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 2.1479
Epoch 13/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 2.0519
Epoch 14/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 1.9599
Epoch 15/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 1.8718
Epoch 16/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 1.7899
Epoch 17/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 1.6976
Epoch 18/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 1.6099
Epoch 19/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 47ms/step - loss: 1.5310
Epoch 20/20
←[1m1/1←[0m ←[32m━━━━━━━━━━━━━━━━━━━━←[0m←[37m←[0m ←[1m0s←[0m 31ms/step - loss: 1.4505
Texto inicial:
little known the feelings or views of su

Texto generado:
little known the feelings or views of suhe w rrno shftoossisohn  oheoibs hrkchhchoh othefgiaoinhevios onxsyho tbbcsvhc goesshh  h u giu sshc hphrhooo no gidtsoheishhsh n nkeaiidhHhcohhh dho rnrsohsrioh sddoohh nnnstih onihwhw, ohnsodoordpheocrsohcssoheedg sheasgerrohrasinohiiie oideoo hmtbtdlsnihoidhro h dsnar rnrniodmrdoheotno bv ylbhuocthhh  siehkorrsohd sshrooo gnhea, ,soo hcthoHstopoo rsohyo tooerdognhtoeohsoeytoH Hahrin,rronosHssoh
"""