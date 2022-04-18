import tensorflow as tf
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import cv2
import os, glob, shutil
from pathlib import Path
import random
import argparse


class BahdanauAttention(tf.keras.Model):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, features, hidden):
        # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

        # hidden shape == (batch_size, hidden_size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, 64, hidden_size)
        score = tf.nn.tanh(self.W1(features) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, 64, 1)
        # you get 1 at the last axis because you are applying score to self.V
        attention_weights = tf.nn.softmax(self.V(score), axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * features
        context_vector = tf.reduce_sum(context_vector, axis=1)

        return context_vector, attention_weights


class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x


class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(self.units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')
        self.fc1 = tf.keras.layers.Dense(self.units)
        self.fc2 = tf.keras.layers.Dense(vocab_size)

        self.attention = BahdanauAttention(self.units)

    def call(self, x, features, hidden):
        # defining attention as a separate model
        context_vector, attention_weights = self.attention(features, hidden)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # shape == (batch_size, max_length, hidden_size)
        x = self.fc1(output)

        # x shape == (batch_size * max_length, hidden_size)
        x = tf.reshape(x, (-1, x.shape[2]))

        # output shape == (batch_size * max_length, vocab)
        x = self.fc2(x)

        return x, state, attention_weights

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


def load_image(image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (299, 299))
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path


def evaluate_greedy(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

        attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        result.append(tokenizer.index_word[predicted_id])

        if tokenizer.index_word[predicted_id] == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def evaluate_beam_search(image, beam_index=10):
    start = [tokenizer.word_index['<start>']]

    result = [[start, 0.0]]

    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = image_features_extract_model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

    dec_input = tf.expand_dims([tokenizer.word_index['<start>']], 0)

    for i in range(max_length):
        temp = []
        for s in result:

            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = tf.reshape(attention_weights, (-1,)).numpy()
            # Getting the top <beam_index>(n) predictions
            word_preds = np.argsort(predictions[0])[-beam_index:]

            # creating a new list so as to put them via the model again
            for w in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(w)
                prob += predictions[0][w]
                temp.append([next_cap, prob])

        result = temp
        # Sorting according to the probabilities
        result = sorted(result, reverse=False, key=lambda l: l[1])
        # Getting the top words
        result = result[-beam_index:]

        predicted_id = result[-1]  # with Max Probability
        pred_list = predicted_id[0]

        prd_id = pred_list[-1]
        if (prd_id != 3):
            dec_input = tf.expand_dims([prd_id],
                                       0)  # Decoder input is the word predicted with highest probability among the top_k words predicted
        else:
            break

    result = result[-1][0]

    intermediate_caption = [tokenizer.index_word[i] for i in result]
    final_caption = []
    for i in intermediate_caption:
        if i != '<end>':
            final_caption.append(i)
        else:
            break

    attention_plot = attention_plot[:len(result)]
    return final_caption, attention_plot


def calc_max_length(tensor):
    return max(len(t) for t in tensor)


def cluster_frames(input_dir):
    glob_dir = input_dir + '/*.jpg'

    images = [cv2.resize(cv2.imread(file), (224, 224)) for file in glob.glob(glob_dir)]
    paths = [file for file in glob.glob(glob_dir)]
    images = np.array(np.float32(images).reshape(len(images), -1) / 255)

    model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    predictions = model.predict(images.reshape(-1, 224, 224, 3))
    pred_images = predictions.reshape(images.shape[0], -1)

    sil = []
    kl = []
    kmax = 2
    s_dict = {}

    for k in range(2, kmax + 1):
        kmeans2 = KMeans(n_clusters=k).fit(pred_images)
        labels = kmeans2.labels_
        score = silhouette_score(pred_images, labels, metric='euclidean')
        s_dict[score] = k

    max_k = max(s_dict)

    kmodel = KMeans(n_clusters=s_dict[max_k], random_state=728)
    kmodel.fit(pred_images)
    kpredictions = kmodel.predict(pred_images)
    if not os.path.exists('output'):
        os.mkdir('output')
    shutil.rmtree('output')
    for i in range(s_dict[max_k]):
        os.makedirs("output\cluster" + str(i))
    for i in range(len(paths)):
        shutil.copy2(paths[i], "output\cluster" + str(kpredictions[i]))
    shutil.rmtree('extracted_frames')


def delete_images(path, number_of_images, extension='jpg'):
    images = Path(path).glob(f'*.{extension}')
    for image in random.sample(images, number_of_images):
        image.unlink()

def drop_frames(path):
    x = [x[0] for x in os.walk(path)]
    for dir in x[1:]:
        files = os.listdir(dir)
        num_files_to_remove = len(files) - 1
        for file in files[:num_files_to_remove]:
            os.remove(dir + '\\' + file)
    print('Removed Frames')


def extract_frames(path):
    cam = cv2.VideoCapture(path)
    fps = cam.get(cv2.CAP_PROP_FPS)
    fps = round(fps)
    try:
        if not os.path.exists('extracted_frames/'):
            os.makedirs('extracted_frames')

    except OSError:
        print('Error: Creating directory of data')

    currentframe = 0

    while (True):
        ret, frame = cam.read()
        if ret:
            if currentframe % fps == 0:
                name = './extracted_frames/frame' + str(currentframe) + '.jpg'
                print('Creating...' + name)

                cv2.imwrite(name, frame)
            currentframe += 1
        else:
            break

    cam.release()


def get_captions(path):
  results = {}
  x = [x[0] for x in os.walk(path)]
  for dir in x[1:]:
      files = os.listdir(dir)
      for file in files:
        result, _ = evaluate_beam_search(dir + '\\' + file, beam_index=5)
        result.pop(0)
        if len(result) <= 10:
          results[dir + '\\' + file] = ' '.join(result)
  return results


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

image_features_extract_model = tf.keras.Model(new_input, hidden_layer)


top_k = 10000
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')


pickle_in = open("train_captions_1024.pkl","rb")
train_captions = pickle.load(pickle_in)
train_seqs = tokenizer.texts_to_sequences(train_captions)
tokenizer.fit_on_texts(train_captions)

tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

embedding_dim = 256
units = 1024
vocab_size = top_k + 1
attention_features_shape = 64

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()

max_length = calc_max_length(train_seqs)

checkpoint_path = "ckpt"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer = optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

ckpt.restore('ckpt/ckpt-10')

extract_frames('test.avi')
cluster_frames('extracted_frames')
drop_frames('output')

b = '5'

results = {}
capts_b = {}
capts_br = {}
res_b = []
res_br = []

x = [x[0] for x in os.walk('output')]
for dir in x[1:]:
    files = os.listdir(dir)
    for file in files:
        cd = dir + '\\' + file
        result_b, _ = evaluate_beam_search(cd, beam_index=int(b))
        result_br, _ = evaluate_greedy(cd)
        result_b.pop(0)
        # if len(result_b) <= 10 and len(result_br):
        capts_b[(cd).replace('\\', '/')] = ' '.join(result_b)
        capts_br[(cd).replace('\\', '/')] = ' '.join(result_br)
        res_b.append(' '.join(result_b))
        res_br.append(' '.join(result_br))

res_b[0] = res_b[0].capitalize()
res_br[0] = res_br[0].capitalize()

results['beam_search_' + b] = capts_b
results['beam_search_caption'] = ', '.join(res_b)

results['greedy'] = capts_br
results['greedy_caption'] = ', '.join(res_br)

sum_text = ""
for i in results['beam_search_5'].values():
    sum_text = sum_text + i + " "

print("summarized text is.... ",sum_text)








