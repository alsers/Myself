import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from MY_UTIL import *
## %matplotlib inline

from winsound import PlaySound
PlaySound("./raw_data/activates/1.wav", flags=1)
PlaySound("./raw_data/negatives/4.wav", flags=1)
PlaySound("./raw_data/backgrounds/1.wav", flags=1)
PlaySound("audio_examples/example_train.wav", flags=1)

x = graph_spectrogram("audio_examples/example_train.wav")
# x.shape : (101, 5511)

_, data = wavfile.read("audio_examples/example_train.wav")

Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram


'''
audio sampled at 44100 Hz (or 44100 Hertz). 
    * This means the microphone gives us 44,100 numbers per second. 
    * Thus, a 10 second audio clip is represented by 441,000
    * Raw audio divides 10 seconds into 441,000 units.
    * A spectrogram divides 10 seconds into 5,511 units.
* You will use a Python module `pydub` to synthesize audio, and it divides 10 seconds into 10,000 units.
* The output of our model will divide 10 seconds into 1,375 units.
    * $T_y = 1375$
    * For each of the 1375 time steps, the model predicts whether someone recently finished saying the trigger word "activate". 
* All of these are hyperparameters and can be changed (except the 441000, which is a function of the microphone). 
'''
Ty = 1375
#  load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio('./raw_data/')


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.
    
    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")
    
    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """
    
    segment_start = np.random.randint(low=0, high=10000-segment_ms)   # Make sure segment doesn't run past the 10sec background 
    segment_end = segment_start + segment_ms - 1
    
    return (segment_start, segment_end)


#  check overlapping
def is_overlapping(segment_time, previous_segments):
    """
    check if the time of a segment overlaps with the times of existing segments.

    Arguments:
    segment time -- a tuple of (segment_start, segment_end) for the new segment
    previous_segments -- a list of tuples of (segments_start, segment_end) for the existing segments.

    Returns:
    True if the time segment overlaps with any of the existing segments, False otherwise. 
    """
    segment_start, segment_end = segment_time
    overlap = False
    for previous_start, previous_end in previous_segments:
        if previous_start <= segment_end and segment_start <= previous_end:  # !!! logic
            overlap = True
            break
    return overlap


def insert_audio_clip(background, audio_clip, previous_segments):
    """
    insert a new audio segment over the background noise at random time step,
    ensuring audio segments dose not overlap with existing segments.
    
    Arguments:
    background -- a 10s background audio recording.
    audio_clip -- the audio clip to be inserted/overlaid.
    previous_segments -- time where audio segments have already been placed.

    Returns:
    new_background -- the updated background audio
    """
    segment_ms = len(audio_clip)
    segment_time = get_random_time_segment(segment_ms)

    retry = 5
    while is_overlapping(segment_time, previous_segments) and retry >= 0:
        segment_time = get_random_time_segment(segment_ms)
        retry -= 1
    if not is_overlapping(segment_time, previous_segments):
        previous_segments.append(segment_time)
        new_background = background.overlay(audio_clip, position = segment_time[0])  # insert
    else:
        new_background = background
        segment_time = (10000, 10000)
    return new_background, segment_time 


## Implement code to update the labels y_t(assuming you just inserted an activate audio clip)
## y is a (1, 1375) dimensional vector, since T_y = 1375
def insert_ones(y, segment_end_ms):
    '''
    Update the label vector y. The labels of the 50 output steps strictly after the end of the segment 
    should be set to 1. By strictly we mean that the label of segment_end_y should be 0 while, the
    50 following labels should be ones.

    Arguments:
    y -- numpy array of shape (1, Ty), the labels of the training example
    segment_end_ms -- the end time of the segment in ms
    
    Returns:
    y -- updated labels    
    '''
    _, Ty = y.shape
    segment_end_y = int(segment_end_ms * Ty / 10000.0)
    
    if segment_end_y < Ty :
        for i in range(segment_end_y + 1, segment_end_y + 51):
            if i < Ty - 1:
                y[0, i] = 1
    return y

arr1 = insert_ones(np.zeros((1, Ty)), 9700)
plt.plot(insert_ones(arr1, 4251)[0, :])
print('sanity checks', arr1[0][1333], arr1[0][634], arr1[0][635])


def create_training_example(background, actrivates, negatives, Ty):
    """
    Creates a training example with a given background, activates, and negatives.
    
    Arguments:
    background -- a 10 second background audio recording
    activates -- a list of audio segments of the word "activate"
    negatives -- a list of audio segments of random words that are not "activate"
    Ty -- The number of time steps in the output

    Returns:
    x -- the spectrogram of the training example
    y -- the label at each time step of the spectrogram
    """

    background = background - 20  ## make background quieter
    y = np.zeros((1, Ty))
    previous_segments = []
    number_of_activates = np.random.randint(0, 5)
    random_indices = np.random.randint(len(activates), size=number_of_activates)
    random_activates = [activates[i] for i in random_indices]  ## select a random number of random 'activate' in |activates| and insert into background

    for random_activate in random_activates:
        background, segment_time = insert_audio_clip(background, random_activate, previous_segments)
        segment_start, segment_end = segment_time
        y = insert_ones(y, segment_end) 

    number_of_negatives = np.random.randint(0, 3)
    random_indices = np.random.randint(len(negatives), size=number_of_negatives)
    random_negatives = [negatives[i] for i in random_indices]

    for random_negative in random_negatives:
        background, _ = insert_audio_clip(background, random_negative, previous_segments)

    background = match_target_amplitude(background, -20.0)   
    ## ⬆⬆⬆ have already decreased the background volume, now restore the inserted background volume.
    file_handle = background.export("train" + ".wav", format='wav')
    x = graph_spectrogram("train.wav")
    return x, y
    ## ⬆⬆⬆⬆⬆⬆ list比较特殊，在函数内虽然没有返回，但是append了新的元素仍会保留在函数结束之后

# Set the random seed
np.random.seed(18)
x, y = create_training_example(backgrounds[0], activates, negatives, Ty)
PlaySound("train.wav", flags=1)

plt.plot(y[0])

np.random.seed(4543)
nsamples = 10
X = []
Y = []
for i in range(0, nsamples):
    if i%10 == 0:
        print(i)
    x, y = create_training_example(backgrounds[i % 2], activates, negatives, Ty)
    X.append(x.swapaxes(0,1))
    Y.append(y.swapaxes(0,1))
X = np.array(X)  ## Tensor
Y = np.array(Y)

# Save the data for further uses
# np.save(f'./XY_train/X.npy', X)
# np.save(f'./XY_train/Y.npy', Y)
# Load the preprocessed training examples
# X = np.load("./XY_train/X.npy")
# Y = np.load("./XY_train/Y.npy")

X_dev = np.load("./XY_dev/X_dev.npy")
Y_dev = np.load("./XY_dev/Y_dev.npy")


''' --- Model --- '''
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Model, load_model, Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Input, Masking, TimeDistributed, LSTM, Conv1D
from tensorflow.keras.layers import GRU, Bidirectional, BatchNormalization, Reshape
from tensorflow.keras.optimizers import Adam

def modelf(input_shape):
    '''
    Function creating the model's graph in Keras.
    
    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    '''
    X_input = Input(shape=input_shape)
    
    X = Conv1D(196, 15, 4)(X_input)
    X = Activation('relu')(X)
    X = Dropout(0.8)(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)

    X = GRU(128, return_sequences=True)(X)
    X = Dropout(0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(0.8)(X)

    X = TimeDistributed(Dense(1, activation='sigmoid'))(X)
    '''
    TimeDistributed 通过将相同的 Dense 层（相同的权重）应用于 LSTM 输出一次一个时间步来实现这一技巧。
    这样，输出层只需要一个连接到每个 LSTM 单元（加上一个偏置）。
    The TimeDistributed achieves this trick by applying the same Dense layer (same weights) to the LSTMs 
    outputs for one time step at a time. In this way, the output layer only needs one connection to each LSTM unit (plus one bias).
    '''
    model = Model(inputs=X_input, outputs = X)
    return model

model = modelf(input_shape=(Tx, n_freq))
model.summary()

from tensorflow.keras.models import model_from_json
json_file = open('./models/model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('./models/model.h5')

model.layers[2].trainable = False
model.layers[7].trainable = False
model.layers[10].trainable = False

opt = Adam(lr=1e-6, beta_1=0.9, beta_2=0.999)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])

model.fit(X, Y, batch_size=16, epochs=1)



def detect_triggerword(filename):
    plt.subplot(2, 1, 1)
    audio_clip = AudioSegment.from_wav(filename)
    audio_clip = match_target_amplitude(audio_clip, -20.0)
    file_handle = audio_clip.export("tmp.wav", format="wav")
    filename = "tmp.wav"

    x = graph_spectrogram(filename)
    # the spectrogram outputs (freqs, Tx) and we want (Tx, freqs) to input into the model
    x  = x.swapaxes(0,1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    
    plt.subplot(2, 1, 2)
    plt.plot(predictions[0,:,0])
    plt.ylabel('probability')
    plt.show()
    return predictions

chime_file = 'audio_examples/chime.wav'
def chime_on_activate(filename, predictions, threshold):  # smart 'for and if'
    audio_clip = AudioSegment.from_wav(filename)
    chime = AudioSegment.from_wav(chime_file)
    Ty = predictions.shape[1]
    # Step 1: Initialize the number of consecutive output steps to 0
    consecutive_timesteps = 0
    # Step 2: Loop over the output steps in the y
    for i in range(Ty):
        # Step 3: Increment consecutive output steps
        consecutive_timesteps += 1
        # Step 4: If prediction is higher than the threshold and more than 20 consecutive output steps have passed
        if consecutive_timesteps > 20:  ## 25或者30可能更好?
            # Step 5: Superpose audio and background using pydub
            audio_clip = audio_clip.overlay(chime, position = ((i / Ty) * audio_clip.duration_seconds) * 1000)
            # Step 6: Reset consecutive output steps to 0
            consecutive_timesteps = 0
        # if amplitude is smaller than the threshold reset the consecutive_timesteps counter
        if predictions[0, i, 0] < threshold:
            consecutive_timesteps = 0
        
    audio_clip.export("chime_output.wav", format='wav')

filename = "./raw_data/dev/1.wav"
prediction = detect_triggerword(filename)
chime_on_activate(filename, prediction, 0.5)
PlaySound("./chime_output.wav", flags=1)