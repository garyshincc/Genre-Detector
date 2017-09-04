import os
import re
import librosa
import sys
import numpy as np
import math
import pyaudio

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Activation
from keras.optimizers import Adam, RMSprop, SGD

from keras.models import model_from_json




class GenreFeatureData:
	def __init__(self):
		self.hop_length = 512

		self.genre_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop']
		self.dir_test_folder = "genres/_test"
		self.dir_validation_folder = "genres/_validation"
		self.dir_train_folder = "genres/_train"

		self.test_x_preprocessed_data = 'data_test_input.npy'
		self.test_y_preprocessed_data = 'data_test_target.npy'

		self.validation_x_preprocessed_data = 'data_validation_input.npy'
		self.validation_y_preprocessed_data = 'data_validation_target.npy'

		self.train_x_preprocessed_data = 'data_train_input.npy'
		self.train_y_preprocessed_data = 'data_train_target.npy'

		self.test_X = None
		self.test_Y = None
		self.validation_X = None
		self.validation_Y = None
		self.train_X = None
		self.train_Y = None

	def load_data(self):
		self.test_files_list = path_to_audio_files(self.dir_test_folder)
		self.validation_files_list = path_to_audio_files(self.dir_validation_folder)
		self.train_files_list = path_to_audio_files(self.dir_train_folder)

		'''
		save the numpy array files so we can use it later
		'''
		self.test_X, self.test_Y = self.extract_audio_features(self.test_files_list, hop_length=self.hop_length)
		with open(self.test_x_preprocessed_data, 'wb') as open_file:
			np.save(open_file, self.test_X)
			print ("saved test set inputs")
		with open(self.test_y_preprocessed_data, 'wb') as open_file:
			np.save(open_file, self.test_Y)
			print ("saved test set targets")

		self.validation_X, self.validation_Y = self.extract_audio_features(self.validation_files_list, hop_length=self.hop_length)
		with open(self.validation_x_preprocessed_data, 'wb') as open_file:
			np.save(open_file, self.validation_X)
			print ("saved validation inputs")
		with open(self.validation_y_preprocessed_data, 'wb') as open_file:
			np.save(open_file, self.validation_Y)
			print ("saved validation targets")

		# Training set
		self.train_X, self.train_Y = self.extract_audio_features(self.train_files_list, hop_length=self.hop_length)
		with open(self.train_x_preprocessed_data, 'wb') as open_file:
			np.save(open_file, self.train_X)
			print ("saved train input")
		with open(self.train_y_preprocessed_data, 'wb') as open_file:
			np.save(open_file, self.train_Y)
			print ("saved train targets")


	def load_preprocessed_data(self):

		self.test_X = np.load(self.test_x_preprocessed_data)
		self.test_Y = np.load(self.test_y_preprocessed_data)
		self.test_Y = self.one_hot(self.test_Y)

		self.validation_X = np.load(self.validation_x_preprocessed_data)
		self.validation_Y = np.load(self.validation_y_preprocessed_data)
		self.validation_Y = self.one_hot(self.validation_Y)

		self.train_X = np.load(self.train_x_preprocessed_data)
		self.train_Y = np.load(self.train_y_preprocessed_data)
		self.train_Y = self.one_hot(self.train_Y)


	def extract_audio_features(self, list_of_audiofiles, hop_length=512):
		timeseries_length_list = []

		''' 
		For Nanoleaf Rhythm, there are 32 fft bins,
		and sampling rate is 11025.
		'''
		default_sample_rate = 11025
		default_num_fft_bins = 32

		for file_name in list_of_audiofiles:
			print ("Loading: " + str(file_name))
			y, sample_rate = librosa.load(file_name, sr=default_sample_rate)
			time_series = math.ceil(len(y) / hop_length)
			timeseries_length_list.append(time_series)

		timeseries_length = int(min(timeseries_length_list) / 2)
		data = np.zeros((len(list_of_audiofiles), timeseries_length, 27), dtype="float64")
		print ("data shape: " + str(data.shape))
		target = []

		for i, file_name in enumerate(list_of_audiofiles):
			print ("Processing: " + str(file_name))
			y, sample_rate = librosa.load(file_name, sr=default_sample_rate)

			mel_freq_cep_coeff = librosa.feature.mfcc(y=y, sr=default_sample_rate, hop_length=hop_length, n_mfcc=13)
			spectral_center = librosa.feature.spectral_centroid(y=y, sr=default_sample_rate, hop_length=hop_length)
			chroma = librosa.feature.chroma_stft(y=y, sr=default_sample_rate, hop_length=hop_length)
			spectral_roll = librosa.feature.spectral_rolloff(y=y, sr=default_sample_rate, hop_length=hop_length)

			splits = file_name.split("/")
			genre = splits[2].split(".")[0]
			target.append(genre)

			data[i, :, 0:13] = mel_freq_cep_coeff.T[0:timeseries_length, :]
			data[i, :, 13:14] = spectral_center.T[0:timeseries_length, :]
			data[i, :, 14:26] = chroma.T[0:timeseries_length, :]
			data[i, :, 26:27] = spectral_roll.T[0:timeseries_length, :]

		target_nps = np.expand_dims(np.asarray(target), axis = 1)
		print (target_nps)
		return data, target_nps

	def one_hot(self, genre_strings):
		one_hot_y = np.zeros((genre_strings.shape[0], len(self.genre_list)))
		for i, genre_string in enumerate(genre_strings):
			index = self.genre_list.index(genre_string[0])
			one_hot_y[i, index] = 1
		return one_hot_y

def path_to_audio_files(dir_folder):
	list_of_music = []
	for file_name in os.listdir(dir_folder):
		if file_name.endswith(".au"):
			directory = "%s/%s" % (dir_folder, file_name)
			list_of_music.append(directory)

	return list_of_music

def extract_song_features(song_name, hop_length=512):
	print ("Extracting: " + str(song_name))
	y, sample_rate = librosa.load(song_name, duration=10, sr=11025)
	timeseries_length = int(math.ceil(len(y) / hop_length))
	data = np.zeros((1, timeseries_length, 27), dtype="float64")
	print ("data shape: " + str(data.shape))
	'''
	mel frequency ceptrum coefficient calculation
	mfcc is basically power spectrum from fft, converted to mel scale
	from frequency, convert to mel scale
	mf = 2595 * log_10 (1 + (freq / 700))
	from mel freq, discrete cosine transform
	mfcc = summation (from k-1 to k) {log(freq_k) * cos[m(k - 0.5) * pi/k]}
	where m = 0, 1, ... k-1
	'''
	mel_freq_cep_coeff = librosa.feature.mfcc(y=y, sr=sample_rate, hop_length=hop_length, n_mfcc=13)

	'''
	spectral centroid calculation
	spectral centroid is the "brighness" of the sound
	spectral_centroid = summation (from k=1 to N){k * freq_k} / summation (from k=1 to N){freq_k}
	as you expect, sc is one value that corresponds to how "high"
	the frequency
	'''
	spectral_center = librosa.feature.spectral_centroid(y=y, sr=sample_rate, hop_length=hop_length)

	'''
	chroma_stft = chromagram
	'normalized chromagram'
	1. estimate (get) tuning (define bands)
	2. use default number of chromas = 12
	3. get spectrogram
	4. define filter bank
	5. compute raw chroma
	6. normalize raw chroma
	'''
	chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate, hop_length=hop_length)
	
	'''
	spectral rolloff frequency
	total_energy = cumulative sum
	use default roll_percent = 0.85
	threshold = roll_percent * last element in cumulative sum
	index = np.where(total_energy < threshold, np.nan, 1)
	return np.nanmin(ind * freq, axis-9, keepdims=True)
	'''
	spectral_roll = librosa.feature.spectral_rolloff(y=y, sr=sample_rate, hop_length=hop_length)

	data[0, :, 0:13] = mel_freq_cep_coeff.T[0:timeseries_length, :]
	data[0, :, 13:14] = spectral_center.T[0:timeseries_length, :]
	data[0, :, 14:26] = chroma.T[0:timeseries_length, :]
	data[0, :, 26:27] = spectral_roll.T[0:timeseries_length, :]
	return data

def get_rnn_dense_model():
	if os.path.isfile("model.json"):
		print ("I've found a model in your directory.")
		response = raw_input("Should I attempt to load it? (y/n) ")
		try:
			json_file = open("model.json", "r")
			json_model = json_file.read()
			json_file.close()
			model = model_from_json(json_model)
			print ("Loaded model")
			model.load_weights("model.h5")
			print ("Loaded weights")
		except:
			print ("Error: " + sys,exc_info()[0])
	else:
		print ("Creating new model")
		print ("_" * 20 + " Model Information " + "_" * 20)
		print ("Keras & Tensorflow backend Recurrent Neural Network")
		print ("2 Long Short-Term Memory layer, with 1 Dense Layer")
		print ("Optimizer: Adam Optimizer")
		print ("Learning Rate: 0.01")
		print ("Batch size: 128")
		print ("Num Epochs: 100")
		print ("_" * 58)

		genre_features = GenreFeatureData()
		#genre_features.load_data()
		genre_features.load_preprocessed_data()
		print ("Done loading")
		optimizer = Adam(lr=0.01)

		batch_size = 128
		nb_epochs = 100

		print ("Creating Model")
		# we are going to be using RNN sequential models
		model = Sequential()

		# Long term short memory
		# Fraction of the units to drop for the linear transformation of the recurrent state.
		# Fraction of the units to drop for the linear transformation of the inputs.
		print ("train_x shape: " + str(genre_features.train_X.shape[2]))
		model.add(
			LSTM(
				input_dim=genre_features.validation_X.shape[2],
				output_dim=128,
				activation='sigmoid',
				dropout_U=0.05,
				dropout_W=0.05,
				return_sequences=True
			)
		)
		model.add(
			LSTM(
				output_dim=64,
				activation='sigmoid',
				dropout_U=0.05,
				dropout_W=0.05,
				return_sequences=False
			)
		)
		model.add(
			Dense(
				units=genre_features.train_Y.shape[1],
				activation="softmax"
			)
		)
		print ("train_y shape:" + str(genre_features.train_Y.shape[1]))


		print ("Compiling the model")
		model.compile(
			loss='categorical_crossentropy',
			optimizer=optimizer,
			metrics=['accuracy']
		)
		model.summary()


		print ("Fitting the model with train data")
		model.fit(
			genre_features.train_X,
			genre_features.train_Y,
			batch_size=batch_size,
			epochs=nb_epochs,
			verbose=2
		)

		print ("Training complete.")
		response = raw_input("Would you like to save the trained model? (y/n) ")

		if response == "y":
			model_json = model.to_json()
			with open("model.json", "w") as open_file:
				open_file.write(model_json)
			print ("Model saved!")
			model.save_weights("model.h5")
			print ("Weights saved!")

		score, accuracy = model.evaluate(
			genre_features.validation_X,
			genre_features.validation_Y,
			batch_size=batch_size,
			verbose=2
		)
		print ("Validation loss: ", score)
		print ("Validation accuracy: ", accuracy)

		score, accuracy = model.evaluate(
			genre_features.test_X,
			genre_features.test_Y,
			batch_size=batch_size,
			verbose=2
		)
		print ("Test loss:  ", score)
		print ("Test accuracy:  ", accuracy)

	return model

def start_listening():
	pass

def process_wave_and_guess(next_song):
	genre_map = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop']
	try:
		song_features = extract_song_features(next_song)
		prediction = model.predict(song_features, batch_size=1, verbose=1)
		prediction = prediction.tolist()[0]
		first_guess = 0
		first_index = 0
		second_guess = 0
		second_index = 0
		for index, value in enumerate(prediction):
			if value > first_guess:
				second_guess = first_guess
				second_index = first_index
				first_guess = value
				first_index = index
			elif value > second_guess:
				second_guess = value
				second_index = index
		print ("I think i heard: " + str(genre_map[first_index]))
		print ("but it also may be: " + str(genre_map[second_index]))
	except IOError:
		print ("No such song found...")

if __name__ == "__main__":
	model = get_rnn_dense_model()
	print ("Ready to input new songs...")

	while (True):
		next_song = raw_input("Next song: ")
		if next_song == "q" or next_song == "quit":
			break
		if next_song == "mic":
			start_listening()
		else:
			process_wave_and_guess(next_song)



























