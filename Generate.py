from music21 import *
import numpy as np
import os
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.models import *
from tensorflow.python.keras.callbacks import *
import tensorflow.python.keras.backend as K
import random


def read_midi(file):
	"""
	a method to read a midi file into numpy array
	:param file: a path to a midi file
	:return: numpy array representing the midi file
	"""
	print("Loading Music File:", file)
	notes = []
	notes_to_parse = None
	# parsing a midi file
	midi = converter.parse(file)
	# grouping based on different instruments
	s2 = instrument.partitionByInstrument(midi)

	# Looping over all the instruments
	for part in s2.parts:

		# select elements of only piano
		if 'Piano' in str(part):

			notes_to_parse = part.recurse()

			# finding whether a particular element is note or a chord
			for element in notes_to_parse:

				# note
				if isinstance(element, note.Note):
					notes.append(str(element.pitch))

				# chord
				elif isinstance(element, chord.Chord):
					notes.append('.'.join(str(n) for n in element.normalOrder))

	return np.array(notes)


def load_files(path):
	"""
	a method to load all files into our project
	:param path: the path to the dir of the files
	:return: a numpy array, containing numpy arrays of midi files
	"""
	# reading file names
	files = [i for i in os.listdir(path) if i.endswith(".mid")]

	# reading each midi file
	return np.array([read_midi(path+i) for i in files])


def filter_music(notes_array, frequent_notes):
	"""
	a method to get only most frequent notes out of all the midi files
	:param notes_array: array containing notes from all midi files
	:param frequent_notes: the most frequent notes
	:return: a new array containing only the most frequent notes from the midi files
	"""
	new_music = []

	for notes in notes_array:
		temp = []
		for note in notes:
			if note in frequent_notes:
				temp.append(note)
		new_music.append(temp)
	return np.array(new_music)


def prepare_input_output(new_music, no_of_timesteps):
	"""
	prepare the input and labels for the training of the network
	:param new_music: array of arrays of notes
	:param no_of_timesteps: length of input to network
	:return: x, y (labels)
	"""
	x = []
	y = []
	for notes in new_music:
		for i in range(0, len(notes) - no_of_timesteps, 1):
			input = notes[i:i+no_of_timesteps]
			label = notes[i+no_of_timesteps]
			x.append(input)
			y.append(label)
	return np.asarray(x), np.asarray(y)


def create_note_int_dict(notes_list):
	unique = list(set(notes_list))
	return dict((note, number) for number, note in enumerate(unique)), unique


def create_x_seq(x, x_note_to_int):
	"""
	create array of numbers from an array of notes
	:param x:
	:param x_note_to_int:
	:return:
	"""
	x_seq = []
	for i in x:
		temp = []
		for j in i:
			temp.append(x_note_to_int[j])
		x_seq.append(temp)
	return x_seq


def waveNet(x_tr, y_tr, unique_x, unique_y, x_val, y_val):
	"""
	a method to define and train a waveNet network
	:param x_tr: training data
	:param y_tr: training labels
	:param unique_x: number of different notes in training data
	:param unique_y: number of different labels
	:param x_val: validation data
	:param y_val: validation labels
	:return: saves a trained model in the file 'best_model.h5'
	"""
	K.clear_session()
	model = Sequential()
	# embedding layer
	model.add(Embedding(len(unique_x), 100, input_length=32, trainable=True))
	model.add(Conv1D(64, 3, padding='causal', activation='relu'))
	model.add(Dropout(0.2))
	model.add(MaxPool1D(2))
	model.add(Conv1D(128, 3, activation='relu', dilation_rate=2, padding='causal'))
	model.add(Dropout(0.2))
	model.add(MaxPool1D(2))
	model.add(Conv1D(256, 3, activation='relu', dilation_rate=4, padding='causal'))
	model.add(Dropout(0.2))
	model.add(MaxPool1D(2))
	# model.add(Conv1D(256,5,activation='relu'))
	model.add(GlobalMaxPool1D())
	model.add(Dense(256, activation='relu'))
	model.add(Dense(len(unique_y), activation='softmax'))
	model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
	model.summary()

	mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True, verbose=1)
	history = model.fit(np.array(x_tr), np.array(y_tr), batch_size=128, epochs=50,
	                    validation_data=(np.array(x_val), np.array(y_val)), verbose=1, callbacks=[mc])

def compose(x_val, no_of_timesteps, model):
	index = np.random.randint(0, len(x_val)-1)
	random_notes = np.asarray(x_val[index])

	predictions = []
	for i in range(10):
		random_notes = random_notes.reshape(1, no_of_timesteps)
		prob = model.predict(random_notes)[0]
		y_pred = np.argmax(prob, axis=0)
		predictions.append(y_pred)
		random_notes = np.insert(random_notes[0], len(random_notes[0]), y_pred)
		random_notes = random_notes[1:]

	return predictions


def convert_to_midi(prediction_output):
	offset = 0
	output_notes = []

	# create note and chord objects based on the values generated by the model
	for pattern in prediction_output:

		# pattern is a chord
		pattern = str(pattern)
		if '.' in pattern or pattern.isdigit():
			notes_in_chord = pattern.split('.')
			notes = []
			for current_note in notes_in_chord:
				cn = int(current_note)
				new_note = note.Note(cn)
				new_note.storedInstrument = instrument.Piano()
				notes.append(new_note)

			new_chord = chord.Chord(notes)
			new_chord.offset = offset
			output_notes.append(new_chord)

		# pattern is a note
		else:

			new_note = note.Note(pattern)
			new_note.offset = offset
			new_note.storedInstrument = instrument.Piano()
			output_notes.append(new_note)

		# increase offset each iteration so that notes do not stack
		offset += 1
	midi_stream = stream.Stream(output_notes)
	midi_stream.write('midi', fp='music.mid')


if __name__ == '__main__':
	notes_array = np.load("notes_array.npy", allow_pickle=True)
	notes_ = [element for note_ in notes_array for element in note_]
	freq = dict(Counter(notes_))
	frequent_notes = [note_ for note_, count in freq.items() if count >= 50]
	x, y = prepare_input_output(filter_music(notes_array, frequent_notes), 32)
	x_note_to_int, unique_x = create_note_int_dict(x.ravel())
	y_note_to_int, unique_y = create_note_int_dict(y)
	x_seq = create_x_seq(x, x_note_to_int)
	y_seq = [y_note_to_int[i] for i in y]
	x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

	model = load_model('best_model.h5')

	predictions = compose(x_val, 32, model)

	convert_to_midi(predictions)














