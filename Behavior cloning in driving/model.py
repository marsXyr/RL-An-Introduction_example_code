from lib.model_utils import *
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Flatten, Dense, Lambda, Convolution2D, Dropout, Cropping2D


class Model(ReadData):
	"""
		your Model 
	"""
	def __init__(self):

		ReadData.__init__(self)
		self.batch_size = 32

	def build_and_train_model(self):
		"""
			model : https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf
		"""
		# build your model and train
		model = Sequential()
		model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
		model.add(Cropping2D(cropping=((50, 20), (0, 0)), input_shape=(160, 320, 3)))
		model.add(Convolution2D(24, (5, 5), activation='relu', strides=(2, 2)))
		model.add(Convolution2D(36, (5, 5), activation='relu', strides=(2, 2)))
		model.add(Convolution2D(48, (5, 5), activation='relu', strides=(2, 2)))
		model.add(Convolution2D(64, (3, 3), activation='relu', strides=(1, 1)))
		model.add(Convolution2D(64, (3, 3), activation='relu', strides=(1, 1)))
		model.add(Flatten())
		model.add(Dense(100, activation='relu'))
		model.add(Dense(50, activation='relu'))
		model.add(Dense(10, activation='relu'))
		model.add(Dense(1))
		model.summary()

		model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001))

		model.fit_generator(self.train_data, steps_per_epoch=len(self.train_log), nb_epoch=16, max_q_size=1,
							validation_data=self.valid_data, validation_steps=len(self.valid_log))

		# Save model to h5
		model.save('model.h5')


def main():
	"""Main function of the model
	"""

	################################
	# change this
	log_dir = "./beta_simulator_mac/data2/driving_log.csv"

	my_model = Model()
	my_model.read_csv_data(log_dir)
	my_model.build_and_train_model()


if __name__ == "__main__":
	main()