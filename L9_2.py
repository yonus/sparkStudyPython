import keras;
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense , Dropout
from keras.optimizers  import RMSprop

(mnist_train_images ,  mnist_train_labels) , (mnist_test_images , mnist_test_labels) = mnist.load_data()

train_images = mnist_train_images.reshape(60000,784).astype('float32')
test_images = mnist_test_images.reshape(10000,784).astype('float32')

train_images = train_images/255;
test_images = test_images/255;

train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
test_labels  = keras.utils.to_categorical(mnist_test_labels,10)


model = Sequential()
model.add(Dense(512 ,activation='relu' ,input_shape=(784,)))
model.add(Dense(10,activation="softmax"))
model.compile(loss="categorical_crossentropy" , optimizer=RMSprop() , metrics=['accuracy'])
history  = model.fit(train_images , train_labels ,batch_size=100 , epochs = 10 ,verbose = 2)

model.summary()