import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.preprocessing import image
import numpy as np



train_dir_path = r'<train dir path>'
test_dir_path = r'<test dir path>'
batch_size = 30
new_prop = 100

validation_ratio  = 0.2

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(
validation_split = validation_ratio
)



train_generator = data_gen.flow_from_directory(
    train_dir_path,
    batch_size= batch_size,
    target_size = (new_prop, new_prop),
    shuffle=True,
    class_mode='categorical',
    subset='training')



eval_generator = data_gen.flow_from_directory(
    train_dir_path,
    batch_size= batch_size,
    target_size = (new_prop, new_prop),
    shuffle=True,
    class_mode='categorical',
    subset='validation')

test_generator = data_gen.flow_from_directory(
    test_dir_path,
    class_mode='categorical',
    target_size = (new_prop, new_prop),
    shuffle=False,
    batch_size=1)

tf.random.set_seed(100)

print(train_generator)


model = tf.keras.models.Sequential([layers.Flatten(input_shape=(3, new_prop, new_prop)),
layers.Dense(40, activation='relu'),
layers.Dense(20, activation='relu'),
layers.Dense(4, activation = 'softmax')])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(0.0001),
    loss=tf.keras.losses.CategoricalCrossentropy(),
    metrics = ['accuracy']
    )

history = model.fit(train_generator, epochs = 6, validation_data = eval_generator)

model.save('shape_predictor.h5')

#------------------------#

test_loss, test_acc = model.evaluate(test_generator)
print(test_acc*100,"%")

filenames = test_generator.filenames
predictions = model.predict(test_generator, steps=len(test_generator), verbose=1)
labels = ['circle','square','star','triangle']
match = 0
mismatch = 0

for i in range(len(predictions)):
    if labels[np.argmax(predictions[i])] in filenames[i]:
        match+=1
    else:
        mismatch+=1
        
print('match_rate is: ',match/len(predictions)*100,'%')
print(match)
print(mismatch)








