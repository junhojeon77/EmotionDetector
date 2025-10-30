import tensorflow as tf

# Dataset
train_dir = "data/train"
test_dir = "data/test"


# Prepare Data

img_size = (48, 48)
batch_size = 64

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=True,
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    label_mode="int",
    image_size=img_size,
    batch_size=batch_size,
    color_mode='grayscale',
    shuffle=False,
)

# Normalization
normalization = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization(x), y))
val_ds = val_ds.map(lambda x, y: (normalization(x), y))

