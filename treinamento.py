import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam

# Diretórios
base_dir = "waldo_dataset"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Parâmetros
img_size = 224
batch_size = 32

# Geradores de imagem
train_gen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.2,
                                horizontal_flip=True, width_shift_range=0.1, height_shift_range=0.1)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(img_size, img_size),
                                           batch_size=batch_size, class_mode='binary')
val_data = val_gen.flow_from_directory(val_dir, target_size=(img_size, img_size),
                                       batch_size=batch_size, class_mode='binary')

# Modelo base
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_size, img_size, 3))
base_model.trainable = False

# Cabeça do modelo
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(64, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)

# Compilação
model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

# Treinamento
model.fit(train_data, validation_data=val_data, epochs=400)

# Salvando modelo
model.save("waldo_classifier3.h5")
