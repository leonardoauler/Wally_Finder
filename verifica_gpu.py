#Script utilizado para testar a detecção de GPU com o TensorFlow para treinamentos

import tensorflow as tf

print("TensorFlow versão:", tf.__version__)
print("Dispositivos físicos encontrados:")
for device in tf.config.list_physical_devices():
    print(device)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("\n✅ GPU detectada!")
else:
    print("\n❌ Nenhuma GPU detectada.")
