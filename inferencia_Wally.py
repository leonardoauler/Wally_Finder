import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import seaborn as sns

#
# CONFIGURAÇÕES
#
img_path = "original-images/1.jpg"
model_path = "waldo_classifier3.h5"

input_size = 224
step_size = 12
threshold = 0.9

#
# CARREGAMENTO DO MODELO
#
model = load_model(model_path, compile=False)

#
# CARREGAMENTO DA IMAGEM
#
image = cv2.imread(img_path)
orig_image = image.copy()
img_height, img_width, _ = image.shape

heatmap = np.zeros((img_height, img_width))

found_windows = []

# Sliding window
for y in range(0, img_height - input_size + 1, step_size):
    for x in range(0, img_width - input_size + 1, step_size):
        window = image[y:y + input_size, x:x + input_size]
        window_resized = window.astype("float32") / 255.0
        window_resized = np.expand_dims(window_resized, axis=0)

        prediction = model.predict(window_resized, verbose=0)[0][0]

        heatmap[y:y + input_size, x:x + input_size] += prediction

        if prediction >= threshold:
            found_windows.append((x, y, prediction))

heatmap = heatmap / heatmap.max()

for (x, y, confidence) in found_windows:
    cv2.rectangle(
        orig_image,
        (x, y),
        (x + input_size, y + input_size),
        (0, 255, 0), 3
    )
    cv2.putText(
        orig_image,
        f"Wally: {confidence:.2f}",
        (x + 5, y + 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2
    )

plt.figure(figsize=(16, 16))

plt.imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))

# Overlay do heatmap
sns.heatmap(
    heatmap,
    cmap="jet",
    alpha=0.5,
    cbar=True,
    xticklabels=False,
    yticklabels=False
)

plt.title("Localização do Wally com Detecção e Heatmap", fontsize=18)
plt.axis('off')
plt.show()

#
# SALVA RESULTADO
#
cv2.imwrite("resultado_wally_com_heatmap.png", orig_image)
print("Imagem salva como resultado_wally_com_heatmap.png")
