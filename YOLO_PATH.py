import onnxruntime as ort
import cv2
import numpy as np

# Cargar el modelo ONNX
session = ort.InferenceSession(r"runs\YOLOv8_train2\weights\best.onnx")

# Cargar la imagen de prueba
image_path = r"YOLO_DataSet\images\test\01_4885_filename0570.jpg"
image = cv2.imread(image_path)
input_image = cv2.resize(image, (512, 512))  # Ajusta al tamaño utilizado en el entrenamiento

# Preprocesar la imagen
input_image = input_image.astype('float32') / 255.0  # Normalizar a [0,1]
input_image = np.transpose(input_image, (2, 0, 1))  # Cambiar a (C, H, W)
input_image = np.expand_dims(input_image, axis=0)  # Agregar la dimensión de batch

# Realizar la inferencia
inputs = {session.get_inputs()[0].name: input_image}
outputs = session.run(None, inputs)

# Interpretar y reorganizar los datos de salida
output_data = outputs[0][0]  # Seleccionar el batch 0, cambia la forma a (6, 5376)
output_data = output_data.T  # Cambia la forma a (5376, 6)

# Parámetros de visualización
conf_threshold = 0.8  # Umbral bajo para ver si aparecen detecciones
class_names = ['crosswalk', 'guide_arrows']  # Nombres de las clases

# Procesar las detecciones
h, w = image.shape[:2]
for detection in output_data:
    x_center, y_center, width, height, confidence, class_id = detection
    
    # Filtrar detecciones por confianza
    if confidence > conf_threshold:
        class_id = int(class_id)

        # Convertir a coordenadas de la imagen original
        x1 = int((x_center - width / 2) * w / 512)
        y1 = int((y_center - height / 2) * h / 512)
        x2 = int((x_center + width / 2) * w / 512)
        y2 = int((y_center + height / 2) * h / 512)

        # Dibujar la caja y la etiqueta en la imagen
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Mostrar la imagen con las detecciones
cv2.imshow("Detecciones", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
