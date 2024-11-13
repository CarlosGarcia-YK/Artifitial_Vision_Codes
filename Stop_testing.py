import onnxruntime as ort
import cv2
import numpy as np

# Cargar el modelo ONNX
session = ort.InferenceSession(r"runs\YOLO_STOP2\weights\best.onnx")

# Configurar la cámara
cap = cv2.VideoCapture(0)  # Cambia el índice si tienes varias cámaras
input_shape = (512, 512)  # Tamaño de entrada del modelo

# Parámetros de visualización
conf_threshold = 0.8  # Umbral para mostrar detecciones con confianza superior a 30%
class_name = 'stop_signal'  # Nombre de la clase única, asumiendo detección de una sola clase

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error al acceder a la cámara.")
        break

    h, w = frame.shape[:2]  # Obtener dimensiones originales de la imagen
    input_image = cv2.resize(frame, input_shape)  # Ajustar tamaño al de entrada del modelo

    # Preprocesar la imagen
    input_image = input_image.astype('float32') / 255.0  # Normalizar a [0,1]
    input_image = np.transpose(input_image, (2, 0, 1))  # Cambiar a (C, H, W)
    input_image = np.expand_dims(input_image, axis=0)  # Agregar la dimensión de batch

    # Realizar la inferencia
    inputs = {session.get_inputs()[0].name: input_image}
    outputs = session.run(None, inputs)

    # Interpretar y reorganizar los datos de salida
    output_data = outputs[0][0]  # Seleccionar el batch 0
    output_data = output_data.T  # Transponer si es necesario

    # Procesar las detecciones
    for detection in output_data:
        x_center, y_center, width, height, confidence = detection
        
        # Filtrar detecciones por confianza
        if confidence > conf_threshold:
            # Convertir a coordenadas de la imagen original
            x1 = int((x_center - width / 2) * w / 512)
            y1 = int((y_center - height / 2) * h / 512)
            x2 = int((x_center + width / 2) * w / 512)
            y2 = int((y_center + height / 2) * h / 512)

            # Dibujar la caja y la etiqueta en la imagen
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name}: {confidence:.2f}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Mostrar el cuadro de video con las detecciones
    cv2.imshow("Detecciones en Tiempo Real", frame)

    # Presiona 'q' para salir
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()
