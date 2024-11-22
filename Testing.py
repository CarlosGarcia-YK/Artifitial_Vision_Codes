from ultralytics import YOLO
import cv2

# Cargar los dos modelos PyTorch .pt
model1 = YOLO(r"runs\YOLO_TOTAL2\weights\best.pt")  # Primer modelo
model2 = YOLO(r"runs\YOLO_HUMAN3\weights\best.pt")  # Segundo modelo

# Parámetros de visualización
conf_threshold = 0.5  # Umbral de confianza
class_names_model1 = ['crosswalk', 'stop_signal']  # Clases del primer modelo
class_names_model2 = ['person']  # Clases del segundo modelo

# Configuración de la cámara
cap = cv2.VideoCapture(0)  # Cambia el índice si no es la cámara principal
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Configurar ancho de la cámara
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 320)  # Configurar altura de la cámara

while True:
    # Leer frame de la cámara
    ret, frame = cap.read()
    if not ret:
        print("No se pudo capturar la imagen de la cámara")
        break

    # Realizar inferencia con el primer modelo
    results1 = model1(frame, conf=conf_threshold)

    # Procesar las detecciones del primer modelo
    for result in results1:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja delimitadora
            confidence = box.conf[0].item()  # Confianza
            class_id = int(box.cls[0].item())  # ID de la clase

            # Dibujar la caja y la etiqueta en la imagen
            color = (255, 0, 0)  # Color para el modelo 1
            label = f"{class_names_model1[class_id]}: {confidence:.2f}" if class_id < len(class_names_model1) else f"ID {class_id}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Realizar inferencia con el segundo modelo
    results2 = model2(frame, conf=conf_threshold)

    # Procesar las detecciones del segundo modelo
    for result in results2:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordenadas de la caja delimitadora
            confidence = box.conf[0].item()  # Confianza
            class_id = int(box.cls[0].item())  # ID de la clase

            # Dibujar la caja y la etiqueta en la imagen
            color = (0, 255, 0)  # Color para el modelo 2
            label = f"{class_names_model2[class_id]}: {confidence:.2f}" if class_id < len(class_names_model2) else f"ID {class_id}: {confidence:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Mostrar la imagen con las detecciones de ambos modelos
    cv2.imshow("Detecciones", frame)

    # Presiona 'q' para salir del bucle
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
