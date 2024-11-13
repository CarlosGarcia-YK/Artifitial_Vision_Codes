import onnxruntime as ort
import cv2
import numpy as np
import time
from matplotlib import pyplot as plt

# Cargar el modelo ONNX de YOLO
session = ort.InferenceSession(r"runs\YOLO_STOP2\weights\best.onnx")

# Nombres de las clases
class_names = ['crosswalk', 'guide_arrows']
conf_threshold = 0.8

# Variables de estado y temporizadores
last_crosswalk_time = 0
ignore_crosswalk_duration = 5  # Duración para ignorar crosswalk
ignore_lines_duration = 5  # Duración para ignorar detección de líneas
ignore_crosswalk = False
detect_lines = True

# Variables para la detección de carriles
last_left_line = None
last_right_line = None

def preprocess_frame(image):
    input_image = cv2.resize(image, (512, 512))
    input_image = input_image.astype('float32') / 255.0
    input_image = np.transpose(input_image, (2, 0, 1))
    input_image = np.expand_dims(input_image, axis=0)
    return input_image

def detect_objects(frame):
    input_image = preprocess_frame(frame)
    inputs = {session.get_inputs()[0].name: input_image}
    outputs = session.run(None, inputs)
    output_data = outputs[0][0].T

    h, w = frame.shape[:2]
    detections = []
    for detection in output_data:
        x_center, y_center, width, height, confidence, class_id = detection
        if confidence > conf_threshold:
            class_id = int(class_id)
            x1 = int((x_center - width / 2) * w / 512)
            y1 = int((y_center - height / 2) * h / 512)
            x2 = int((x_center + width / 2) * w / 512)
            y2 = int((y_center + height / 2) * h / 512)
            detections.append((x1, y1, x2, y2, confidence, class_id))
    return detections

def draw_detections(frame, detections):
    global last_crosswalk_time, ignore_crosswalk, detect_lines
    for (x1, y1, x2, y2, confidence, class_id) in detections:
        if class_id == 0 and not ignore_crosswalk:  # 0 es el índice de "crosswalk"
            last_crosswalk_time = time.time()
            ignore_crosswalk = True
            detect_lines = False
            print("Crosswalk detectado, pausando detección de líneas por 5 segundos.")

        if ignore_crosswalk and class_id == 0:
            continue  # Ignora detecciones de crosswalk cuando está en modo de ignorar

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{class_names[class_id]}: {confidence:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Funciones para detección de carriles
def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    canny = cv2.Canny(blur, 30, 150)
    mask = np.ones_like(canny) * 255
    cutoff = int(canny.shape[0] - canny.shape[0] // 8)
    mask[cutoff:, :] = 0
    return cv2.bitwise_and(canny, mask)

def region_of_interest(canny):
    height, width = canny.shape[:2]
    mask = np.zeros_like(canny)
    triangle = np.array([[
        (width // 9, height),
        (width // 2, int(height * 0.60)),
        (width - width // 9, height)
    ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    return cv2.bitwise_and(canny, mask)

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 75, minLineLength=30, maxLineGap=100)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

"""def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.2 / 5) #3.2 /5
    x1 = max(0, min(int((y1 - intercept) / slope), image.shape[1]))
    x2 = max(0, min(int((y2 - intercept) / slope), image.shape[1]))
    return [[x1, y1, x2, y2]]
"""

def make_points(image, line, line_length_ratio=0.25):
    slope, intercept = line
    y1 = int(image.shape[0])  # Punto inferior en la imagen (base de la línea)
    
    # Calcular el punto final y2 basado en la proporción de longitud deseada
    y2 = int(y1 - (image.shape[0] * line_length_ratio))  # Acorta la línea en una fracción de la altura
    
    # Calcular x1 y x2 basados en y1 y y2
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    
    # Limitar x1, x2 para mantenerlas dentro de la imagen
    x1 = max(0, min(x1, image.shape[1]))
    x2 = max(0, min(x2, image.shape[1]))

    return [[x1, y1, x2, y2]]
def average_slope_intercept(image, lines):
    global last_left_line, last_right_line
    left_fit, right_fit = [], []
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)
            if slope < -0.5:
                left_fit.append((slope, intercept))
            elif slope > 0.5:
                right_fit.append((slope, intercept))

    left_line = last_left_line if not left_fit else make_points(image, np.average(left_fit, axis=0))
    right_line = last_right_line if not right_fit else make_points(image, np.average(right_fit, axis=0))
    
    last_left_line = left_line if left_line else last_left_line
    last_right_line = right_line if right_line else last_right_line

    if left_line and right_line:
        return [left_line, right_line]
    elif left_line:
        return [left_line]
    elif right_line:
        return [right_line]
    return None

# Captura de video desde la cámara
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()

    # Revisar el tiempo para activar/desactivar detecciones
    if ignore_crosswalk and (current_time - last_crosswalk_time >= ignore_crosswalk_duration):
        ignore_crosswalk = False
        print("Reanudando detección de crosswalk.")
        
    if not detect_lines and (current_time - last_crosswalk_time >= ignore_lines_duration):
        detect_lines = True
        print("Reanudando detección de líneas.")

    # Detectar objetos con YOLO
    detections = detect_objects(frame)
    draw_detections(frame, detections)

    # Detectar y visualizar carriles si está permitido
    if detect_lines:
        canny_image = canny(frame)
        cropped_canny = region_of_interest(canny_image)
        lines = houghLines(cropped_canny)
        averaged_lines = average_slope_intercept(frame, lines)

        if averaged_lines:
            line_image = display_lines(frame, averaged_lines)
            frame = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # Mostrar el cuadro usando matplotlib
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.draw()
    plt.pause(0.001)
    plt.clf()

cap.release()
plt.close()
