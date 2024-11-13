import cv2
import numpy as np

last_left_line = None
last_right_line = None

def canny(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = 5
    blur = cv2.GaussianBlur(gray, (kernel, kernel), 0)
    canny = cv2.Canny(blur, 30, 150)
    height, width = canny.shape
    mask = np.ones_like(canny) * 255  # Crear una máscara blanca (255)
    cutoff = int(height - height // 8)
    mask[cutoff:height, :] = 0  # Enmascarar la parte inferior (color negro)
    masked_canny = cv2.bitwise_and(canny, mask)
    return masked_canny

def region_of_interest(canny):
    height = canny.shape[0]
    width = canny.shape[1]
    mask = np.zeros_like(canny)
    triangle = np.array([[ 
        (width // 9, height),  # Punto en el borde izquierdo
        (width // 2, int(height * 0.60)),  # Punto en el centro, más arriba
        (width - width // 9, height)  # Punto en el borde derecho
    ]], np.int32)
    cv2.fillPoly(mask, triangle, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    return masked_image

def houghLines(cropped_canny):
    return cv2.HoughLinesP(cropped_canny, 2, np.pi / 180, 50, 
                           np.array([]), minLineLength=30, maxLineGap=100)

def addWeighted(frame, line_image):
    return cv2.addWeighted(frame, 0.8, line_image, 1, 1)

def display_lines(img, lines):
    line_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            for x1, y1, x2, y2 in line:
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 0, 255), 10)
    return line_image

def make_points(image, line):
    slope, intercept = line
    y1 = int(image.shape[0])
    y2 = int(y1 * 3.5 / 5)
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return [[x1, y1, x2, y2]]

def average_slope_intercept(image, lines):
    global last_left_line, last_right_line

    left_fit = []
    right_fit = []
    if lines is None:
        return None

    for line in lines:
        for x1, y1, x2, y2 in line:
            fit = np.polyfit((x1, x2), (y1, y2), 1)
            slope = fit[0]
            intercept = fit[1]

            if slope < -0.5:  # Filtro para líneas del lado izquierdo
                left_fit.append((slope, intercept))
            elif slope > 0.5:  # Filtro para líneas del lado derecho
                right_fit.append((slope, intercept))
 
    left_line = None
    right_line = None
 
    if len(left_fit) > 0:
        left_fit_average = np.average(left_fit, axis=0)
        left_line = make_points(image, left_fit_average)
        last_left_line = left_line  # Actualizar la última línea izquierda válida

    if len(right_fit) > 0:
        right_fit_average = np.average(right_fit, axis=0)
        right_line = make_points(image, right_fit_average)
        last_right_line = right_line  # Actualizar la última línea derecha válida

    if left_line is None and last_left_line is not None:
        left_line = last_left_line

    if right_line is None and last_right_line is not None:
        right_line = last_right_line
 
    if left_line is not None and right_line is not None:
        return [left_line, right_line]
    elif left_line is not None:
        return [left_line]
    elif right_line is not None:
        return [right_line]
    else:
        return None

# Función para verificar la posición del vehículo
def check_position(image, left_line, right_line):
    image_center = image.shape[1] // 2  # Obtener el centro de la imagen (en píxeles)
    
    # Obtener las posiciones horizontales en la parte inferior de la imagen (x1 para cada línea)
    left_x1 = left_line[0][0]
    right_x1 = right_line[0][0]
    
    # Calcular el centro del carril
    lane_center = (left_x1 + right_x1) // 2
    
    # Definir un margen de tolerancia para considerar el coche centrado
    tolerance = 125  # Puedes ajustar este valor según el tamaño de la imagen

    # Comparar la posición del centro del carril con el centro de la imagen
    if abs(image_center - lane_center) <= tolerance:
        return "Centrado"
    elif lane_center < image_center:
        return "Desviado a la izquierda"
    else:
        return "Desviado a la derecha"

# Iniciar captura de video
cap = cv2.VideoCapture("C:\\Users\\yourk\\Documents\\9QT\\Artificial Vision\\Codes\\pictures\\test2.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Aplicar Canny con enmascarado
    canny_image = canny(frame)
    
    # Aplicar la región de interés (triángulo)
    cropped_canny = region_of_interest(canny_image)

    # Detección de líneas Hough
    lines = houghLines(cropped_canny)

    # Promedio de las líneas detectadas
    averaged_lines = average_slope_intercept(frame, lines)

    # Dibujar las líneas detectadas
    if averaged_lines is not None:
        line_image = display_lines(frame, averaged_lines)
        combo_image = addWeighted(frame, line_image)
        
        # Verificar la posición del vehículo
        if len(averaged_lines) == 2:
            left_line, right_line = averaged_lines
            position = check_position(frame, left_line, right_line)
            print(f"Posición del vehículo: {position}")
    else:
        combo_image = frame

    # Mostrar el resultado
    cv2.imshow("Lane Detection", combo_image)
    
    # Presionar 'q' para salir
