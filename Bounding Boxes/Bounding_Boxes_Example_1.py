import cv2

filepath = "\\Users\\mauricio\\Documents\\Maurício Devincentis\\UFABC\\Pesquisas\\IC - Faster R-CNN\\Bounding boxes\\letras.jpg"

print("Leitura da imagem...")
img = cv2.imread(filepath)


print(f"Tamanho da imagem: {img.shape}")
print("Redefinindo o tamanho da imagem para 512 x 512")
img = cv2.resize(img, (512, 512))
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

cv2.imshow("Imagem", img_gray)
cv2.waitKey(0)

img_blurred = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("Imagem", img_blurred)
cv2.waitKey(0)

img_edge = cv2.Canny(img_blurred, 150, 300)
cv2.imshow("Imagem", img_edge)
cv2.waitKey(0)

print("Calculando os contornos...")
contornos, _ = cv2.findContours(img_edge, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print("Iteração sobre os contornos identificados...")
for nc, c in enumerate(contornos):
    (x, y, w, h) = cv2.boundingRect(c)

    limite_h = 80
    limite_w = 20
    if (w >= limite_w) and (h >= limite_h):
        print("w = ", w ," e  h: " , h)
        print("x = ", x ," e y: " , y)
        print("-------------------")
        roi = img_blurred[y: y+h, x: x+w]
        ret, roi = cv2.threshold(roi, 127, 255, cv2.THRESH_BINARY_INV)
        roi = cv2.bitwise_not(roi)
        cv2.imshow(f"ROI: {nc} - H: {h} W: {w}", roi)
        cv2.waitKey(0)
