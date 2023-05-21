import cv2
import  numpy as np

def SimpleThresholding(image, th, blur_coef, sigmaColor, sigmaSpace):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_blurred = cv2.GaussianBlur(gray, (blur_coef, blur_coef), sigmaColor)#, sigmaSpace)
    (Th, simpleTh_image) = cv2.threshold(bilateral_blurred, th, 255, cv2.THRESH_BINARY_INV)
    return simpleTh_image

def AdaptiveThresholding(image, blur_coef, sigmaColor, sigmaSpace, blockSize, C):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bilateral_blurred = cv2.bilateralFilter(gray, blur_coef, sigmaColor, sigmaSpace)
    adaptiveTh_image = cv2.adaptiveThreshold(
        bilateral_blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,blockSize,C
        )
    return adaptiveTh_image


def Laplacian_edge(th_image):
    image_lapEdge = np.uint8(np.abs(cv2.Laplacian(th_image, cv2.CV_64F)))
    return image_lapEdge

def Sobel_edge(th_image):
    image_sobelX = np.uint8(np.abs(cv2.Sobel(th_image, cv2.CV_64F,1,0)))
    image_sobelY = np.uint8(np.abs(cv2.Sobel(th_image, cv2.CV_64F,0,1)))
    image_sobel = cv2.bitwise_or(image_sobelX, image_sobelY)
    return image_sobel

def Canny_edge(th_image, lower_th, upper_th):
    canny_image = cv2.Canny(th_image, lower_th, upper_th)
    return canny_image

def Contours(edge_image, frame):
    (cnts,_) = cv2.findContours(edge_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(frame, cnts, -1, color = (10,155,250), thickness = 10)

video = cv2.VideoCapture(0)
process_this_time = True
while (True):
    ret, frame = video.read()

    if process_this_time:
        frame = cv2.flip(frame, 1)
        th_image = SimpleThresholding(frame, 150, 11, 0, 0)
        # th_image = AdaptiveThresholding(frame, 11, 111,131, 13,0)
        edge_image = Laplacian_edge(th_image)
        # edge_image = Sobel_edge(th_image)
        # edge_image = Canny_edge(th_image, 50, 150)
        new_frame = frame.copy()
        Contours(edge_image, new_frame)
        
        
    process_this_time = not process_this_time
    cv2.imshow('Video', new_frame)
    
    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video.release()
cv2.destroyAllWindows()

