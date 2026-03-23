import cv2
import mediapipe as mp

cap = cv2.VideoCapture(0)

mp_face_detection = mp.solutions.face_detection
# face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


mpDraw = mp.solutions.drawing_utils


while True:
    success, img = cap.read()
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    face_detection_results = face_detection.process(imgRGB)
    
    print(face_detection_results.detections)
        
    if face_detection_results.detections:
        for face_no, face in enumerate(face_detection_results.detections):
            print(f"Face Number: {face_no+1}")
            print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')

            face_data = face.location_data
            print(f'FACE BOUNDING BOX:n{face_data.relative_bounding_box}')
      
            mpDraw.draw_detection(img, face)

        for i in range(2):
            print(f'{mp_face_detection.FaceKeyPoint(i).name}:')
            print(f'{face_data.relative_keypoints[mp_face_detection.FaceKeyPoint(i).value]}')
    
    cv2.imshow("Image", img)
    
    if cv2.waitKey() & 0xFF == ord('q'):
        break
    
    
    
    
