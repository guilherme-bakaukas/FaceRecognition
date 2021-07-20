import face_recognition
import os
import cv2

KNOWN_FACES_DIR = "known_faces"
TOLERANCE = 0.6
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = "cnn"

video = cv2.VideoCapture(-1)

print("loading known faces")

known_faces = []
known_names = []

# adicionamos as imagens com os nomes lo e gui para cada imagem

for name in os.listdir(KNOWN_FACES_DIR):
    for filename in os.listdir(f"{KNOWN_FACES_DIR}/{name}"):
        image = face_recognition.load_image_file(f"{KNOWN_FACES_DIR}/{name}/{filename}")
        encoding = face_recognition.face_encodings(image)[0]
        known_faces.append(encoding)
        known_names.append(name)

# agora vamos analisar as unkonwn faces para comparar

print("processing unknown faces")



while (video.isOpened()):

    ret, image = video.read()
    
    locations = face_recognition.face_locations(image, model=MODEL)
    encodings = face_recognition.face_encodings(image, locations)

    for face_encoding, face_location in zip(encodings, locations):
        
        results = face_recognition.compare_faces(known_faces, face_encoding, TOLERANCE)
        matchName = None
        
        if True in results:
            matchName = known_names[results.index(True)]
            print(f"Match found: {matchName}")

            top_left = (face_location[3], face_location[0])
            botton_right = (face_location[1], face_location[2])

            color = [0, 255, 0]

            cv2.rectangle(image, top_left, botton_right, color, FRAME_THICKNESS)

            # retangulo menor
            top_left = (face_location[3], face_location[2])
            botton_right = (face_location[1], face_location[2]+22)
            cv2.rectangle(image, top_left, botton_right, color, cv2.FILLED)
            cv2.putText(image, matchName, (face_location[3]+10, face_location[2]+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), FONT_THICKNESS)
    
    cv2.imshow(filename, image)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # pressionar q para sair do video
