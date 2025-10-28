import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

with mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_draw.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_draw.DrawingSpec(color=(0,255,0), thickness=1)
                )

              
                top_lip = face_landmarks.landmark[13]
                bottom_lip = face_landmarks.landmark[14]
                mouth_open = abs(top_lip.y - bottom_lip.y)

               
                left_eyebrow = face_landmarks.landmark[65]
                left_eye_top = face_landmarks.landmark[159]
                eyebrow_raise = abs(left_eyebrow.y - left_eye_top.y)

                
                if mouth_open > 0.05 and eyebrow_raise > 0.02:
                    expression = "Surprised 😲"
                elif mouth_open > 0.04:
                    expression = "Happy 😄"
                else:
                    expression = "Neutral 😐"

                cv2.putText(frame, expression, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

        cv2.imshow("Face Expression Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()