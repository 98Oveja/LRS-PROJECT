from tensorflow.keras.models import load_model
from streamlit_webrtc import VideoProcessorBase, RTCConfiguration,webrtc_streamer, WebRtcMode

import streamlit as st
import mediapipe as mp
import numpy as np
import cv2
import av

mp_holistic = mp.solutions.holistic #modelo de mp
mp_drawing = mp.solutions.drawing_utils #importando utilidades

holistic = mp_holistic.Holistic(
    min_detection_confidence=0.5, 
    min_tracking_confidence=0.5
)


MODEL_PATH = 'models/modelFrases.h5'
# Cargamos el modelo preentrenado

@st.cache
def load_model():
	  return load_model(MODEL_PATH)
model = load_model()

actions = np.array(['por favor','feliz','mucho gusto','perdoname','hola','adios','gracias','yo','ayuda'])


RTC_CONFIGURATION = RTCConfiguration(
{"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

def draw_formateado_landmarks(image, results):
    # dibujar conexiones de cara
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
            mp_drawing.DrawingSpec(color=(0,255,255), thickness=1, circle_radius=1), 
            mp_drawing.DrawingSpec(color=(255,51,51), thickness=1, circle_radius=1)
            ) 
    # dibujar conexiones de poses
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
            ) 
    # dibujar conexiones de mano izquierda
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
            ) 
    # dibuajr conexiones de mano derecha
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
            mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4), 
            mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
            ) 
# ----------- Extraer los keypoints ------------
st.title("Sistema de Reconocimiento de Lengua de Señas (LENSEGUA) 🇬🇹")
st.markdown("**LENSEGUA** es la Lengua de señas oficial en Guatemala, según Decreto 3-2020. **LRS** es un sistema de reconocimiento que permite a través de la cámara de su dispositivo capturar los movimientos de LENSEGUA y traducirlo a texto.")

secuencia =[]
sentencia = []
predicciones = []
threshold = 0.3
traduction = ''



def process(image):
    global secuencia, sentencia, predicciones, threshold, traduction

    # with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    #     # while True:
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # CONVERSIÓN DE COLOR BGR 2 RGB
    image.flags.writeable = False                  # La imagen ya no se puede escribir, por eso es false
    results = holistic.process(image)                 # realizar prediction
    image.flags.writeable = True                 # ahora se puede escribir en la img
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 

    # col1_text.write()
    draw_formateado_landmarks(image, results)

    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    keypoints = np.concatenate([pose, face, lh, rh])
    # keypoints = extract_keypoints(results)
    secuencia.append(keypoints)
    secuencia = secuencia[-30:]

    if len(secuencia) == 30:
        resultado = model.predict(np.expand_dims(secuencia, axis=0))[0]
        print(actions[np.argmax(resultado)])
        predicciones.append(np.argmax(resultado))
        if np.unique(predicciones[-5:])[0]==np.argmax(resultado): 
            if resultado[np.argmax(resultado)] > threshold: 
                if len(sentencia) > 0: 
                    if actions[np.argmax(resultado)] != sentencia[-1]:
                        sentencia.append(actions[np.argmax(resultado)])
                        traduction = actions[np.argmax(resultado)]
                else:
                    sentencia.append(actions[np.argmax(resultado)])
                    traduction = actions[np.argmax(resultado)]

            if len(sentencia) > 5:
                sentencia = sentencia[-5:]

        cv2.rectangle(image, (0,440), (640, 580), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(traduction), (220,470), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image


class OpenCVVideoProcessor(VideoProcessorBase):
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="TEST",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=OpenCVVideoProcessor,
    async_processing=True,
)

st.markdown("Repositorio del proyecto [aquí](https://github.com/98Oveja/LRS-PROJECT)")

