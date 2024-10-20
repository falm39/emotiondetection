from keras.models import load_model
from time import sleep
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt

# Haar cascade ve eğitilmiş model yolu
face_classifier = cv2.CascadeClassifier('/Users/falm/uygulamakodları/emotion/haarcascade_frontalface_default.xml')
classifier = load_model('/Users/falm/uygulamakodları/emotion/model.h5')

emotion_labels = ['Kizgin', 'Igrenmis', 'Korkmus', 'Mutlu', 'Notr', 'Uzgun', 'Saskin']

cap = cv2.VideoCapture(0)

# Boş bir DataFrame oluştur
emotion_data = pd.DataFrame(columns=['Time', 'Emotion'])

while True:
    _, frame = cap.read()
    labels = []
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)

        if np.sum([roi_gray]) != 0:
            roi = roi_gray.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            prediction = classifier.predict(roi)[0]
            label = emotion_labels[prediction.argmax()]
            label_position = (x, y)
            cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Zaman bilgisini al
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

            # DataFrame'e yeni bir satır ekle
            new_row = pd.DataFrame({'Time': [current_time], 'Emotion': [label]})
            emotion_data = pd.concat([emotion_data, new_row], ignore_index=True)

    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        # OpenCV ile popup oluştur ve kullanıcıdan isim al
        name_window = np.zeros((200, 600), dtype=np.uint8)
        cv2.putText(name_window, "Lutfen isminizi giriniz: ", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        cv2.imshow('Enter Name', name_window)
        user_name = ""
        while True:
            key = cv2.waitKey(0)
            if key == 13:  # Enter tuşuna basıldığında döngüden çık
                break
            elif key == 8:  # Backspace tuşuna basıldığında son karakteri sil
                user_name = user_name[:-1]
            elif key >= 32 and key <= 126:  # Diğer karakterleri kullanıcı ismine ekle
                user_name += chr(key)
            # Kullanıcı ismini güncelle
            name_display = name_window.copy()
            cv2.putText(name_display, user_name, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.imshow('Enter Name', name_display)

        cv2.destroyWindow('Enter Name')

        if user_name:
            # Kaydedilen duygu verilerini CSV dosyasına kaydet
            emotion_data.to_csv(f'{user_name}_emotion_data.csv', index=False)

            # Her duygunun sayısını hesapla
            emotion_counts = emotion_data['Emotion'].value_counts()

            # Pasta grafiği olarak kaydet ve göster
            plt.figure(figsize=(10, 6))
            plt.pie(emotion_counts, labels=emotion_counts.index, autopct='%1.1f%%', startangle=140)
            plt.title('Emotion Distribution')
            plt.savefig(f'{user_name}_emotion_distribution.png')
            plt.show()
        break

cap.release()
cv2.destroyAllWindows()
