from tensorflow.keras.preprocessing.image import img_to_array
from tflite_runtime.interpreter import Interpreter
from time import time, sleep
from imutils import paths
from loguru import logger
import streamlit as st
import numpy as np
import imutils
import cv2


class App:


    def __init__(self, model, label, images):
        self.model = model 
        self.label = label 
        self.images = sorted(list(paths.list_images(images)))
        self.resize = (224, 224)
    
    def save_image(self, name):

        n = name.replace(".", "_")
        n = n.replace("%", "")
        n = n.replace(":", "")
        n = n.replace(" ", "")

        return n

    def load_labels(self):
        with open(self.label, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}

    @staticmethod
    def set_input_tensor(interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def classify_image(self, interpreter, image, top_k=1):
        self.set_input_tensor(interpreter, image)
        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        if output_details['dtype'] == np.uint8:
            scale, zero_point = output_details['quantization']
            output = scale * (output - zero_point)

        ordered = np.argpartition(-output, top_k)
        return [(i, output[i]) for i in ordered[:top_k]]


    def inference(self):

        all_images = []
        all_labels = []

        total_inserted = 0
        total_not_inserted = 0

        labels = self.load_labels()

        interpreter = Interpreter(self.model)
        interpreter.allocate_tensors()

        _, height, width, _ = interpreter.get_input_details()[0]['shape']

        init_time = time()

        for i, frame in enumerate(self.images):

            image = cv2.imread(frame)
            image = cv2.resize(image, self.resize, fx=0.5, fy=0.5)

            show_image = image.copy()

            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image = img_to_array(image)

            image = np.expand_dims(image, axis=0)

            results = self.classify_image(interpreter, image)
            
            label_id, prob = results[0]

            if labels[label_id] == 'inserted':
                total_inserted += 1
            else:
                total_not_inserted += 1

            label = f'{labels[label_id]} {round(prob*100,2)} %'
            
            cv2.putText(show_image, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.imwrite(f'./processadas/{self.save_image(label)}_{i}.jpg', show_image)

            all_images.append(show_image)
            all_labels.append(label)

            logger.info(label)

        my_bar = st.progress(0)

        for p in range(100):
            sleep(0.01)
            my_bar.progress(p+1)

        time_process = round(time()-init_time, 2)

        total_images = len(all_images)

        st.write(f'**Foram processadas: {total_images} imagens em {time_process} segundos**')

        st.write('## Resultado:')
        st.write('Para o modelo treinado temos as seguintes predições:')

        st.write(f'> *:+1: inserted:*')
        st.write(f'**{total_inserted} imagens, que representam  {round((total_inserted*100) / total_images, 2)}%**')
       
        st.write(f'> *:-1: not_inserted:*')
        st.write(f'**{total_not_inserted} imagens, que {round((total_not_inserted*100) / total_images, 2)}%**')

        st.write('As imagens foram salvas na pasta **processadas**.')



    
    def run(self):


        st.title('Classificação de abastecimento')
        st.write('Clique no botão **"Classificar imagens"**, para carregar e classificar as imagens coletadas.')

        button = st.button(label='Classificar imagens')

        if button:

            st.write(' :hourglass: Processando as imagens ...')
            self.inference()


if __name__ == '__main__':
    
    app = App(model='./model/supply_15.tflite', label='./model/labels.txt', images='./images')
    app.run()