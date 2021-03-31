from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils import paths
from loguru import logger
from time import sleep
from PIL import Image
import tensorflow as tf
import streamlit as st 
import numpy as np
import itertools
import imutils
import cv2 
import os


class App:


    def __init__(self, model, path):
        self.model = model
        self.resize = (64, 64)
        self.images = sorted(list(paths.list_images(path))) 
    
    def save_image(self, label):

        a = label.replace(".", "_")
        a = a.replace("%", "")
        a = a.replace(":", "")
        a = a.replace(" ", "")

        return  a        

    def classify_images(self):  

        supply = load_model(self.model)
        
        all_images = []
        all_labels = []

        total_inserted = 0
        total_not_inserted = 0

        for i, frame in enumerate(self.images):
            
            
            image = cv2.imread(frame)
            show_image = image.copy()

            image = cv2.resize(image, self.resize, fx=0.5, fy=0.5)
            image = image.astype('float') / 255.0
            image = img_to_array(image)
            image = np.expand_dims(image, axis=0)

            (not_inserted, insertd) = supply.predict(image)[0]

            label = 'inserted' if insertd > not_inserted else 'not_inserted'

            if label == 'inserted':
                total_inserted += 1
            else:
                total_not_inserted += 1

            probability = insertd if insertd > not_inserted else not_inserted
            
            label = '{}: {:.2f}%'.format(label, probability * 100)
            
            output = cv2.resize(show_image, (360, 360), fx=0.5, fy=0.5)

            cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            cv2.imwrite(f'./processadas/{self.save_image(label)}_{i}.jpg', output)
            
            all_images.append(output)
            all_labels.append(label)

        
        st.write(f'**Foram processadas: {len(all_images)} imagens**')

        st.write('## Resultado:')
        st.write('Para o modelo treinado temos as seguintes predições:')

        st.write(f'*inserted: {total_inserted} imagens*')
        st.write(f'*not_inserted: {total_not_inserted} imagens*')

        st.write('As imagens foram salvas na pasta **processadas**.')

        st.image(list(all_images)[:12], width=180, caption=all_labels[:12], channels="BGR")
        

    def paginator(self, label, items, items_per_page=1000, on_sidebar=False):

        location = st.empty()
        items = list(items)
        n_pages = len(items)
        n_pages = (len(items) - 1) // items_per_page + 1
        page_format_func = lambda i: "Página %s" % i
        page_number = location.selectbox(label, range(n_pages), format_func=page_format_func)

        min_index = page_number * items_per_page
        max_index = min_index + items_per_page
        
        return itertools.islice(enumerate(items), min_index, max_index)



    def run(self):
        
        # Title
        st.title('Classificação de abastecimento')
        st.write('Clique no botão **"Classificar imagens"**, para carregar e classificar as imagens coletadas.')

        button = st.button(label='Classificar imagens')

        if button:
            
            # barra de progresso antes de mostrar as imagens

            st.write(' :hourglass: Processando imagens...')
            my_bar = st.progress(0)
            
            for p in range(100):
                sleep(0.01)
                my_bar.progress(p+1)

        
            self.classify_images()

        
            
    

if __name__ == '__main__':

    st.set_page_config(page_title='Solinftec', page_icon=':fuelpump:')

    app = App(model='./model/supply.model', path='./images')
    app.run()