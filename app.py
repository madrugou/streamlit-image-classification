from tflite_runtime.interpreter import Interpreter
from loguru import logger
from PIL import Image
import streamlit as st 
import numpy as np
import imutils 
import cv2 



class App:


    def __init__(self, label, model):
        self.label = label
        self.model = model
        self.resize = (224, 224)
    
    @st.cache
    def load_label(self):
        
        with open(self.label, 'r') as f:
            return {i: line.strip() for i, line in enumerate(f.readlines())}

    
    def set_input_tensor(self, interpreter, image):
        tensor_index = interpreter.get_input_details()[0]['index']
        input_tensor = interpreter.tensor(tensor_index)()[0]
        input_tensor[:, :] = image

    def classify_image(self, interpreter, image, top_k=1):
        self.set_input_tensor(interpreter, image)

        interpreter.invoke()
        output_details = interpreter.get_output_details()[0]
        output = np.squeeze(interpreter.get_tensor(output_details['index']))

        scale, zero_point = output_details['quantization']
        output = scale * (output - zero_point)

        ordered = np.argpartition(-output, 1)
        return [(i, output[i]) for i in ordered[:top_k]][0]
    
    def run(self):
        
        st.title(' :fuelpump: Classificação de abastecimento')

        st.write('Aplicativo para classificação de imagem, onde é possível identificar através da imagem se está ou não abastacendo o equipamento corretamente. ')

        labels = self.load_label()
        st.write('**Labels:**')
        st.write(f'```json {labels} ```')

        st.set_option('deprecation.showfileUploaderEncoding', False)
        image = st.file_uploader("Carregue aqui a sua imagem:", type=['png', 'jpg', 'jpeg'])

        if image is not None:
            
            interpreter = Interpreter(self.model)
            interpreter.allocate_tensors()

            _, height, width, _ = interpreter.get_input_details()[0]['shape']

            user_image = Image.open(image).convert('RGB').resize((width, height))

            label_id, prob = self.classify_image(interpreter, user_image)
            
            frame = np.asarray(user_image) 
            
            st.write(f'**Classificação:**')
            st.write(f"*{labels[label_id]}: {np.round(prob*100, 2)} %*")

            show = st.image([])
            show.image(frame, '', use_column_width=True)


           

            


if __name__ == '__main__':

    app = App(label='./model/labels.txt', model='./model/model.tflite')
    app.run()