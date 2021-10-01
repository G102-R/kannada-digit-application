import os
import numpy as np
import cv2
import pandas as pd
from tensorflow.keras.models import load_model
import streamlit as st
from streamlit_drawable_canvas import st_canvas

#MODEL_DIR = os.path.join(os.path.dirname(__file__), 'model_conv,h5')
#if not os.path.isdir(MODEL_DIR):
#    os.system('/CONVNET.ipynb')

model = load_model('model_conv.h5')
st.markdown('<style>body{color: white; background-color: DarkSlateGrey}</style>', unsafe_allow_html=True)

st.write("""
         # Kannada Digit Recognizer
         ### Recognizing kannada digits in realtime from the user data
         """
         )
st.markdown('''
Try to write a Kannada digit!
''')
#st.write("![Your Awsome GIF](https://media.giphy.com/media/3ohzdIuqJoo8QdKlnW/giphy.gif)")
#st.write("![Your Awsome GIF](https://media.giphy.com/media/26n7b7PjSOZJwVCmY/giphy.gif)")

data = np.random.rand(28,28)
img = cv2.resize(data, (256, 256), interpolation=cv2.INTER_NEAREST)

SIZE = 350
mode = st.checkbox("Draw (or Delete)?", True)
canvas_result = st_canvas(
    fill_color='#000000',
    stroke_width=20,
    stroke_color='#FFFFFF',
    background_color='#000000',
    width=SIZE,
    height=SIZE,
    drawing_mode="freedraw" if mode else "transform",
    key='canvas')

if canvas_result.image_data is not None:
    img = cv2.resize(canvas_result.image_data.astype('uint8'), (28, 28))
    rescaled = cv2.resize(img, (SIZE, SIZE), interpolation=cv2.INTER_NEAREST)
    st.write('Model Input')
    st.image(rescaled)

if st.button('Predict'):
    test_x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    val = model.predict(test_x.reshape(-1,28,28, 1))
    st.write(f'result: {np.argmax(val[0])}')
    st.bar_chart(val[0])
    #st.area_chart(val[0])



st.write("""
# About our Project
## Sample from the dataset we used!
![Your Awsome GIF](https://raw.githubusercontent.com/vinayprabhu/Kannada_MNIST/master/example.png)


Bibtex entry:
```latex
@article{prabhu2019kannada,
  title={Kannada-MNIST: A new handwritten digits dataset for the Kannada language},
  author={Prabhu, Vinay Uday},
  journal={arXiv preprint arXiv:1908.01242},
  year={2019}
}

""")

st.write("""### Sample Data from the dataset  """)
df = pd.read_csv("test.csv")
df = pd.DataFrame(df)
st.write(df.head())


my_expander = st.beta_expander(label='Neural Net. Models used')
with my_expander:
    'Simple Convolutional Neural Network (CNN)'
    'VGG-16'
    'Capsule Neural Network'
