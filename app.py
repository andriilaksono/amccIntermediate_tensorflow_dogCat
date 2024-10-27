import tensorflow as tf
import streamlit as st 
import numpy as np
from PIL import Image

def load_model():
    model = tf.keras.models.load_model('intermediate_amcc.keras')
    return model 

def preprocessing_image(image):
    target_size = (64,64)
    image = image.resize(target_size)
    image_array = np.array(image)
    image_array = np.expand_dims(image_array,axis=0)
    image_array = image_array.astype('float32') / 255.0
    return image_array

def predict(model, image):
    return model.predict(image,batch_size=1)

def interpret_prediction(prediction):
    if prediction[-1] == 1:
        score = prediction[0][0]
        predicted_class = 0 if score >= 0.5 else 1
        confidence_scores = [score,1-score,0]
    else:
        confidence_scores = prediction[0]
        predicted_class = np.argmax(confidence_scores)
    return predicted_class,confidence_scores


def main():
    st.set_page_config(
        page_title = "pet image classifier",
        layout = "centered"
    )

    st.title("Dog and Cat")
    
    try:
        model = load_model()
        
    except Exception as err:
        st.error(f"error : {str(err)}")
        return
    
    st.sidebar.header('About')
    st.sidebar.write("""
        Aplikasi ini digunakan untuk mengklasifikasikan sebuah
        gambar ke dalam kategori berikut :
        - Kucing 
        - Anjing 

        **Instruction** 
        1. Drag and drog file ke pilih gambar
        2. Tunggu gambarnya sampai selesai di proses
        3. klik classify
        4. Hasil predikis akan keluar
    """)

    uploader = st.file_uploader("Pilih Gambar", type=('jpg', 'jpeg', 'png'))

    if uploader is not None:
        try:
            col1, col2 = st.columns([2,1])
            with col1:
                image = Image.open(uploader)
                st.image(image, caption= "gambar anda", use_column_width=True)
            
            with col2:
                if st.button("classify", use_container_width=True):
                    with st.spinner("Sedang Menghitung"):
                        processed_image = preprocessing_image(image)
                        prediction = predict(model, processed_image)
                        predicted_class,confidence_scores = interpret_prediction(prediction)
                        class_names = ["anjing", "kucing"]
                        result = class_names[predicted_class]
                        st.success(f"Hasil Prediksi : {result.capitalize()} {predicted_class}, {confidence_scores}")

        except Exception as err:
            st.error(f"error : {str(err)}")
            st.write("Tolong pilih file yang sesuai")
            st.write(f"error : {str(err)}")


if __name__ == "__main__":
    main()