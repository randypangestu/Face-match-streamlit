import streamlit as st
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
# If required, create a face detection pipeline using MTCNN:

def main(face_det, face_extract):
    st.write('Hello world!')
    uploaded_file_1 = st.file_uploader("Choose first image",type=["png","jpg","jpeg"])
    if uploaded_file_1 is not None:
        image_1 = Image.open(uploaded_file_1)
        st.image(image_1, caption='First Image')

    uploaded_file_2 = st.file_uploader("Choose second image",type=["png","jpg","jpeg"])

    if uploaded_file_2 is not None:
        image_2 = Image.open(uploaded_file_2)
        st.image(image_2, caption='Second Image')

    if st.button('calculate distance'):
        st.write('processing......')
        aligned_1 = torch.stack([face_det(image_1)]).to(device)
        aligned_2 = torch.stack([face_det(image_2)]).to(device)
        embed_1 = face_extract(aligned_1)[0].detach().cpu()
        embed_2 = face_extract(aligned_2)[0].detach().cpu()
        
        distance = (embed_1 - embed_2).norm().item()
        st.write(' Distance = {}'.format(distance))

if __name__=="__main__":
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(image_size=256, margin=0, min_face_size=20, device=device, post_process=True)

    # Create an inception resnet (in eval mode):
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    main(mtcnn, resnet)
