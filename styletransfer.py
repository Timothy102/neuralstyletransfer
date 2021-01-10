import streamlit as st

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')
import functools
import requests
import io
import os
import PIL
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
print("Eager mode enabled: ", tf.executing_eagerly())

def crop_center(image):
  """Returns a cropped square image."""
  shape = image.shape
  new_shape = min(shape[1], shape[2])
  offset_y = max(shape[1] - shape[2], 0) // 2
  offset_x = max(shape[2] - shape[1], 0) // 2
  image = tf.image.crop_to_bounding_box(
      image, offset_y, offset_x, new_shape, new_shape)
  return image

@functools.lru_cache(maxsize=10000)
@st.cache(allow_output_mutation=True, max_entries=10, ttl=3600)
def load_image(image_url, image_size=(256, 256), preserve_aspect_ratio=True):
  """Loads and preprocesses images."""
  # Cache image file locally.
  image_path = tf.keras.utils.get_file(os.path.basename(image_url)[-128:], image_url)
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img = plt.imread(image_path).astype(np.float32)[np.newaxis, ...]
  if img.max() > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  img = crop_center(img)
  img = tf.image.resize(img, image_size, preserve_aspect_ratio=True)
  return img


# @title Load example images  { display-mode: "form" }

content_image_url = 'https://upload.wikimedia.org/wikipedia/commons/a/ad/Angelina_Jolie_2_June_2014_%28cropped%29.jpg'  # @param {type:"string"}
style_image_url = 'https://media.overstockart.com/optimized/cache/data/product_images/VG485-1000x1000.jpg'


output_image_size = 460  # @param {type:"integer"}

# The content image size can be arbitrary.
content_img_size = (output_image_size, output_image_size)
# The style prediction model was trained with image size 256 and it's the 
# recommended image size for the style image (though, other sizes work as 
# well but will lead to different results).
style_img_size = (256, 256)  # Recommended to keep it at 256.

content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')


# Load TF-Hub module.
hub_handle = 'https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/2'
hub_module = hub.load(hub_handle)


''' # Neural Style Transfer'''
st.write('Neural Style Transfer is a technique that uses deep learning to compose one image in the style of another image. Have your ever wished you could paint like Picasso or Van Gogh? This is your chance! \n')

st.write('NST algorithms are characterized by their use of deep neural networks for the sake of image transformation. This is implemented by optimizing the output image to match the content statistics of the content image and the style statistics of the style reference image. These statistics are extracted from the images using a convolutional network. Common uses for NST are the creation of artificial artwork from photographs, for example by transferring the appearance of famous paintings to user-supplied photographs. Several notable mobile apps use NST techniques for this purpose, including DeepArt and Prisma. This method has been used by artists and designers around the globe to develop new artwork based on existent style(s).')
st.write("Feel free to play around with different styles and I strongly encourage you to try out the already made styles in the sidebar on your left. You'll be making transformations like the one below in a few seconds!")
def imageFromURL(url,img_shape=(320,240)):
    response = requests.get(url)
    image_bytes = io.BytesIO(response.content)

    img = PIL.Image.open(image_bytes)
    return img.resize(img_shape)
gen_image_url = 'https://s3.amazonaws.com/book.keras.io/img/ch8/style_transfer.png'
gen_image = imageFromURL(gen_image_url,img_shape=(600,200))
st.image(gen_image)
st.write("Okay, are you ready? Here's all you need to do. Choose the content image and paste its URL into the first box. Do the same for the style image. Once you're ready, click the button below and get ready to be amazed!!! :sunglasses:")
# Sidebar along with premade styles
st.sidebar.title("Here's a list of some premade styles you can use!")
select = st.sidebar.selectbox('List', ['Starry night','Clocks','Picasso Portret',
'Mona Lisa','The Kiss by Klimt','Birth of Venus','Church in Auvers','Sejalec','The Scream','Kofetarica',], key='1')

if not st.sidebar.checkbox("Hide", False, key='1'):
    st.title("Styles")
    if select == 'Starry night':
      style_image_url = 'https://media.overstockart.com/optimized/cache/data/product_images/VG485-1000x1000.jpg'
    elif select =='Clocks':
      style_image_url = 'https://upload.wikimedia.org/wikipedia/en/d/dd/The_Persistence_of_Memory.jpg'
    elif select =='Picasso Portret':
      style_image_url = 'https://images.saatchiart.com/saatchi/1311333/art/6500245/5569923-AOAGHVQR-7.jpg'
    elif select == 'Mona Lisa':
      style_image_url = 'https://cdn.cnn.com/cnnnext/dam/assets/190430171751-mona-lisa.jpg'
    elif select == 'The Kiss by Klimt':
      style_image_url = 'https://i.pinimg.com/originals/46/44/7b/46447b35c81b2d750d29e27f7738a6a6.jpg'
    elif select == 'Birth of Venus':
      style_image_url = 'https://art-sheep.com/wp-content/uploads/2019/06/Sandro-Botticelli-Birth-of-Venus-1024x683.jpg'
    elif select == 'Church in Auvers':
      style_image_url = 'https://cdn.theculturetrip.com/wp-content/uploads/2019/01/vincent_van_gogh_-_the_church_in_auvers-sur-oise_view_from_the_chevet_-_google_art_project.jpg'
    elif select =='The Scream':
      style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/9/9d/The_Scream_by_Edvard_Munch%2C_1893_-_Nasjonalgalleriet.png'
    elif select == 'Kofetarica':
      style_image_url = 'https://upload.wikimedia.org/wikipedia/commons/6/61/Ivana_Kobilca_-_Kofetarica.jpg'
    elif select == 'Sejalec':
      style_image_url = 'https://www.bolha.com/image-bigger/slike-umetnine-starine/slika-sejalec-lilijana-levstik-olje-platno-slika-15130001.jpg'

st.sidebar.write("Check the box above if you want to use your own style image. Otherwise, feel free to check what your image would look like painted by the world's greatest artists!!!")      

content_image_url = st.text_input('Enter the content image URL here: ',content_image_url)
style_image_url = st.text_input('Enter the style image URL here: ',style_image_url)

content_image = load_image(content_image_url, content_img_size)
style_image = load_image(style_image_url, style_img_size)
style_image = tf.nn.avg_pool(style_image, ksize=[3,3], strides=[1,1], padding='SAME')

outputs = hub_module(tf.constant(content_image), tf.constant(style_image))
stylized_image = outputs[0]

# Visualize input images and the generated stylized image.

stylized_image = tf.reshape(stylized_image,(output_image_size,output_image_size,3))
stylized_image = np.array(stylized_image)


if st.button('Press this button to see your new image!'):
  st.image(stylized_image)
''' ## Further reading '''
st.write("If you find neural style transfer fascinating and want to learn more, there are numerous number of resources you can use. As you read, don't get too caught up in the hard core of deep learning and mathematics and keep in mind that the same idea of applying styles is still essential.")
st.write('Conceptually most closely related are methods using texture transfer to achieve artistic style transfer. However, these  approaches mainly rely on non-parametric techniques to directly manipulate the pixel representation of an image. In contrast, by using Deep Neural Networks trained on object recognition, we carry out manipulations in feature spaces that explicitly represent the high level content of an image.So this means that the specialty of the deep learning approach is to extract the style of an image not with mere pixel observation of the style picture, but rather the extracted features of the pre-trained model combined with the content of the style image. So, in essence, to discover the style of an image, we process the style image by analyzing its pixels feeding this information to the layer of a pre-trained model to “understand”/classify the provided input as objects. For a concrete walkthrough, there is a briliant article on [Sunhine at the moon](https://sunshineatnoon.github.io/posts/2017/05/blog-post-1/).')
st.write('All of these ideas are supported by the machine learning community on websites like [TensorFlow](https://www.tensorflow.org/tutorials/generative/style_transfer) as well as [Towards Data Science](https://towardsdatascience.com/neural-style-transfer-a-high-level-approach-250d4414c56b)')

arc_url = 'https://sunshineatnoon.github.io/assets/posts/2017-05-19-a-brief-summary-on-neural-style-transfer/styleBank.png'
architecture = imageFromURL(arc_url,img_shape=(600,200))
st.image(architecture)

st.write('')
st.write('Based on the model code in [Original TensorFlow Model which is also available on TensorFlow Hub](https://github.com/tensorflow/magenta/tree/master/magenta/models/arbitrary_image_stylization) and the publication:[Exploring the structure of a real-time, arbitrary neural artistic stylization network](https://arxiv.org/abs/1705.06830).*Golnaz Ghiasi, Honglak Lee.')
st.write('Author: Tim Cvetko, you can find me on [LinkedIn](https://www.linkedin.com/in/tim-cvetko-32842a1a6/), [Github] (https://github.com/timothy102), or [Medium](https://cvetko-tim.medium.com/). ')


