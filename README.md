# Neural Style Transfer Web Application built with Streamlit

Recently, inspired by the power of Convolutional Neural
Networks (CNNs), Gatys et al. [10] first studied how to use
a CNN to reproduce famous painting styles on natural
images. They proposed to model the content of a photo as
the feature responses from a pre-trained CNN, and further
model the style of an artwork as the summary feature
statistics. Their experimental results demonstrated that a
CNN is capable of extracting content information from an
arbitrary photograph and style information from a wellknown artwork. Based on this finding, Gatys et al. [10] first
proposed to exploit CNN feature activations to recombine
the content of a given photo and the style of famous artworks. The key idea behind their algorithm is to iteratively
optimise an image with the objective of matching desired
CNN feature distributions, which involves both the photo’s
content information and artwork’s style information. Their
proposed algorithm successfully produces stylised images
with the appearance of a given artwork.

<img src="https://github.com/Timothy102/neuralstyletransfer/blob/main/download.jpeg" alt="drawing" width="600"/>
