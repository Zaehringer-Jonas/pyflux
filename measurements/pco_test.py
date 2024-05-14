import pco
import matplotlib.pyplot as plt

with pco.Camera() as cam:
    while True:
        cam.record()
        image, meta = cam.image()
    
        plt.imshow(image, cmap='gray')
        plt.show()