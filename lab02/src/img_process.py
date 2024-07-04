from RPCA import RPCA
import numpy as np
from PIL import Image 
import os

class ImageChannelException(Exception):
    pass
    
def image_repair(image, in_max_iter: int, out_max_iter: int, tol):
    tol = float(tol)
    image = np.array(image)/255
    img_matrix = [image[:,:,0], image[:,:,1], image[:,:,2]]
    if len(img_matrix) != 3:
        raise(ImageChannelException("The number of input channels is inconrect"))
        
    rpca = []
    for item in img_matrix:
        rpca.append(RPCA(item, tol, out_max_iter, in_max_iter))
    
    A0, E0 = rpca[0].ADMM()
    A1, E1 = rpca[1].ADMM()
    A2, E2 = rpca[2].ADMM()
    
    A_img = 255*np.ones([A0.shape[0], A0.shape[1], 4])
    E_img = 255*np.ones([E0.shape[0], E0.shape[1], 4])
    A_img[:, :, 0] = 255*A0
    A_img[:, :, 1] = 255*A1
    A_img[:, :, 2] = 255*A2
    E_img[:, :, 0] = 255*E0
    E_img[:, :, 1] = 255*E1
    E_img[:, :, 2] = 255*E2
    A_img = Image.fromarray(np.uint8(A_img), 'RGBA')
    E_img = Image.fromarray(np.uint8(E_img), 'RGBA')
    #A_img.save(f"../figures/A-{filename}")
    #E_img.save(f"../figures/E-{filename}")
    counter = 0
    if os.path.exists("../figures/A-0.png"):
        while os.path.exists(f"../figures/A-{counter}.png"):
            counter += 1 
    A_img.save(f"../figures/A-{counter}.png")
    E_img.save(f"../figures/E-{counter}.png")
    
    return [A_img, E_img]
