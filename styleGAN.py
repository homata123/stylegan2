import tensorflow as tf 
print(tf.__version__)
#recommend using tf version 1.x
from IPython.display import Image 

#Kiểm tra shape của các network,nếu giống nhau thì mô hình mới hoạt động
from stylegan2 import pretrained_networks
_, _, Gs_1 = pretrained_networks.load_networks('network-snapshot-018528.pkl')
_, _, Gs_2 = pretrained_networks.load_networks('ukiyoe-256-slim-diffAug-002789.pkl')
_, _, Gs_3 = pretrained_networks.load_networks('stylegan2-ffhq.pkl')
_, _, Gs_4 = pretrained_networks.load_networks('2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl')
_, _, Gs_5 = pretrained_networks.load_networks('epic-slim-256-000040.pkl')

Gs_1.output_shape, Gs_2.output_shape,Gs_3.output_shape,Gs_4.output_shape,Gs_5.output_shape

#blending networks

from stylegan2 import blend_models

resolutions = (8, 16, 32, 64, 128)
for res in resolutions:
  filename = f"blended-{res}.jpg"
  # có thể tùy chọn 2 network có cùng shape để blend
  #blend_models.main("epic-slim-256-000040.pkl", "stylegan2-ffhq.pkl", res, output_grid=filename)
  #blend_models.main("/content/drive/MyDrive/styleGAN/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl", "/content/stylegan2/stylegan2-ffhq-config-f.pkl", res, output_grid=filename)
  blend_models.main("/content/stylegan2/ukiyoe-256-slim-diffAug-002789.pkl", "/content/stylegan2/stylegan2-ffhq.pkl", res, output_grid=filename)
  img = Image(filename=filename)
  print(f"blending at {res}x{res}")
  display(img)