from PIL import Image
from glob import glob
import imageio

def png_to_jpeg(path):
    images_path = glob(f"{path}/*")
    for i, image_path in enumerate(images_path):
        image = Image.open(image_path).convert('RGBA')
        image = image.resize((64,64))
        new_image = Image.new("RGBA", (64,64), 'WHITE')
        new_image.paste(image, mask = image)
        new_image.convert('RGB').save(f'transformed_data/{i}.jpg', 'JPEG')
        
def save_as_gif(path):
    images_path = glob(f'{path}/*')
    anim_file = 'training_process.gif'

    with imageio.get_writer(anim_file, mode='I',duration=0.25) as writer:
        filenames = sorted(images_path)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
        image = imageio.imread(filename)
        writer.append_data(image)
    
    