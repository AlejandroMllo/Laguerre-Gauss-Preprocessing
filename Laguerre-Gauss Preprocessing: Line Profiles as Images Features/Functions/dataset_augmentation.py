import Augmentor

source_path = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Datasets/Mineria/64x64/samples/'
output_path = '/home/alejandro/Documents/Universidad/Semestre 6/PI1/Datasets/Mineria/64x64/samples_augmentor/'
p = Augmentor.Pipeline(source_path, output_directory=output_path, save_format='.png')
p.rotate90(probability=0.9)
p.rotate270(probability=0.9)
p.rotate(probability=0.9,  max_left_rotation=20, max_right_rotation=20)
p.flip_left_right(probability=0.9)
p.flip_top_bottom(probability=0.9)
# p.crop_random(probability=0.5, percentage_area=0.2)

samples = p.sample(20000)
