import os
import math
import numpy as np
from functools import partial
import PIL.Image
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
from PIL import ImageFilter

import tensorflow as tf

import random
import cv2

tf.app.flags.DEFINE_string("model", "tensorflow_inception_graph.pb", "Model")
tf.app.flags.DEFINE_string("input", "", "Input Image (JPG)");
tf.app.flags.DEFINE_string("output", "output", "Output prefix");
tf.app.flags.DEFINE_string("layer", "import/mixed4c", "Layer name");  #4c
tf.app.flags.DEFINE_integer("feature", "-1", "Individual feature");
tf.app.flags.DEFINE_integer("cycles", "100", "How many cycles to run");
tf.app.flags.DEFINE_integer("octaves", "8", "How many mage octaves (scales)");
tf.app.flags.DEFINE_integer("iterations", "20", "How many gradient iterations per octave");
tf.app.flags.DEFINE_float("octave_scale", "1.4", "Octave scaling factor");
tf.app.flags.DEFINE_integer("tilesize", "512", "Size of tiles. Decrease if out of GPU memory. Increase if bad utilization.");

FLAGS = tf.app.flags.FLAGS

# creating TensorFlow session and loading the model
graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph, config=tf.ConfigProto(log_device_placement=False))
graph_def = tf.GraphDef.FromString(open(FLAGS.model).read())
t_input = tf.placeholder(np.float32, name='input') # define the input tensor
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input':t_preprocessed})

print "--- Available Layers: ---"
layers = []
for name in (op.name for op in graph.get_operations()):
  layer_shape = graph.get_tensor_by_name(name+':0').get_shape()
  if not layer_shape.ndims: continue
  layers.append((name, int(layer_shape[-1])))
  print name, "Features/Channels: ", int(layer_shape[-1])
print 'Number of layers', len(layers)
print 'Total number of feature channels:', sum((layer[1] for layer in layers))
print 'Chosen layer: '
print graph.get_operation_by_name(FLAGS.layer);

def T(layer):
    '''Helper for getting layer output tensor'''
    return graph.get_tensor_by_name("%s:0"%layer)

def tffunc(*argtypes):
    '''Helper that transforms TF-graph generating function into a regular one.
    See "resize" function below.
    '''
    placeholders = map(tf.placeholder, argtypes)
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)

def calc_grad_tiled(img, t_grad, tile_size=512):
    '''Compute the value of tensor t_grad over the image in a tiled way.
    Random shifts are applied to the image to blur tile boundaries over
    multiple iterations.'''
    sz = tile_size
    h, w = img.shape[:2]
    sx, sy = np.random.randint(sz, size=2)
    img_shift = np.roll(np.roll(img, sx, 1), sy, 0)
    grad = np.zeros_like(img)
    for y in xrange(0, max(h-sz//2, sz),sz):
        for x in xrange(0, max(w-sz//2, sz),sz):
            sub = img_shift[y:y+sz,x:x+sz]
            g = sess.run(t_grad, {t_input:sub})
            grad[y:y+sz,x:x+sz] = g
    return np.roll(np.roll(grad, -sx, 1), -sy, 0)

def render_deepdream(t_obj, img,
                     iter_n=10, step=1.5, octave_n=12, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj)
    t_grad = tf.gradients(t_score, t_input)[0]

    # split the image into a number of octaves
    img = img
    octaves = []
    for i in xrange(octave_n-1):
        hw = img.shape[:2]
        lo = resize(img, np.int32(np.float32(hw)/octave_scale**(0.9+random.random()*0.2)))
        hi = img-resize(lo, hw)
        img = lo
        octaves.append(hi)

    # generate details octave by octave
    for octave in xrange(octave_n):
        print " Octave: ", octave, "Res: ", img.shape
        if octave>0:
            hi = octaves[-octave]
            img = resize(img, hi.shape[:2])+hi
        for i in xrange(int(float(iter_n)/(octave_scale**(2*octave))+random.random())):
            g = calc_grad_tiled(img, t_grad, FLAGS.tilesize)
            img += g*(step / (np.abs(g).mean()+1e-7))
    return img

def main(_):
  img = np.float32(PIL.Image.open(FLAGS.input));
  # Make RGB if greyscale:
  if len(img.shape)==2 or img.shape[2] == 1:
    img = np.stack([img]*3, axis=2)

  design_img = my_img(img.shape)    
  for i_frame in range(FLAGS.cycles):
    print "Cycle", i_frame, " Res:", img.shape
    t_obj = tf.square(T(FLAGS.layer))
    if FLAGS.feature >= 0:
      t_obj = T(FLAGS.layer)[:,:,:,FLAGS.feature]
    img = blend(img, design_img)
    #cv2.imshow('image',resize(img, (650, 462))/255)
    #cv2.waitKey(0)
    print "Saving ", i_frame
    img = np.uint8(np.clip(img, 0, 255))
    PIL.Image.fromarray(img).save("%s_%05d.jpg"%(FLAGS.output, i_frame), "jpeg")
    img = render_deepdream(t_obj, img,
        iter_n = FLAGS.iterations,
        octave_n = FLAGS.octaves,
        octave_scale = FLAGS.octave_scale)

def blend(a, b):
  # a=deepdream
  # b=mask + alpha
  a = np.float32(a)
  b = np.float32(b)
  diff = (np.sum(np.square(a-b[...,:3])*b[...,3, None]) / np.sum(b[...,3, None]))**0.5
  
  blend_fact = max(0,((diff-60)/255))
  print blend_fact
  b_mask = blend_fact * b[...,3,None]/255
  out = a * (1-b_mask) + b[...,:3] * b_mask
  return out

def font_in_box(img, x, y, w, h, color, txt, fill=None, fontname='Roboto-Regular.ttf'):
  draw = ImageDraw.Draw(img)
  fontsize = 0
  while True:
    # iterate until the text size is just larger than the criteria
    fontsize += 1
    font = ImageFont.truetype(fontname, fontsize)
    size = draw.textsize(txt, font=font)
    if size[0] > w or size[1] > h:
      break

  fontsize -= 1
  font = ImageFont.truetype(fontname, fontsize)
  draw.text((x,y), txt, color, font=font)
  

def my_img(shape):

  txt = 'Oliver Mattos\nWinham Farm\nWestcott\nCullompton\nDevon\nEX15 1SA'
  img = Image.new("RGBA", (shape[1]*2, shape[0]*2), (255,255,255,0))
  
  
  #ImageDraw.Draw(img).rectangle([0,1900,2100,2970], fill=(255,255,255,0))
  #img = img.filter(PIL.ImageFilter.GaussianBlur(radius=80))
  font_in_box(img, 600, 2100, 1500, 800, (255,255,255,255), 'Oliver',fontname='Pacifico-Regular.ttf')
  ImageDraw.Draw(img).rectangle([150,540,1120,950], fill=(255,255,255,255))
  img = img.filter(PIL.ImageFilter.GaussianBlur(radius=25))
  font_in_box(img, 600, 2100, 1500, 800, (255,255,255,255), 'Oliver',fontname='Pacifico-Regular.ttf')
  img = img.filter(PIL.ImageFilter.GaussianBlur(radius=10))
  font_in_box(img, 600, 2100, 1500, 800, (0,0,0,255), 'Oliver',fontname='Pacifico-Regular.ttf')
  img = img.filter(PIL.ImageFilter.GaussianBlur(radius=4))
  font_in_box(img, 190, 570, 900, 350, (0,0,0,255), txt)
  img.save('template.png')
  return np.asarray(img.resize((shape[1], shape[0]), Image.ANTIALIAS))

if __name__ == "__main__":
  tf.app.run()
