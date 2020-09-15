import models.resunet as resunet
import models.res16unet as res16unet
import models.fcgfnet as fcgfnet

# from models.trilateral_crf import TrilateralCRF
from models.conditional_random_fields import BilateralCRF, TrilateralCRF

MODELS = []


def add_models(module):
  MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])


add_models(resunet)
add_models(res16unet)
add_models(fcgfnet)

WRAPPERS = [BilateralCRF, TrilateralCRF]


def get_models():
  '''Returns a tuple of sample models.'''
  return MODELS


def get_wrappers():
  return WRAPPERS


def load_model(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_models = get_models()
  mdict = {model.__name__: model for model in all_models}
  if name not in mdict:
    print('Invalid model index. Options are:')
    # Display a list of valid model names
    for model in all_models:
      print('\t* {}'.format(model.__name__))
    return None
  NetClass = mdict[name]

  return NetClass


def load_wrapper(name):
  '''Creates and returns an instance of the model given its class name.
  '''
  # Find the model class from its name
  all_wrappers = get_wrappers()
  mdict = {wrapper.__name__: wrapper for wrapper in all_wrappers}
  if name not in mdict:
    print('Invalid wrapper index. Options are:')
    # Display a list of valid model names
    for wrapper in all_wrappers:
      print('\t* {}'.format(wrapper.__name__))
    return None
  WrapperClass = mdict[name]

  return WrapperClass
