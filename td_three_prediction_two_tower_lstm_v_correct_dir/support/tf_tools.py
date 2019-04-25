import tensorflow as tf
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

latest_ckp = tf.train.latest_checkpoint(
    '/cs/oschulte/Galen/soccer-models/hybrid_sl_saved_NN/Scale-tt-three-cut_together_saved_networks_feature5_batch64_iterate30_lr0.0001_v4_v_correct__MaxTL10')
print_tensors_in_checkpoint_file(latest_ckp, tensor_name='', all_tensors=True)
