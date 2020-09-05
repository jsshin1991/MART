import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Images.InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions, \
    top_label_id_and_score
from Images.run_superpixel_MART import run_superpixel_MART
from Images.VisualizationLibrary.visualization_lib import Visualize, pil_image

MODEL_LOC = './Images/InceptionModel/tensorflow_inception_graph.pb'
LABELS_LOC = './Images/InceptionModel/imagenet_comp_graph_label_strings.txt'

img_arr_list = ['goldfinch', 'killer_whale']
img_arr = np.array([])
for img in img_arr_list:
    img_arr = np.append(img_arr, img)

# Load the Inception model.
sess, graph = load_model(MODEL_LOC)

# Load the Labels vocabulary.
labels = load_labels_vocabulary(LABELS_LOC)

# Make the predictions_and_gradients function
inception_predictions = make_predictions(sess, graph)


# Load the image.
def load_image(img_path):
    with tf.gfile.FastGFile(img_path, 'rb') as f:
        img = f.read()
        img = sess.run(tf.image.decode_jpeg(img))
        return img


# # Run superpixel MART
# for img_name in img_arr:
#     img = load_image('./Images/test_data/'+img_name+'.jpg')
#
#     # Determine top label and score.
#     top_label_id, score = top_label_id_and_score(img, inception_predictions)
#     print(img_name)
#     print("Top label: %s, score: %f" % (labels[top_label_id], score))
#
#     mart = run_superpixel_MART(input=img, target_label_index=top_label_id, predictions=inception_predictions, iter=5, n_segments=500)
#     np.save('./Images/results/'+img_name+'_arr', mart)

# Visualization
img_vis = 'fcd9bbea9f6f5c4a'

img = load_image('./Images/test_data/' + img_vis + '.jpg')
_, ax = plt.subplots(1, 3)
ax[0].imshow(img)

mart = np.load('./Images/results/' + img_vis + '_arr.npy')
scaler = np.multiply(np.sqrt(np.var(mart)), 5e+5)

orig_vis, tmp_vis = Visualize(
    mart, img,
    scaling=scaler,
    upper_threshold=0.9,
    lower_threshold=0.2)

ax[1].imshow(pil_image(orig_vis))
ax[2].imshow(pil_image(tmp_vis))

for idx in range(3):
    plt.setp(ax[idx].get_xticklabels(), visible=False)
    plt.setp(ax[idx].get_yticklabels(), visible=False)
    ax[idx].tick_params(axis='both', which='both', length=0)

plt.show()
