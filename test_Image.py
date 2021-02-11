import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from Images.InceptionModel.inception_utils import load_model, load_labels_vocabulary, make_predictions, \
    top_label_id_and_score, comp_score
from Images.run_superpixel_MART import run_superpixel_MART
from Images.VisualizationLibrary.visualization_lib import Visualize, pil_image, ConvertToGrayscale

MODEL_LOC = './Images/InceptionModel/tensorflow_inception_graph.pb'
LABELS_LOC = './Images/InceptionModel/imagenet_comp_graph_label_strings.txt'

img_arr_list = ['goldfinch', 'killer_whale']
img_arr = np.array([])
for img in img_arr_list:
    img_arr = np.append(img_arr, img)

#### Load the Inception model ####
sess, graph = load_model(MODEL_LOC)

#### Load the Labels vocabulary ####
labels = load_labels_vocabulary(LABELS_LOC)

#### Make the predictions_and_gradients function ####
inception_predictions = make_predictions(sess, graph)

#### Load the image ####
def load_image(img_path):
    with tf.gfile.FastGFile(img_path, 'rb') as f:
        img = f.read()
        img = sess.run(tf.image.decode_jpeg(img))
        return img

#### Run superpixel MART ####
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

import random
import os
os.getcwd()

folder_name = './Images/test_data/'
abs_dir = os.path.join(os.getcwd(), folder_name)

img_vis_list = os.listdir(abs_dir)

#### Evaluation ####
percentage_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
target_list = ['mart', 'eig', 'ig']
random.seed(0)
#### Set num_seeds = 1 for Mean Masking and set others for Rand_Same and Rand_Diff Masking ####
num_seeds = 5
seed_list = [random.randint(0, 1e5) for i in range(num_seeds)]
num_img = len(img_vis_list)

#### KIP ####
for target in target_list:
    print("Target: " + target)
    for percentage in percentage_list:
        avg_score = np.array([])
        for seed in seed_list:
            score_diff = np.array([])
            for img_vis in img_vis_list:

                img_vis = img_vis[:-4]
                img = load_image('./Images/test_data/' + img_vis + '.jpg' )

                if target == 'mart':
                    mart = np.load('./Images/results/mart/' + img_vis + '_arr.npy')
                    avg_attr = ConvertToGrayscale(mart)
                elif target == 'eig':
                    eig = np.load('./Images/results/eig/' + img_vis + '_arr.npy')
                    avg_attr = ConvertToGrayscale(eig)
                elif target == 'ig':
                    ig = np.load('./Images/results/ig/' + img_vis + '_arr.npy')
                    avg_attr = ConvertToGrayscale(ig)
                else:
                    print("Error")

                orig_top_label_id, orig_score = top_label_id_and_score(img, inception_predictions)
                imp_sorted = np.dstack(np.unravel_index(np.argsort(avg_attr.ravel()), (224, 224)))[0]
                masking = np.flip(imp_sorted, axis=0)[int(224 * 224 * percentage):]

                #### Mean Masking ####
                # avg_rgb = np.array([0, 0, 0])
                # for idx in masking:
                #     for c in range(3):
                #         avg_rgb[c] += img[idx[0], idx[1], c]
                # avg_rgb = avg_rgb / masking.shape[0]
                # avg_rgb = np.average(np.average(img, axis=0), axis=0)
                #
                # for c in range(3):
                #     for idx in masking:
                #         img[idx[0], idx[1], c] = int(avg_rgb[c])

                #### Rand_Diff Masking ####
                # random.seed(seed)
                # for c in range(3):
                #     for idx in masking:
                #         img[idx[0], idx[1], c] = random.randint(0, 255)

                #### Rand_Same Masking ####
                random.seed(seed)
                mask_value = random.randint(0, 255)
                for idx in masking:
                    img[idx[0], idx[1], :] = mask_value

                mod_score = comp_score(img, inception_predictions, orig_top_label_id)
                score_diff = np.append(score_diff, max(orig_score - mod_score, 0))

            avg_score = np.append(avg_score, 100 * np.average(score_diff))
        print(str(int(100 * percentage)) + "%: " + str(np.average(avg_score)))

#### MPN ####
# percentage_list = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
#
# for target in target_list:
#     print("Target: " + target)
#     equal_label = np.array([])
#     for img_vis in img_vis_list:
#
#         img_vis = img_vis[:-4]
#         img = load_image('./Images/test_data/' + img_vis + '.jpg' )
#
#         if target == 'mart':
#             mart = np.load('./Images/results/mart/' + img_vis + '_arr.npy')
#             avg_attr = ConvertToGrayscale(mart)
#         elif target == 'eig':
#             eig = np.load('./Images/results/eig/' + img_vis + '_arr.npy')
#             avg_attr = ConvertToGrayscale(eig)
#         elif target == 'ig':
#             ig = np.load('./Images/results/ig/' + img_vis + '_arr.npy')
#             avg_attr = ConvertToGrayscale(ig)
#         else:
#             print("Error")
#
#         orig_top_label_id, orig_score = top_label_id_and_score(img, inception_predictions)
#
#     #### Mean Masking ####
#     #     for percentage in percentage_list:
#     #         img = load_image('./Images/test_data/' + img_vis + '.jpg' )
#     #         imp_sorted = np.dstack(np.unravel_index(np.argsort(avg_attr.ravel()), (224, 224)))[0]
#     #         masking = np.flip(imp_sorted, axis=0)[int(224 * 224 * percentage):]
#     #         # masking = np.flip(imp_sorted, axis=0)[:int(224 * 224 * percentage)]
#     #
#     #         avg_rgb = np.array([0, 0, 0])
#     #         for idx in masking:
#     #             for c in range(3):
#     #                 avg_rgb[c] += img[idx[0], idx[1], c]
#     #         avg_rgb = avg_rgb / masking.shape[0]
#     #         # avg_rgb = np.average(np.average(img, axis=0), axis=0)
#     #
#     #         for c in range(3):
#     #             for idx in masking:
#     #                 img[idx[0], idx[1], c] = int(avg_rgb[c])
#     #
#     #         mod_top_label_id, mod_score = top_label_id_and_score(img, inception_predictions)
#     #         if mod_top_label_id == orig_top_label_id:
#     #             equal_label = np.append(equal_label, percentage)
#     #             break
#     #         elif percentage == 0.95:
#     #             equal_label = np.append(equal_label, 1.0)
#     # print(str(int(100 * percentage)) + "%: " +  str(100 * np.average(equal_label)))
#
#     #### Rand_Same / Rand_Diff ####
#         avg_tmp_label = np.array([])
#         for seed in seed_list:
#             for percentage in percentage_list:
#                 img = load_image('./Images/test_data/' + img_vis + '.jpg' )
#                 imp_sorted = np.dstack(np.unravel_index(np.argsort(avg_attr.ravel()), (224, 224)))[0]
#                 masking = np.flip(imp_sorted, axis=0)[int(224 * 224 * percentage):]
#                 # masking = np.flip(imp_sorted, axis=0)[:int(224 * 224 * percentage)]
#
#                 #### Rand_Same Masking ####
#                 # random.seed(seed)
#                 # mask_value = random.randint(0, 255)
#                 # for idx in masking:
#                 #     img[idx[0], idx[1], :] = mask_value
#
#                 #### Rand_Diff Masking ####
#                 random.seed(seed)
#                 for c in range(3):
#                     for idx in masking:
#                         img[idx[0], idx[1], c] = random.randint(0, 255)
#
#                 mod_top_label_id, mod_score = top_label_id_and_score(img, inception_predictions)
#                 if mod_top_label_id == orig_top_label_id:
#                     avg_tmp_label = np.append(avg_tmp_label, percentage)
#                     break
#                 elif percentage == 0.95:
#                     avg_tmp_label = np.append(avg_tmp_label, 1.0)
#
#         equal_label = np.append(equal_label, 100 * np.average(avg_tmp_label))
#     print(np.average(equal_label))



#### Visualization ####
# img_vis = img_vis_list[0][:-4]
# img = load_image('./Images/test_data/' + img_vis + '.jpg')
# _, ax = plt.subplots(1, 3)
# ax[0].imshow(img)
#
# mart = np.load('./Images/results/mart/' + img_vis + '_arr.npy')
# scaler = np.multiply(np.sqrt(np.var(mart)), 5e+5)
#
# orig_vis, tmp_vis = Visualize(
#     mart, img,
#     scaling=scaler,
#     upper_threshold=0.9,
#     lower_threshold=0.2)
#
# ax[1].imshow(pil_image(orig_vis))
# ax[2].imshow(pil_image(tmp_vis))
#
# for idx in range(3):
#     plt.setp(ax[idx].get_xticklabels(), visible=False)
#     plt.setp(ax[idx].get_yticklabels(), visible=False)
#     ax[idx].tick_params(axis='both', which='both', length=0)
#
# plt.show()
