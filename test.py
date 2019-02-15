import os,time,cv2, sys, math, datetime
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from utils.utils import list_to_string as l2s
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
parser.add_argument('--metric_average', type=str, default="weighted", required=False, help='The average to apply for the performance metrics as used by sklearn.metrics.precision_recall_fscore_support. One of ["macro", "micro", "weighted"].')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
num_classes = len(class_names)

# Initializing network
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess=tf.Session(config=config)

net_input = tf.placeholder(tf.float32,shape=[None,None,None,3])
net_output = tf.placeholder(tf.float32,shape=[None,None,None,num_classes]) 

network, _ = model_builder.build_model(args.model, net_input=net_input, num_classes=num_classes, crop_width=args.crop_width, crop_height=args.crop_height, is_training=False)

sess.run(tf.global_variables_initializer())

print('Loading model checkpoint weights ...')
saver=tf.train.Saver(max_to_keep=1000)
saver.restore(sess, args.checkpoint_path)

# Load the data
print("Loading the data ...")
train_input_names,train_output_names, val_input_names, val_output_names, test_input_names, test_output_names = utils.prepare_data(dataset_dir=args.dataset)

# Create directories if needed
if not os.path.isdir("%s"%("Test")):
        os.makedirs("%s"%("Test"))

target=open("%s/test_scores.csv"%("Test"),'w')
target.write("test_name, global_accuracy, balanced_accuracy, iou, precision, recall, f1, support, run_time\n")

global_accuracy_list = []
balanced_accuracy_list = []
iou_list = []
precision_list = []
recall_list = []
f1_list = []
support_list = []
run_time_list = []

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})
    run_time = time.time() - st
    run_time_list.append(run_time)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    global_accuracy, balanced_accuracy, iou, precision, recall, f1, support = utils.evaluate_segmentation(output_image, gt, args.metric_average)

    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %f, %f, %s, %s, %s, %s, %s, %f\n" % (file_name, global_accuracy, balanced_accuracy, l2s(iou), l2s(precision), l2s(recall), l2s(f1), l2s(support), run_time))

    global_accuracy_list.append(global_accuracy)
    balanced_accuracy_list.append(balanced_accuracy)
    iou_list.append(iou)
    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    support_list.append(support)

    gt = helpers.colour_code_segmentation(gt, label_values)

    os.symlink(test_input_names[ind], "%s/%s"%("Test", os.path.basename(test_input_names[ind])))
    cv2.imwrite("%s/%s_pred.png"%("Test", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


avg_global_accuracy = np.mean(global_accuracy_list)
avg_balanced_accuracy = np.mean(balanced_accuracy_list)
avg_iou = np.mean(iou_list, axis=0)
avg_precision = np.mean(precision_list, axis=0)
avg_recall = np.mean(recall_list, axis=0)
avg_f1 = np.mean(f1_list, axis=0)
avg_support = np.mean(support_list, axis=0)
avg_run_time = np.mean(run_time_list)

target.write("%s, %f, %f, %s, %s, %s, %s, %s, %f\n" % ("Average", avg_global_accuracy, avg_balanced_accuracy, l2s(avg_iou), l2s(avg_precision), l2s(avg_recall), l2s(avg_f1), l2s(avg_support), avg_run_time))
target.close()

print('\n')
print(f"Global accuracy = {avg_global_accuracy:%}")
print(f"Balanced accuracy = {avg_balanced_accuracy:%}")
print(f"IoU = {avg_iou[0]:%}")
for index, item in enumerate(avg_iou[1:]):
    print(f" * {class_names[index]} = {item:%}")
print(f"Precision = {avg_precision[0]:%}")
for index, item in enumerate(avg_precision[1:]):
    print(f" * {class_names[index]} = {item:%}")
print(f"Recall = {avg_recall[0]:%}")
for index, item in enumerate(avg_recall[1:]):
    print(f" * {class_names[index]} = {item:%}")
print(f"F1 = {avg_f1[0]:%}")
for index, item in enumerate(avg_f1[1:]):
    print(f" * {class_names[index]} = {item:%}")
print(f"Support = {avg_support[0]:%}")
for index, item in enumerate(avg_support[1:]):
    print(f" * {class_names[index]} = {item:%}")
print(f"Run time = {datetime.timedelta(seconds=avg_run_time)}")
