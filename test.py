import os,time,cv2, sys, math
import tensorflow as tf
import argparse
import numpy as np

from utils import utils, helpers
from builders import model_builder

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', type=str, default=None, required=True, help='The path to the latest checkpoint weights for your model.')
parser.add_argument('--crop_height', type=int, default=512, help='Height of cropped input image to network')
parser.add_argument('--crop_width', type=int, default=512, help='Width of cropped input image to network')
parser.add_argument('--model', type=str, default=None, required=True, help='The model you are using')
parser.add_argument('--dataset', type=str, default="CamVid", required=False, help='The dataset you are using')
args = parser.parse_args()

# Get the names of the classes so we can record the evaluation results
print("Retrieving dataset information ...")
class_names_list, label_values = helpers.get_label_info(os.path.join(args.dataset, "class_dict.csv"))
class_names_string = ""
for class_name in class_names_list:
    if not class_name == class_names_list[-1]:
        class_names_string = class_names_string + class_name + ", "
    else:
        class_names_string = class_names_string + class_name

num_classes = len(label_values)

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
target.write("test_name, accuracy, balanced_accuracy, precision, recall, f1, iou, %s\n" % (class_names_string))
scores_list = []
balanced_scores_list = []
class_scores_list = []
precision_list = []
recall_list = []
f1_list = []
iou_list = []
run_times_list = []

# Run testing on ALL test images
for ind in range(len(test_input_names)):
    sys.stdout.write("\rRunning test image %d / %d"%(ind+1, len(test_input_names)))
    sys.stdout.flush()

    input_image = np.expand_dims(np.float32(utils.load_image(test_input_names[ind])[:args.crop_height, :args.crop_width]),axis=0)/255.0
    gt = utils.load_image(test_output_names[ind])[:args.crop_height, :args.crop_width]
    gt = helpers.reverse_one_hot(helpers.one_hot_it(gt, label_values))

    st = time.time()
    output_image = sess.run(network,feed_dict={net_input:input_image})

    run_times_list.append(time.time()-st)

    output_image = np.array(output_image[0,:,:,:])
    output_image = helpers.reverse_one_hot(output_image)
    out_vis_image = helpers.colour_code_segmentation(output_image, label_values)

    accuracy, balanced_accuracy, class_accuracies, prec, rec, f1, iou = utils.evaluate_segmentation(pred=output_image, label=gt, num_classes=num_classes)

    file_name = utils.filepath_to_name(test_input_names[ind])
    target.write("%s, %f, %f, %f, %f, %f, %f"%(file_name, accuracy, balanced_accuracy, prec, rec, f1, iou))
    for item in class_accuracies:
        target.write(", %f"%(item))
    target.write("\n")

    scores_list.append(accuracy)
    balanced_scores_list.append(balanced_accuracy)
    class_scores_list.append(class_accuracies)
    precision_list.append(prec)
    recall_list.append(rec)
    f1_list.append(f1)
    iou_list.append(iou)
    
    gt = helpers.colour_code_segmentation(gt, label_values)

    os.symlink(test_input_names[ind], "%s/%s"%("Test", os.path.basename(test_input_names[ind])))
    cv2.imwrite("%s/%s_pred.png"%("Test", file_name),cv2.cvtColor(np.uint8(out_vis_image), cv2.COLOR_RGB2BGR))
    cv2.imwrite("%s/%s_gt.png"%("Test", file_name),cv2.cvtColor(np.uint8(gt), cv2.COLOR_RGB2BGR))


avg_score = np.mean(scores_list)
avg_balanced_score = np.mean(balanced_scores_list)
class_avg_scores = np.mean(class_scores_list, axis=0)
avg_precision = np.mean(precision_list)
avg_recall = np.mean(recall_list)
avg_f1 = np.mean(f1_list)
avg_iou = np.mean(iou_list)
avg_time = np.mean(run_times_list)

target.write("%s, %f, %f, %f, %f, %f, %f"%("Average", avg_score, avg_balanced_score, avg_precision, avg_recall, avg_f1, avg_iou))
for item in class_avg_scores:
    target.write(", %f"%(item))
target.write("\n")
target.close()

print(f"Accuracy = {avg_score}")
print(f"Balanced accuracy = {avg_balanced_score}")
print(f"Class accuracies:")
for index, item in enumerate(class_avg_scores):
    print("\t%s = %f" % (class_names_list[index], item))
print(f"Precision = {avg_precision}")
print(f"Recall = {avg_recall}")
print(f"F1 = {avg_f1}")
print(f"IoU = {avg_iou}")
print(f"Average run time = {avg_time}")
