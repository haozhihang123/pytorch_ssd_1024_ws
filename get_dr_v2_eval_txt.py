from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms.functional as FT
from datasets import PascalVOCDataset
from torch.autograd import Variable
from torchvision import transforms
import matplotlib.pyplot as plt
from ipdb import set_trace
from tqdm import tqdm
import numpy as np
from utils import *
import warnings
import shutil
import torch
import os

warnings.filterwarnings("ignore")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#新建存储路径
if not os.path.exists("./input"):
    os.makedirs("./input")
if not os.path.exists("./input/detection-results"):
    os.makedirs("./input/detection-results")
if not os.path.exists("./input/images-optional"):
    os.makedirs("./input/images-optional")

# 加载待检测列表
TEST_images_dir = 'json/TEST_images.json'
with open(os.path.join(TEST_images_dir), 'r') as j:
    TEST_images = json.load(j)

# Parameters
data_folder = './json'
keep_difficult = True  # difficult ground truth objects must always be considered in mAP calculation, because these objects DO exist!
# 为了保存检测结果到txt中，目前batch_size只能设置为１
batch_size = 1
workers = 1
checkpoint = './checkpoint_ssd300.pth.tar'

# Load model checkpoint that is to be evaluated
checkpoint = torch.load(checkpoint)
model = checkpoint['model']
model = model.to(device)

# Switch to eval mode
model.eval()

# Load test data
test_dataset = PascalVOCDataset(data_folder,
                                split='test',
                                keep_difficult=keep_difficult)

# for image, boxes, labels, difficulties in test_dataset:
#     img_np = image.numpy()
#     img = img_np.transpose([1,2,0])
#     img = (img - np.min(img))/(np.max(img) - np.min(img)) *255.0
#     img = img.astype(int)
#     cv2.imwrite('test.jpg', img)
#     set_trace()

test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,collate_fn=test_dataset.collate_fn, num_workers=workers, pin_memory=True)

def evaluate(test_loader, model):
    """
    Evaluate.

    :param test_loader: DataLoader for test data
    :param model: model
    """

    # Make sure it's in eval mode
    model.eval()

    # Lists to store detected and true boxes, labels, scores
    det_boxes = list()
    det_labels = list()
    det_scores = list()
    true_boxes = list()
    true_labels = list()
    true_difficulties = list()  # it is necessary to know which objects are 'difficult', see 'calculate_mAP' in utils.py

    with torch.no_grad():
        # Batches
        for i, (images, boxes, labels, difficulties) in enumerate(tqdm(test_loader, desc='Evaluating')):
            image_show = images
            images = images.to(device)  # (N, 3, 300, 300)
            # print(image_show.shape)
            # show_train_pic(image_show,boxes,i)
            # set_trace()

            # Forward prop.
            predicted_locs, predicted_scores = model(images)

            # Detect objects in SSD output
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(predicted_locs, predicted_scores,min_score=0.5, max_overlap=0.1,top_k=200)
            # print('label:',det_labels_batch)
            # print('score:',det_scores_batch)
            image_dir = TEST_images[i]
            show_eval_pic(image_dir,det_boxes_batch,i,det_labels_batch,det_scores_batch)
            # Evaluation MUST be at min_score=0.01, max_overlap=0.45, top_k=200 for fair comparision with the paper's results and other repos
            
            # 如果未检测到任何结果，不保存任何信息
            if len(det_labels_batch) == 1 and det_labels_batch[0][0].item()==0:
                # print('no box!')
                continue

            # 保存检测结果图像
            if os.path.exists(TEST_images[i]):
                target =  "./input/images-optional"
                shutil.copy(TEST_images[i], target)
                # print("save pic:",TEST_images[i].split('/')[-1])
            # 保存检测结果txt
            image_name = TEST_images[i].split('/')[-1].split('.')[0]
            f = open("./input/detection-results/"+image_name+".txt","a") 
            for i in range(len(det_boxes_batch[0])):
                f.write("%s %s %s %s %s %s\n" % (rev_label_map[det_labels_batch[0][i].item()] , str(det_scores_batch[0][i].item())[:6], str(int(det_boxes_batch[0][i][0].item()*1024)), str(int(det_boxes_batch[0][i][1].item()*1024)), str(int(det_boxes_batch[0][i][2].item()*1024)),str(int(det_boxes_batch[0][i][3].item()*1024))))
            f.close()
            # print('{} finish!'.format(image_name))
        print("Conversion completed!")
    
if __name__ == '__main__':
    evaluate(test_loader, model)
    