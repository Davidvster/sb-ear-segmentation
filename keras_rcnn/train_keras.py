import os
# set os.environ['CUDA_VISIBLE_DEVICES'] = '0' if you wish to execute on a CUDA GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from numpy import zeros
from numpy import asarray
import glob
from mrcnn.config import Config
from mrcnn.model import MaskRCNN
from mrcnn.utils import Dataset
import warnings
warnings.filterwarnings("ignore")


# class that defines and loads the ear dataset
class EarsDataset(Dataset):
    # load the dataset definitions
    def load_dataset(self, is_train=True):
        # define one class
        self.add_class("dataset", 1, "ear")
        # define data locations
        for im_path in glob.glob("../AWEForSegmentation/train/*.png"):
            im_name = im_path.split("../AWEForSegmentation/train\\")[1].replace(".png", "")
            if is_train and int(im_name) >= 500:
                continue
            if not is_train and int(im_name) < 500:
                continue
            self.add_image('dataset', image_id=im_name, path=im_path,
                           annotation="annotations_keras/" + im_name + ".txt")

    # extract bounding boxes from an annotation file
    def extract_boxes(self, filename):
        annotations = open(filename, 'r')
        width, height = annotations.readline().split(" ")
        lines = annotations.readlines()
        boxes = list()
        for line in lines:
            x_min, y_min, x_max, y_max = line.split(" ")
            coors = [int(x_min), int(y_min), int(x_max), int(y_max)]
            boxes.append(coors)
        return boxes, int(width), int(height)

    # load the masks for an image
    def load_mask(self, image_id):
        # get details of image
        info = self.image_info[image_id]
        # define box file location
        path = info['annotation']
        # load XML
        boxes, w, h = self.extract_boxes(path)
        # create one array for all masks, each on a different channel
        masks = zeros([h, w, len(boxes)], dtype='uint8')
        # create masks
        class_ids = list()
        for i in range(len(boxes)):
            box = boxes[i]
            row_s, row_e = box[1], box[3]
            col_s, col_e = box[0], box[2]
            masks[row_s:row_e, col_s:col_e, i] = 1
            class_ids.append(self.class_names.index('ear'))
        return masks, asarray(class_ids, dtype='int32')

    # load an image reference
    def image_reference(self, image_id):
        info = self.image_info[image_id]
        return info['path']


class EarConfig(Config):
    # Give the configuration a recognizable name
    NAME = "ear_cfg"
    # Number of classes (background + kangaroo)
    NUM_CLASSES = 1 + 1
    # Number of training steps per epoch
    STEPS_PER_EPOCH = 100
    # has to be 1, since for detecting we are detecting one image at a time
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    # train set
    train_set = EarsDataset()
    train_set.load_dataset(is_train=True)
    train_set.prepare()
    print('Train: %d' % len(train_set.image_ids))

    # test/val set
    test_set = EarsDataset()
    test_set.load_dataset(is_train=False)
    test_set.prepare()
    print('Test: %d' % len(test_set.image_ids))

    config = EarConfig()
    config.display()
    # define the model
    model = MaskRCNN(mode='training', model_dir='./', config=config)
    # load weights (mscoco) and exclude the output layers
    # When starting, load the mask_rcnn_coco5 pretrained model since it can speed up the initial process
    model.load_weights('mask_rcnn_coco.h5', by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    # When we already have our own model, we can continue training it
    # model.load_weights('mask_rcnn_ear_cfg_0011.h5', by_name=True)
    # train weights (output layers or 'heads')
    model.train(train_set, test_set, learning_rate=config.LEARNING_RATE, epochs=6, layers='heads')
