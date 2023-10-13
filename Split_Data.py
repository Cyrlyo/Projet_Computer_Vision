import os
from sklearn.model_selection import train_test_split
import glob
import shutil


def split_data(data_train_path: str, save_train_path: str, save_val_path: str, img_extension: str, split_size=0.2)\
        -> None:
    """

    :param data_train_path:
    :param save_train_path:
    :param save_val_path:
    :param img_extension:
    :param split_size:
    :return:
    """

    folders = os.listdir(data_train_path)

    for folder in folders:

        full_path = os.path.join(data_train_path, folder)
        images_path = glob.glob(full_path + img_extension)

        x_train, x_val = train_test_split(images_path, test_size=split_size)

        for x in x_train:

            folder_path = os.path.join(save_train_path, folder)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            shutil.copy(x, folder_path)

        for x in x_val:

            folder_path = os.path.join(save_val_path, folder)
            if not os.path.isdir(folder_path):
                os.makedirs(folder_path)

            shutil.copy(x, folder_path)

if __name__ == '__main__':

   #train_path = str(input("Enter train set path\t:"))
   #new_train_path = str(input("Enter the new train path (where you want your data)\t:"))
   #new_val_path = str(input("Enter the new validation path (where you want your validation data)\t:"))
    image_ext = "/*.jpg"

    split_data("../Images/seg_train/seg_train/", "../Images/train_set_raw/train_raw", "../Images/train_set_raw/val_raw", image_ext)