# C3D-tensorflow

## Repeat the experiment result:

1. Install the following two python libs:
    
    a) [tensorflow][1](version r1.0)

    b) [Pillow][2]

    c) [Opencv][7]
2. Download the [UCF101][3] (Action Recognition Data Set)
3. Extract the `UCF101.rar` file and you will get `UCF101/{action_name}/{video.avi}` folder structure
4. Use the `./list/convert_video_to_images.sh` script to decode the ucf101 video files (from video to images)
    - run `./list/convert_video_to_images.sh .../UCF101 5` (number `5` means the fps rate)
5. Create the `train.list` and `test.list` files in `list` directory. 
6. Use the `./list/convert_images_to_list.sh` script to update the `{train,test}.list` according to the `UCF101` folder structure (from images to files)
    - run `./list/convert_images_to_list.sh .../UCF101 4`, this will update the `test.list` and `train.list` files (number `4` means the ratio of test and train data is 1/4)

    ```
    database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c01 0
    database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c02 0
    database/ucf101/train/ApplyEyeMakeup/v_ApplyEyeMakeup_g01_c03 0
    database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c01 1
    database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c02 1
    database/ucf101/train/ApplyLipstick/v_ApplyLipstick_g01_c03 1
    database/ucf101/train/Archery/v_Archery_g01_c01 2
    database/ucf101/train/Archery/v_Archery_g01_c02 2
    database/ucf101/train/Archery/v_Archery_g01_c03 2
    database/ucf101/train/Archery/v_Archery_g01_c04 2
    database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c01 3
    database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c02 3
    database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c03 3
    database/ucf101/train/BabyCrawling/v_BabyCrawling_g01_c04 3
    database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c01 4
    database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c02 4
    database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c03 4
    database/ucf101/train/BalanceBeam/v_BalanceBeam_g01_c04 4
    ...
    ```
7. Run the training program `python train_c3d_gpu.py` (you can pause or stop the training procedure and resume the training by runing this command again)

8. Evaluate the result `python eval_c3d.py`

## Option

### Use the pretrained model
If you want to test the pre-trained model (sports1m), you need to download the model from here: https://www.dropbox.com/sh/8wcjrcadx4r31ux/AAAkz3dQ706pPO8ZavrztRCca?dl=0 and move the file `sports1m_finetuning_ucf101.model` to the `root` folder

### Use other dataset than UCF101
1. modify the `NUM_CLASSES` variable in the `c3d_model.py` file
2. change the `NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN` and `NUM_EXAMPLES_PER_EPOCH_FOR_EVAL` variables in the `c3d_model.py` file
    - `NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN` = (total number of training image)/`NUM_FRAMES_PER_CLIP`
    - `NUM_EXAMPLES_PER_EPOCH_FOR_EVAL` = (total number of evaluating image)/`NUM_FRAMES_PER_CLIP`
    - `NUM_FRAMES_PER_CLIP` is in the `c3d_model.py` file

## Experiment result:

Top-1 accuracy of 80% should be achieved for the validation dataset with this code

![image](/images/ucf101_result.png)

## References:

- Thanks the author [Du tran][4]'s code: [C3D-caffe][5]
- [C3D: Generic Features for Video Analysis][6]
- [frankgu][8]'s contribution

[1]: https://www.tensorflow.org/
[2]: http://pillow.readthedocs.io/en/3.1.x/reference/Image.html
[3]: http://crcv.ucf.edu/data/UCF101/UCF101.rar
[4]: https://github.com/dutran
[5]: https://github.com/facebook/C3D
[6]: http://vlg.cs.dartmouth.edu/c3d/
[7]: http://opencv.org/
[8]: https://github.com/frankgu