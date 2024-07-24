# Working Towards Skin Cancer Diagnosis (NVIDIA Final Project)

For my final project, I decided to utilize Rev18 and image detection techniques to classify seven different types of skin cancer. The dataset I employed for this project is the HAM10000 ("Human Against Machine with 10000 training images") dataset extracted from Kaggle (See link bellow), which addresses the challenges posed by the small size and lack of diversity in available dermatoscopic image datasets. My data set dataset comprises 10,015 dermatoscopic images collected from various populations and acquired using different modalities (Over 50% of the lesions in the dataset are confirmed through histopathology), which I then devided evenly to seperate training and test dat. This dataset serves as an academic training set for machine learning purposes, encompassing representative cases of all significant diagnostic categories in pigmented lesions. 

The seven types of skin cancer classified in this project are:
1. Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
2. Basal cell carcinoma (bcc)
3. Benign keratosis-like lesions (bkl)
4. Dermatofibroma (df)
5. Melanoma (mel)
6. Melanocytic nevi (nv)
7. Vascular lesions (vasc)

Neural networks hold immense potential for advancing healthcare by automating complex diagnostic tasks and improving the accuracy of early disease detection. This project is crucial in the real world as it addresses the need for scalable and reliable diagnostic tools in dermatology. Skin cancer, if detected early, can be treated effectively, reducing mortality rates and healthcare costs. By developing an automated system to classify skin cancer types accurately, we can facilitate early intervention, enhance patient outcomes, and contribute to the overall improvement of healthcare delivery.

![Access To The Original Skin Cancer Image Dataset](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000?select=HAM10000_metadata.csv)

![Skinc Cancer Model](https://github.com/user-attachments/assets/ec6b747f-2ab6-4b6f-9860-e64a2ea4a6ec)

![Training Process](https://github.com/user-attachments/assets/903a101b-5250-466e-92b4-aa414622968d)

![Training Process pt. 2](https://github.com/user-attachments/assets/04d10b07-89a5-4ab0-9161-b52b4cfb5d28)


## The Algorithm

Neural networks are a class of machine learning algorithms inspired by the structure and function of the human brain. They consist of interconnected layers of nodes (neurons), where each node processes input and passes it to the next layer. Convolutional Neural Networks (CNNs) are a specialized type of neural network designed for image processing tasks. They use convolutional layers to automatically and adaptively learn spatial hierarchies of features from input images. In this project, I created a Python script to organize the dataset. Initially, the images were listed in a CSV file with their respective categories, but they were not separated into folders. My code divided the images into individual folders for each category and further split them into training, validation, and test sets to ensure robust model training and evaluation. Using my NVIDIA Jetson Nano, I trained an image detection model. I utilized a mix of Python and Linux to run these processes efficiently on the Jetson Nano. The training involved using the ImageNet module from the Jetson Inference library. I trained the model by varying the number of epochs, workers, and batch sizes to optimize accuracy, ultimately achieving an accuracy of 99%. The core of my model is the RevNet18 architecture, a type of CNN with 18 convolutional layers. These layers perform convolutions, which are mathematical operations that merge two sets of information. In this context, they help the model learn to detect and recognize features such as edges, textures, and patterns in the skin lesion images, enabling accurate classification. Neural networks, particularly CNNs, are revolutionizing healthcare by automating complex diagnostic tasks and improving the accuracy of early disease detection. This project is crucial in the real world as it addresses the need for scalable and reliable diagnostic tools in dermatology. Early detection of skin cancer can significantly reduce mortality rates and healthcare costs. By developing an automated system to classify skin cancer types accurately, we can facilitate early intervention, enhance patient outcomes, and contribute to the overall improvement of healthcare delivery.

## Running this project
1. First, begin by setting up an SSH conection with your Jetson Nano and opening a functioning terminal.
  
2. Use cd commands to change directories until you are in your jetson-inference/python/training/classification/data
   
`cd jetson-inference/python/training/classification/data`

3. Run this command to download the image dataset.

`wget  PUT LINK HERE ISA`

4. cd back to 'classification' directory, and then cd into the 'models' directory 

`cd ..`
`cd models`

5. Run this command to download the skin cancer classification model.

`wget  PUT LINK HERE ISA`

6. cd back to 'classification' directory

`cd ..`

7. Use the following command to make sure that the model is on the nano. You should see a file called resnet18.onnx.

 `ls models/skincancer/` 

8. Set the NET and DATASET variables by running each of these commands separately

`NET=models/skincancer`

`DATASET=data/skincancerdata`

9. Run this command try the model and see how it operates on an image from the test folder!! Change 'NAME HERE' to name your output file and rename 'NAME OF CATEGORY' and 'IMAGE NAME'
    
`imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/NAME OF CATEGORY/IMAGE NAME .jpg $DATASET/output/NAME OUTPUT.jpg`

This is an example of what your command should look like after you replace the fill-ins.

`imagenet.py --model=$NET/resnet18.onnx --input_blob=input_0 --output_blob=output_0 --labels=$DATASET/labels.txt $DATASET/test/bcc/bcc2.jpg $DATASET/output/test1.jpg`

10. Look at your results by opening the image that just saved in the 'output' folder! This folder should be located in jetson-inference/python/training/classification/data/skincancerdata/output

![Example Output](https://github.com/user-attachments/assets/55858f82-f29d-43d1-b153-b811ed9a7542)


[View a video explanation here!!](video link)
