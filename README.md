# ResNet-18 and Image Classification

<p align="justify">
<strong>ResNet-18</strong> is a deep convolutional neural network (CNN) that is part of the ResNet (Residual Network) family, introduced by Microsoft in 2015. It consists of 18 layers and is designed to address the vanishing gradient problem, which can hinder the training of deep networks. ResNet-18 uses "residual connections" or shortcuts that allow the network to learn identity mappings, enabling more efficient training of deeper models.
</p>

## Image Classification
<p align="justify">
<strong>Image classification</strong> is a fundamental problem in computer vision where the goal is to assign a label or category to an image based on its content. This task is critical for a variety of applications, including medical imaging, autonomous vehicles, content-based image retrieval, and social media tagging.
</p>


## ❤️❤️❤️


```bash
If you find this project useful, please give it a star to show your support and help others discover it!
```

## Getting Started

### Clone the Repository

To get started with this project, clone the repository using the following command:

```bash
git clone https://github.com/TruongNV-hut/AIcandy_ResNet18_ImageClassification_hnxhkhii.git
```

### Install Dependencies
Before running the scripts, you need to install the required libraries. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Training the Model

To train the model, use the following command:

```bash
python aicandy_resnet18_train_spigbrgr.py --train_dir ../dataset --num_epochs 10 --batch_size 32 --model_path aicandy_model_out_expntoop/aicandy_model_pth_vixylhua.pth
```

### Testing the Model

After training, you can test the model using:

```bash
python aicandy_resnet18_test_eeedrhkp.py --image_path ../image_test.jpg --model_path aicandy_model_out_expntoop/aicandy_model_pth_vixylhua.pth --label_path label.txt
```

### Converting to ONNX Format

To convert the model to ONNX format, run:

```bash
python aicandy_resnet18_convert_onnx_kslhrnsa.py --model_path aicandy_model_out_expntoop/aicandy_model_pth_vixylhua.pth --onnx_path aicandy_model_out_expntoop/aicandy_model_onnx_iihkdlns.onnx --num_classes 2
```

### More Information

To learn more about this project, [see here](https://aicandy.vn/ung-dung-mang-resnet-18-vao-phan-loai-hinh-anh).

To learn more about knowledge and real-world projects on Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL), visit the website [aicandy.vn](https://aicandy.vn/).

❤️❤️❤️




