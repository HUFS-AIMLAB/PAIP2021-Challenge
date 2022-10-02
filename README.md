# PAIP 2021 Challenge endorsed MICCAI: Perineural Invasion in Multiple Organ Cancer
![Python 3.8](https://img.shields.io/badge/python-3.8-green.svg?style=plastic)
![PyTorch 1.10](https://img.shields.io/badge/PyTorch%20-%23EE4C2C.svg?style=plastic)
![cuDNN 8](https://img.shields.io/badge/cudnn-8-green.svg?style=plastic)

3rd place solution to [PAIP 2021 Challenge endorsed MICCAI: Perineural Invasion in Multiple Organ Cancer](https://paip2021.grand-challenge.org/Home/)

[PDF](https://drive.google.com/file/d/16gWjI5cbwn7zI9Rov7oyPETr_fYU101Q/view?usp=sharing) [Article](http://www.hufsnews.co.kr/news/articleView.html?idxno=21998) [Final Rank](https://paip2021.grand-challenge.org/Final-rank/)

In this work, we propose an organ-specific method to detect perineural invasion (PNI) in multiple organ cancer. The proposed method consists of a classification step and a segmentation step. In the classification step, PNI is detected at the patch level using the organ-specifically trained convolutional neural network. In the segmentation step, PNI is finally delineated at the pixel level on the result of the previous step. In particular, the weights pretrained in classification step were utilized in segmentation step. The results show that our method will be helpful for effectively detecting PNI in multiple organ cancer.

Team Members:
**Dayoung Baik, Seungun Jang, Hwanseung Yoo, Gawon Lee, Junhyeok Lee**
---

## 🎯 Aims
