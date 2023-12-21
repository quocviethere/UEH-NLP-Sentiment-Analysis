# UEH NLP Sentiment Analysis
This repository contains the code and report for our final project of Natural Language Processing course at UEH. For 3 machine learning models including Naive Bayes [[Mosteller and Wallace (1964)](https://www.tandfonline.com/doi/abs/10.1080/01621459.1963.10500849)], Maxent [[Berger et al., 1996](https://dl.acm.org/doi/10.5555/234285.234289)] and XGBoost [[Chen and Guestrin, 2016](https://dl.acm.org/doi/10.1145/2939672.2939785)], we create a custom UI that allows users to directly enter the review and select a machine learning model of their choice. We deploy the RoBERTa model [[Liu et al., 2019](https://arxiv.org/abs/1907.11692)] via Gradio and you can try experiment with it on our website [here](https://ueh-nlp.github.io/). The model checkpoint is available [here](https://drive.google.com/file/d/14mPTsgoV-Px0lBEjmwU8RQReHwWWKC3D/view?usp=sharing). 


## Demo
---


https://github.com/quocviethere/UEH-NLP-Sentiment-Analysis/assets/96617645/b75a671a-6f67-492f-97de-a78d4bb4f309


---

## Implementation

To reproduce our results, first you need to clone the repository:

```
git clone https://github.com/quocviethere/UEH-NLP-Sentiment-Analysis
```

We provide the code for 3 models. 

**Example usage:**

```
$ python naivebayes.py
$ python maxent.py
$ python xgboostclassifier.py
```

Please do note that you need to specify the dataset path correctly for the code to work. To run the UI, simply use:

```
python app.py
```

The Colab Notebook is also available:
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1xVJJBzXdzL3dGXZQw9glIKw7c7IcJ77f?usp=sharing)

## Results

We use 3 machine learning models including Naive Bayes, Maxent and XGBoost and a pretrained RoBERTa. The result is as follows:

<img width="1255" alt="Screen Shot 2023-12-15 at 10 28 22" src="https://github.com/quocviethere/UEH-NLP-Sentiment-Analysis/assets/96617645/395ba14c-4125-4c32-ab07-b5ab73af7f93">

For more detailed results as well as model details, check out our full report here.

---





