<h1 align="center">
  Computer Vision
</h1>

## Run

#### 1. Create virtual environment

```shell
python3 -m venv ./venv
```

#### 2. Install libraries

```shell
pip3 install -r requirements.txt
```

#### 3. Prepare dataset

- You can find dataset I use [here](https://www.kaggle.com/code/alkanerturan/facedetection-recognition/notebook)

#### 4. Create train, validate dataset and begin to train your model

- Create necessary folder

```shell
mkdir resources
```

- You must follow all steps in [main.ipynb](main.ipynb)
- After training, please save your model(under filename vgg16_1.h5) and metadata file(metadata.json) to `resources`.

#### 5. Run demo app

```shell
make run
```

## Reference

- https://www.kaggle.com/code/alkanerturan/facedetection-recognition/notebook
- https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
