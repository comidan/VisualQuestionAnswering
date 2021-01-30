# VisualQuestionAnswering
![tensorflow](https://aleen42.github.io/badges/src/tensorflow.svg)

## Introduction
This model is meant for the Visual Question Answering (VQA) problem on the following proposed data set. The data set is composed by synthetic scenes, in which people and objects interact, and by corresponding questions, which are about the content of the images. Given an image and a question, the goal is to provide the correct answer. Answers belong to 3 possible categories: 'yes/no', 'counting' (from 0 to 5) and 'other' (e.g. colors, location, ecc.) answers. An example can be seen in the following.

## Dataset
The data set is constituted of 29333 RGB images of 400 x 700 and 58832 different textual questions, so multiple
questions per image. Being the data set composed of synthetic scenes it didn't required any kind of augmentation
in this specific case.

Sample data | Another sample
:-------------------------:|:-------------------------:
<img src="/results/vqa1.png" width="256" height="192">  |  <img src="/results/vqa2.png" width="256" height="192">

While the RGB images were of an original dimension of just 400 x 700, we choose to handle 200 x 350 RGB images
so that we are both able to loose the minimum of information from the original, double-sized, input images but
also be able to handle such an amount of images given our limited resources. We then also loaded in a lazy way
the images in the RAM and pre-produced the elaboration, further discussed next, on the questions in order to get
faster training per epoch.

## Model Choice
For the image handling part it has been chosen as first architecture the VGG16 model which showed good results in
validation but then we moved to using the InceptionResNetV2 model, given that gave us better validation results.
Regarding instead the architecture for the question handling we decided that, instead of training our model from
scratch or using some old embeddings like Glove or Word2Vec, to use a state-of-the-art model which is using
Self-Attention with Transformers architecture. We initially chose BERT base of 12 layers but given that it was
quite heavy we moved to an its simplification in depth using the DistilBert model. DistilBert, while being smaller
than the actual BERT one, has too many parameters so we are not able to train it using our GPU. Therefore, we
render the DistilBert embeddings of every question, this allows us to use the embeddings of the State-Of-The-Art
model and speed up the training but the embeddings are not fine-tuned for the task. Moreover, DistilBert generate
embeddings which size scales with the length of the question, but we need a constant size input to feed to the
classifier, to handle this we naively average the emebddings on the words axis. Summarizing, this enables us to
better handle each kind of question letting the model already have the best possible inference on sequence text
data through the Self-Attention technique.

## Architecture of the final model
The final model is so composed of InceptionResNetV2 for the image handling while DistilBert is being used to
pre-produce (to enhance training speed) its interpretation of the question which is then given in input to a Dense
layer as a vector of 768 elements.
We also added as said Early Stopping technique during training in order to prevent over fitting and stop the
training at a reasonable epoch.
The actual model here described can be seen in the below Figure 2.

```python
image_input = Input(shape=(*IMAGE_SHAPE, 3), name="image")
vision_model = tf.keras.applications.InceptionResNetV2(include_top=False,weights=imagenet,input_tensor=image_input)
encoded_image = Flatten()(vision_model.output)

features_extraction_pipeline = pipeline('feature-extraction', model=NLP_MODEL, tokenizer=NLP_MODEL)
question_data = Input(shape=(768,), name="question")

h = Concatenate()([question_data, encoded_image])
h = Dense(128)(h)
output = Dense(len(labels_dict), activation=softmax)(h)

vqa_model = Model(inputs=[encoded_question, image_input], outputs=output)
vqa_model.compile(loss="categorical_crossentropy", optimizer=tf.keras.optimizers.Nadam(), metrics=["accuracy"])
```

## Initialization
For the weight initialization with Xavier Initialization we used GlorotNormal to better initialize weights W and
letting backpropagation algorithm start in advantage position, given that final result of gradient descent is affected
by weights initialization.<br>
![equation](https://latex.codecogs.com/gif.latex?W%20%5Csim%20%5Cmathcal%7BN%7D%5Cleft%28%5Cmu%3D0%2C%5C%2C%20%5C%3B%5Csigma%5E%7B2%7D%3D%5Cfrac%7B2%7D%7BN_%7Bin%7D%20&plus;%20N_%7Bout%7D%7D%5Cright%29)

## Results
With this architecture as it's shown in the below figure that it has been reached an accuracy in the validation as
best value of 59.43%, given that we set the parameter restore best weights to True.
On the loss graph we can observe over fitting, so further regularization might be needed but the accuracy on the
validation set does not differ much.<br><br>
![densenet](/results/training_vqa.png)

Here in the following figure it's also shown just for explanatory reasons a small sample of the provided test set with the
corresponding predicted answers.<br><br>
![resnet](/results/test_sample_vqa.png)

## Possible improvements
We want the network to see most of the possible details, so one possible improvement would be to use the tiling
technique so that we would be able to virtually train the network on the full-sized images and so let it being able
to spot most of details.
Another possible improvement would be trying to apply the concept of Self-Attention not only to the text data
but also to the features maps generated by the CNN model part of this VQA architecture. Another embedding
aggregation method that we think it's promising, when aggregating the DistilBert embeddings, is doing a weighted
average using techniques such as binned TF-IDF to identify which are the most important "features" of the
embeddings of each word.
