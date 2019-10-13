import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras
import pydot as pydot
from keras.utils.vis_utils import model_to_dot
keras.utils.vis_utils.pydot = pydot

from cnn_model import link_text_image

from keras import optimizers
from keras import layers
from keras.utils import plot_model, to_categorical
from keras.models import Model, load_model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
#from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.applications.vgg16 import preprocess_input

from nltk.util import ngrams
from nltk.translate.bleu_score import sentence_bleu


def split_test_val_train(dtexts,Ntest,Nval):
    """Split data in certain order
    Inputs
    ----------
    dtexts: numpy array that needed to be splited
    Ntest: the number of testing data
    Nval: the nnumber of valuation data
    Outputs
    -------
    testing data
    valuation data
    training data
    """
    print("in split_test_val_train")
    #print(len(dtexts),Ntest,Nval)
    return dtexts[:Ntest], dtexts[Ntest:Ntest+Nval], dtexts[Ntest+Nval:]

def preprocessing(dtexts,dimages,vocab_size):
    """Final preprocessing for the input and output of the RNN model
    Inputs
    ----------
    dtexts: numpy array of the integer vectors from captions
    dimages: numpy array of the feature vectors generated from CNN model
    Ouputs
    -------
    caption input of the RNN model
    image input of the RNN model
    caption output of the RNN model
    """
    print("in preprocessing")
    N = len(dtexts)
    maxlen = np.max([len(text) for text in dtexts])
    print(maxlen)
    print("# captions/images = {}".format(N))

    assert(N==len(dimages))
    Xtext, Ximage, ytext = [],[],[]
    for text,image in zip(dtexts,dimages):

        for i in range(1,len(text)):
            in_text, out_text = text[:i], text[i]
            in_text = pad_sequences([in_text],maxlen=30).flatten()
            out_text = to_categorical(out_text,num_classes = vocab_size)

            Xtext.append(in_text)
            Ximage.append(image)
            ytext.append(out_text)

    Xtext  = np.array(Xtext)
    Ximage = np.array(Ximage)
    ytext  = np.array(ytext)
    print(" {} {} {}".format(Xtext.shape,Ximage.shape,ytext.shape))
    print(Xtext.shape)
    print(Ximage.shape)
    print(ytext.shape)
    return Xtext, Ximage, ytext

def split_data(dtexts, dimages, fnames, vocab_size):
    """
    Split data into training set, valuation set and testing set
    Prepare inputs and outputs of the caption generating model
    """
    print("in split_data")
    prop_test, prop_val = 0.15, 0.15

    maxlen = np.max([len(text) for text in dtexts])

    N = len(dtexts)
    #print(dimages)
    Ntest, Nval = int(N*prop_test), int(N*prop_val)
    dt_test,  dt_val, dt_train   = split_test_val_train(dtexts,Ntest,Nval)
    di_test,  di_val, di_train   = split_test_val_train(dimages,Ntest,Nval)
    fnm_test,fnm_val, fnm_train  = split_test_val_train(fnames,Ntest,Nval)

    # Final preprocessing for the input and output of the Keras model
    Xtext_train, Ximage_train, ytext_train = preprocessing(dt_train,di_train, vocab_size)
    Xtext_val,   Ximage_val,   ytext_val   = preprocessing(dt_val,di_val,vocab_size)
    # pre-processing is not necessary for testing data
    Xtext_test,  Ximage_test,  ytext_test  = preprocessing(dt_test,di_test,vocab_size)

    val_data=([Ximage_val, Xtext_val], ytext_val)

    return Xtext_train, Ximage_train, ytext_train, val_data, Xtext_test,  Ximage_test,  ytext_test,fnm_test,di_test,dt_test

def define_model(vocab_size, max_length):
    """Define the RNN model
    Inputs:
    ----------
    vocab_size: the total number of the tokens
    max_length: the maxinum length of the captions
    Ouputs
    -------
    caption generating model
    """
    print("in define_model")
    #feature extractor model
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)
     #sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)
    # decoder model
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    #adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    #sgd = optimizers.SGD(lr=0.001, decay=0, momentum=0.9, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    # summarize model
    print(model.summary())
    #print(inputs1,inputs2,outputs)

    #plot_model(model, to_file='model.png', show_shapes=True)
    return model

def plot_loss(hist):
    """Plot historical loss score over epochs
    Inputs
    ----------
    hist: historical loss score
    Ouputs
    -------
    Historical loss plot over epoches
    """
    print("in plot_loss")
    for label in ["loss","val_loss"]:
        plt.plot(hist.history[label],label=label)
    plt.legend()
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.show()

def predict_caption(image):
    '''
    image.shape = (1,4462)
    '''
    #print("in predict_caption")

    maxlen=30
    in_text = 'startseq'

    for iword in range(maxlen):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence],maxlen)
        yhat = model.predict([image,sequence],verbose=0)
        yhat = np.argmax(yhat)
        newword = index_word[yhat]
        in_text += " " + newword
        if newword == "endseq":
            break
    return(in_text)


def plot_prediction():
    """
    Plot images and predicted captions
    """
    print("in plot_prediction")
    npic = 5
    npix = 224
    target_size = (npix,npix,3)

    #231 332 644 963 664 591 592
    fnm_test_sample = (fnm_test[231],fnm_test[332],fnm_test[664],fnm_test[591],fnm_test[1087])
    di_test_sample = (di_test[231],di_test[332],di_test[664],di_test[591],di_test[1087])

    count = 1
    fig = plt.figure(figsize=(10,20))
    for jpgfnm, image_feature in zip(fnm_test_sample,di_test_sample):
        #images
        filename = dir_Flickr_jpg + '/' + jpgfnm
        image_load = load_img(filename, target_size=target_size)
        ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
        ax.imshow(image_load)
        count += 1

        #captions
        caption = predict_caption(image_feature.reshape(1,len(image_feature)))
        ax = fig.add_subplot(npic,2,count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.text(0,0.5,caption,fontsize=20)
        count += 1

    plt.show()
    plt.savefig('C:/Users/Xpro/autocaption/image/prediction.png',bbox_inches='tight')
    


def cal_bleu(fnm_train,di_train,dt_train):
    """
    Calculate bleu scores
    """
    print("in cal_bleu")
    #print(index_word)
    #nkeep = 25
    pred_good, pred_bad, pred_mid, bleus = [], [], [], []
    count = 0
    for jpgfnm, image_feature, tokenized_text in zip(fnm_test,di_test,dt_test):
        count += 1
        if count % 200 == 0:
            print("  {:4.2f}% is done..".format(100*count/float(len(fnm_test))))
        caption_true = [index_word[i] for i in tokenized_text]
        caption_true = caption_true[1:-1] ## remove startreg, and endreg
        ## captions
        caption = predict_caption(image_feature.reshape(1,len(image_feature)))
        caption = caption.split()
        caption = caption[1:-1]## remove startreg, and endreg
        bleu = sentence_bleu([caption_true],caption)
        bleus.append(bleu)
        if bleu > 0.7 and len(pred_good) < len(fnm_test):
            pred_good.append((bleu,jpgfnm,caption_true,caption))
        elif bleu < 0.3 and len(pred_bad) < len(fnm_test):
            pred_bad.append((bleu,jpgfnm,caption_true,caption))
        elif bleu > 0.3 and bleu < 0.7 and len(pred_bad) < len(fnm_test):
            pred_mid.append((bleu,jpgfnm,caption_true,caption))
    print("Mean BLEU {:4.3f}".format(np.mean(bleus)))
    #print(len(pred_good), 'Good images', len(pred_bad),'Bad images', len(pred_mid),'Medium image' )
    
    return pred_good, pred_bad, pred_mid

def plot_images(pred):
    """
    Plot the images with good captions (BLEU > 0.7) or bad captions (BLEU < 0.3)
    """
    print("in plot_images")
    def create_str(caption_true):
        strue = ""
        for s in caption_true:
            strue += " " + s
        return(strue)
    npix = 224
    target_size = (npix,npix,3)
    count = 1
    fig = plt.figure(figsize=(10,20))
    npic = len(pred)
    #print(npic)
    for pb in pred:
        bleu,jpgfnm,caption_true,caption = pb
        ## images
        filename = dir_Flickr_jpg + '/' + jpgfnm
        image_load = load_img(filename, target_size=target_size)
        ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
        ax.imshow(image_load)
        count += 1

        caption_true = create_str(caption_true)
        caption = create_str(caption)

        ax = fig.add_subplot(npic,2,count)
        plt.axis('off')
        ax.plot()
        ax.set_xlim(0,1)
        ax.set_ylim(0,1)
        ax.text(0,0.7,"True:" + caption_true,fontsize=20)
        ax.text(0,0.1,"Pred:" + caption,fontsize=20)
        #ax.text(0,0.1,"BLEU: {}".format(round(bleu,2)),fontsize=20)
        count += 1
    plt.show()
    #plt.savefig('C:/Users/Xpro/autocaption/image/bleu_good.png',bbox_inches='tight')

def tokenize_text(dcaptions):
    """
    Change character vector to integer vector using Tokenizer
    """
    # the maximum number of words in dictionary
    print("in tokenize_text")
    nb_words = 8000
    tokenizer = Tokenizer(nb_words=nb_words)
    tokenizer.fit_on_texts(dcaptions)
    vocab_size = len(tokenizer.word_index) + 1
    print("vocabulary size : {}".format(vocab_size))
    dtexts = tokenizer.texts_to_sequences(dcaptions)
    return vocab_size, dtexts


dir_Flickr_jpg = "C:/Users/Xpro/autocaption/Flickr8k_Dataset/Flicker8k_Dataset"
images = pd.read_pickle(r'C:\Users\Xpro\autocaption\data\images.pkl', compression='infer')
df_txt0 = pd.read_csv('C:/Users/Xpro/autocaption/data/token0.txt', sep='\t')
fnames, dcaptions, dimages = link_text_image(df_txt0,images)
#vocab_size, dtexts = tokenize_text(dcaptions)
    
nb_words = 8000
tokenizer = Tokenizer(nb_words=nb_words)
tokenizer.fit_on_texts(dcaptions)
vocab_size = len(tokenizer.word_index) + 1
print("vocabulary size : {}".format(vocab_size))
dtexts = tokenizer.texts_to_sequences(dcaptions)

#print("dtexts: ",dtexts)
#print(tokenizer.word_index)

maxlen = np.max([len(text) for text in dtexts])
#maximum words in a sentence
print("maxlen:",maxlen)
Xtext_train, Ximage_train, ytext_train, val_data, Xtext_test,  Ximage_test,  ytext_test,fnm_test,di_test,dt_test = split_data(dtexts, dimages, fnames,vocab_size)
    
model = define_model(vocab_size=vocab_size, max_length=maxlen)
index_word = dict([(index,word) for word, index in tokenizer.word_index.items()])
    

#x=predict_caption(Ximage_test,model)
#print(Ximage_test.shape)
# checkpoint
filepath="C:/Users/Xpro/autocaption/tmp/rnn8k-{epoch:02d}.h5"

checkpointer = ModelCheckpoint(filepath, save_best_only=False, save_weights_only=False, mode='auto', period=1)

# fit model
print("Started fitting")
print(Xtext_train.shape)
#start = time.time()
    
#hist = model.fit(x=[Ximage_train, Xtext_train], y=ytext_train,
#                      epochs=50, verbose=2,
 #                     batch_size=64,
  #                    validation_data=val_data,
   #                   callbacks=[checkpointer])
#end = time.time()
print("out of fitting")
#print("TIME TOOK {:3.2f}MIN".format((end - start )/60))
#model.save('C:/Users/Xpro/autocaption/tmp/rnn8k_4e.h5')
#plot_loss(hist)
model = load_model('C:/Users/Xpro/autocaption/tmp/rnn8k_4e.h5')


npic = 20
npix = 224
target_size = (npix,npix,3)

count = 1
fig = plt.figure(figsize=(10,20))
for jpgfnm, image_feature in zip(fnm_test[:npic],di_test[:npic]):
    ## images 
    filename = dir_Flickr_jpg + '/' + jpgfnm
    image_load = load_img(filename, target_size=target_size)
    ax = fig.add_subplot(npic,2,count,xticks=[],yticks=[])
    #ax.imshow(image_load)
    count += 1

    ## captions
    #caption = predict_caption(image_feature.reshape(1,len(image_feature)))
    #print(caption)
    ax = fig.add_subplot(npic,2,count)
    plt.axis('off')
    ax.plot()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    #ax.text(0,0.5,caption,fontsize=20)
    count += 1

#plt.show()

# Prediction
pred_good, pred_bad, pred_mid = cal_bleu(fnm_test,di_test,dt_test)

plot_images(pred_mid)

