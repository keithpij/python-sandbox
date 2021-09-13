from string import punctuation

import numpy as np
import torch

import pytorch_sa

def tokenize_text(text, mapping, sequence_length):
    lower_text = text.lower() # lowercase

    # Remove punctuation
    clean_text = ''.join([c for c in lower_text if c not in punctuation])

    # splitting by spaces
    text_words = clean_text.split()

    # Tokens
    text_ints = [mapping[word] for word in text_words]

    
    # Pad with zeros
    length = len(text_ints)
    if length <= sequence_length:
        zeroes = list(np.zeros(sequence_length-length, dtype = int))
        features = zeroes + text_ints
    elif length > sequence_length:
        features = text_ints[0:sequence_length]

    reshaped = []
    reshaped.append(features)
    return reshaped


def predict(net, features, train_on_gpu=False):
    
    net.eval()
    
    # convert to tensor to pass into your model
    feature_tensor = torch.from_numpy(np.array(features))
    
    batch_size = feature_tensor.size(0)
    #batch_size = 1
    print('Batch size: ', batch_size)    

    # initialize hidden state
    val_h = net.init_hidden(batch_size)

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    #val_h = tuple([each.data for each in val_h])

    if(train_on_gpu):
        feature_tensor = feature_tensor.cuda()

    # get the output from the model
    output, val_h = net(feature_tensor, val_h)

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze()) 
    # printing output value, before rounding
    print('Prediction value, pre-rounding: {:.6f}'.format(output.item()))

    # print custom response
    if(pred.item()==1):
        print('Positive review detected.')
    else:
        print('Negative review detected.')


if __name__ == "__main__":
    #text = 'It was so amazing. Best Marvel movie, better than endgame not kidding.'
    text = 'This sucks it totally stunk.'
    X, y = pytorch_sa.get_all_data()
    print('Reviews: ', len(X))

    word_to_int_mapping = pytorch_sa.create_tokens(X)

    # test code and generate tokenized review
    features = tokenize_text(text, word_to_int_mapping, 200)
    print(features)

    # test conversion to tensor and pass into your model
    #feature_tensor = torch.from_numpy(np.array(features))
    #print(feature_tensor.size())
    net = pytorch_sa.load_network()    
    predict(net, features)
