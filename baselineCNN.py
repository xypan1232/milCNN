import sys
import os
import numpy as np
import pdb
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.merge import Concatenate
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.optimizers import SGD, RMSprop, Adadelta, Adagrad, Adam
from keras.layers import normalization, Lambda, GlobalMaxPooling2D
from keras.layers import LSTM, Bidirectional, Reshape
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Conv2D, MaxPooling2D,Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from customlayers import Recalc, ReRank, ExtractDim, SoftReRank, ActivityRegularizerOneDim, RecalcExpand, Softmax4D
from keras.constraints import maxnorm
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from keras import objectives
from keras import backend as K
from keras.utils import np_utils, plot_model
from sklearn.metrics import roc_curve, auc, roc_auc_score
_EPSILON = K.epsilon()
import random
import gzip
import pickle
# import tensorflow as tf
# os.environ["CUDA_VISIBLE_DEVICES"]="1"
# from keras import backend as K
# config = tf.ConfigProto()
# config.gpu_options.allow_growth=True
# sess = tf.Session(config=config)
# K.set_session(sess)
def padding_sequence_new(seq, max_len = 101, repkey = 'N'):
    seq_len = len(seq)
    new_seq = seq
    if seq_len < max_len:
        gap_len = max_len -seq_len
        new_seq = seq + repkey * gap_len
    else:
        new_seq = seq[:max_len]
    return new_seq

def read_rna_dict(rna_dict = 'rna_dict'):
    odr_dict = {}
    with open(rna_dict, 'r') as fp:
        for line in fp:
            values = line.rstrip().split(',')
            for ind, val in enumerate(values):
                val = val.strip()
                odr_dict[val] = ind
    
    return odr_dict

def get_6_trids():
    nucle_com = []
    chars = ['A', 'C', 'G', 'U']
    base=len(chars)
    end=len(chars)**6
    for i in range(0,end):
        n=i
        ch0=chars[n%base]
        n=n/base
        ch1=chars[n%base]
        n=n/base
        ch2=chars[n%base]
        n=n/base
        ch3=chars[n%base]
        n=n/base
        ch4=chars[n%base]
        n=n/base
        ch5=chars[n%base]
        nucle_com.append(ch0 + ch1 + ch2 + ch3 + ch4 + ch5)
    return  nucle_com 

def split_overlap_seq(seq):
    window_size = 101
    overlap_size = 20
    #pdb.set_trace()
    bag_seqs = []
    seq_len = len(seq)
    if seq_len >= window_size:
        num_ins = (seq_len - 101)/(window_size - overlap_size) + 1
        remain_ins = (seq_len - 101)%(window_size - overlap_size)
    else:
        num_ins = 0
    bag = []
    end = 0
    for ind in range(num_ins):
        start = end - overlap_size
        if start < 0:
            start = 0
        end = start + window_size
        subseq = seq[start:end]
        bag_seqs.append(subseq)
    if num_ins == 0:
        seq1 = seq
        pad_seq = padding_sequence_new(seq1)
        bag_seqs.append(pad_seq)
    else:
        if remain_ins > 10:
            #pdb.set_trace()
            #start = len(seq) -window_size
            seq1 = seq[-window_size:]
            pad_seq = padding_sequence_new(seq1)
            bag_seqs.append(pad_seq)
    return bag_seqs
            
def get_6_nucleotide_composition(tris, seq, ordict):
    seq_len = len(seq)
    tri_feature = []
    k = len(tris[0])
    #tmp_fea = [0] * len(tris)
    for x in range(len(seq) + 1- k):
        kmer = seq[x:x+k]
        if kmer in tris:
            ind = tris.index(kmer)
            tri_feature.append(ordict[str(ind)])
        else:
            tri_feature.append(-1)
    #tri_feature = [float(val)/seq_len for val in tmp_fea]
        #pdb.set_trace()        
    return np.asarray(tri_feature)

def read_seq_graphprot(seq_file, label = 1):
    seq_list = []
    labels = []
    seq = ''
    with open(seq_file, 'r') as fp:
        for line in fp:
            if line[0] == '>':
                name = line[1:-1]
            else:
                seq = line[:-1].upper()
                seq = seq.replace('T', 'U')
                seq_list.append(seq)
                labels.append(label)
    
    return seq_list, labels

def get_RNA_concolutional_array(seq, motif_len = 4):
    seq = seq.replace('U', 'T')
    alpha = 'ACGT'
    #for seq in seqs:
    #for key, seq in seqs.iteritems():
    row = (len(seq) + 2*motif_len - 2)
    new_array = np.zeros((row, 4))
    for i in range(motif_len-1):
        new_array[i] = np.array([0.25]*4)
    
    for i in range(row-3, row):
        new_array[i] = np.array([0.25]*4)
        
    #pdb.set_trace()
    for i, val in enumerate(seq):
        i = i + motif_len-1
        if val not in 'ACGT':
            new_array[i] = np.array([0.25]*4)
            continue
        try:
            index = alpha.index(val)
            new_array[i][index] = 1
        except:
            pdb.set_trace()
        #data[key] = new_array
    return new_array

def load_graphprot_data(protein, train = True, path = '../milVec/GraphProt_CLIP_sequences/'):
    data = dict()
    tmp = []
    listfiles = os.listdir(path)
    
    key = '.train.'
    if not train:
        key = '.ls.'
    mix_label = []
    mix_seq = []
    mix_structure = []    
    for tmpfile in listfiles:
        if protein not in tmpfile:
            continue
        if key in tmpfile:
            if 'positive' in tmpfile:
                label = 1
            else:
                label = 0
            seqs, labels = read_seq_graphprot(os.path.join(path, tmpfile), label = label)
            #pdb.set_trace()
            mix_label = mix_label + labels
            mix_seq = mix_seq + seqs
    
    data["seq"] = mix_seq
    data["Y"] = np.array(mix_label)
    
    return data

def loaddata_graphprot(protein, train = True, ushuffle = True):
    #pdb.set_trace()
    data = load_graphprot_data(protein, train = train)
    label = data["Y"]
    rna_array = []
    #trids = get_6_trids()
    #nn_dict = read_rna_dict()
    for rna_seq in data["seq"]:
        #rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        
        seq_array = get_RNA_seq_concolutional_array(seq)
        #tri_feature = get_6_nucleotide_composition(trids, rna_seq_pad, nn_dict)
        rna_array.append(seq_array)
    
    return np.array(rna_array), label

def get_bag_data(data):
    bags = []
    seqs = data["seq"]
    labels = data["Y"]
    longlen = 0
    for seq in seqs:
        bag_seq = padding_sequence_new(seq, max_len = 501)
        tri_fea = get_RNA_concolutional_array(bag_seq)
        bags.append(tri_fea)
    '''
        #pdb.set_trace()
        bag_seqs = split_overlap_seq(seq)
        #flat_array = []
        bag_subt = []
        for bag_seq in bag_seqs:
            tri_fea = get_RNA_concolutional_array(bag_seq)
            bag_subt.append(tri_fea)
        data = np.array(bag_subt)
        if longlen < data.shape[0]:
            longlen = data.shape[0]
        # print(data.shape, type(data.shape))
        bags.append(data) #np.reshape(data, [1]+list(data.shape)))
    padbags = []
    for data in bags:
        paddata = np.zeros([longlen]+list(data.shape[1:]))
        paddata[:data.shape[0], :, :] = np.array(data)
        paddata = np.reshape(paddata, [1]+list(paddata.shape))
        padbags.append(paddata)
    '''
    return bags, labels # bags,

def mil_squared_error(y_true, y_pred):
    return K.tile(K.square(K.max(y_pred) - K.max(y_true)), 5)

def custom_objective(y_true, y_pred):
    #prediction = Flatten(name='flatten')(dense_3)
    #prediction = ReRank(k=k, label=1, name='output')(prediction)
    #prediction = SoftReRank(softmink=softmink, softmaxk=softmaxk, label=1, name='output')(prediction)
    '''Just another crossentropy'''
    #y_true = K.clip(y_true, _EPSILON, 1.0-_EPSILON)
    y_true = K.max(y_true)
    #y_armax_index = numpy.argmax(y_pred)
    y_new = K.max(y_pred)
    #y_new = max(y_pred)
    '''
    if y_new >= 0.5:
        y_new_label = 1
    else:
        y_new_label = 0
    cce = abs(y_true - y_new_label)
    '''
    logEps=1e-8
    cce = - (y_true * K.log(y_new+logEps) + (1 - y_true)* K.log(1-y_new + logEps))
    return cce

def set_cnn_model_mil(ninstance=4, input_dim = 4, input_length = 107):
    nbfilter = 16
    model = Sequential() # #seqs * seqlen * 4
    #model.add(brnn)
    model.add(Conv2D(input_shape=(ninstance, input_length, input_dim),
                            filters=nbfilter,
                            kernel_size=(1,10),
                            padding="valid",
                            #activation="relu",
                            strides=1))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1,3))) # 32 16
    # model.add(Dropout(0.25)) # will be better
    model.add(Conv2D(filters=nbfilter*2, kernel_size=(1,32), padding='valid', activation='relu', strides=1))
    # model.add(Flatten())
    #model.add(Softmax4D(axis=1))

    #model.add(MaxPooling1D(pool_length=3))
    #model.add(Flatten())
    #model.add(Recalc(axis=1))
    # model.add(Flatten())
    # model.add(Dense(nbfilter*2, activation='relu'))
    model.add(Dropout(0.25))
    #model.add(Conv2D(filters=1, kernel_size=(1,1), padding='valid', activation='sigmoid', strides=1))
    return model

def set_cnn_model(input_dim = 4, input_length = 107):
    nbfilter = 16
    model = Sequential()
    #model.add(brnn)
    
    model.add(Conv1D(input_dim=input_dim,input_length=input_length,
                            nb_filter=nbfilter,
                            filter_length=10,
                            border_mode="valid",
                            #activation="relu",
                            subsample_length=1))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_length=3))
    

    #model.add(Flatten())
    #model.add(Softmax4D(axis=1))

    #model.add(MaxPooling1D(pool_length=3))
    #model.add(Flatten())
    #model.add(Recalc(axis=1))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(nbfilter*2, activation='relu'))
    model.add(Dropout(0.5))

    return model

        
def get_all_embedding(protein):
    
    data = load_graphprot_data(protein)
    #pdb.set_trace()
    train_bags, label = get_bag_data(data)
    #pdb.set_trace()
    test_data = load_graphprot_data(protein, train = False)
    test_bags, true_y = get_bag_data(test_data) 
    
    return train_bags, label, test_bags, true_y

def run_network(model, total_hid, train_bags, test_bags, y_bags):
    model.add(Dense(2)) # binary classification
    model.add(Activation('softmax')) # #instance * 2
    #model.add(GlobalMaxPooling2D()) # max pooling multi instance 

    model.summary()
    savemodelpng = 'net.png'
    #plot_model(model, to_file=savemodelpng, show_shapes=True)
    print(len(train_bags), len(test_bags), len(y_bags), train_bags[0].shape, y_bags[0].shape, len(train_bags[0]))
    #categorical_crossentropy, binary_crossentropy, mil_squared_error
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    #model.compile(loss=mil_squared_error, optimizer='rmsprop') 
    # print 'model training'
    #nb_epos= 5
    #model.fit(train_bags, y_bags, batch_size = 60, epochs=nb_epos, verbose = 0)
    

    #categorical_crossentropy, binary_crossentropy
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) custom_objective
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    #model.compile(loss=custom_objective, optimizer='rmsprop')
    print 'model training'
    nb_epochs= 5
    y_bags = to_categorical(y_bags, 2)
    model.fit(np.array(train_bags), np.array(y_bags), batch_size = 100, epochs=nb_epochs, verbose = 1)
    '''
    for iterate in range(nb_epos):
        print 'train epoch', iterate
        for training, y in zip(train_bags, y_bags):
            tmp_size = len(training)
            #pdb.set_trace()
            #ys = np.array(tmp_size *[y]) # make the labels in the bag all have the same labels, maybe not correct?
            # ys = np.zeros((tmp_size,2))
            # ys[:, y] = 1  # binary class ############################################################################### one hot encoding
            # ys = y*np.ones((4,))   #  I do not understand the correspondence of ys and tarining, need to confirm  ####
            # trainingreshap = np.reshape(training, (1, training.shape[0], training.shape[1], training.shape[2]))
            # print(training.shape, y.shape)
            model.fit(training, y*np.ones((1,1)), batch_size = tmp_size, epochs=1, verbose = 1)
        #model.reset_states()
            #ys = np_utils.to_categorical(ys)
            #model.train_on_batch(training, ys)
    '''
    print 'predicting'         
    predictions = model.predict_proba(np.array(test_bags))[:, 1]
    #for testing in test_bags:
	#pdb.set_trace()
    #    pred = model.predict_proba(testing, verbose = 0)
    #    predictions.append(pred[0])
    return predictions

def run_milcnn():
    data_dir = '../milVec/GraphProt_CLIP_sequences/'
    fw = open('result_micnn', 'w')
    print(len(os.listdir(data_dir)))
    print(os.listdir(data_dir))
    finished_protein = set()
    for protein in os.listdir(data_dir):
        
        protein = protein.split('.')[0]
        if protein in finished_protein:
            continue
        finished_protein.add(protein)
        print protein
        fw.write(protein + '\t')
        train_bags, train_labels, test_bags, test_labels = get_all_embedding(protein)
        print(train_bags[0].shape, train_labels[0])
        net =  set_cnn_model(input_length = 507)
        
        #seq_auc, seq_predict = calculate_auc(seq_net)
        hid = 16
        predict = run_network(net, hid, train_bags, test_bags, train_labels)
        
        auc = roc_auc_score(test_labels, predict)
        print 'AUC:', auc
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    fw.close()
        #run_mil_classifier(train_bags, train_labels, test_bags, test_labels)
run_milcnn()
#seq= 'TTATCTCCTAGAAGGGGAGGTTACCTCTTCAAATGAGGAGGCCCCCCAGTCCTGTTCCTCCACCAGCCCCACTACGGAATGGGAGCGCATTTTAGGGTGGTTACTCTGAAACAAGGAGGGCCTAGGAATCTAAGAGTGTGAAGAGTAGAGAGGAAGTACCTCTACCCACCAGCCCACCCGTGCGGGGGAAGATGTAGCAGCTTCTTCTCCGAACCAA'
#print len(seq)
#split_overlap_seq(seq)
