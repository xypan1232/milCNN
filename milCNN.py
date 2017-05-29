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
from keras.layers import normalization, Lambda, GlobalMaxPooling2D, Lambda, GlobalMaxPooling1D
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
from utils import doublet_shuffle, split_training_validation
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

def get_embed_dim_new(embed_file):
    with open(embed_file) as f:
        pepEmbedding = pickle.load(f)
        
    embedded_dim = pepEmbedding[0].shape
    print embedded_dim
    n_aa_symbols, embedded_dim = embedded_dim
    print n_aa_symbols, embedded_dim
    # = embedded_dim[0]
    embedding_weights = np.zeros((n_aa_symbols + 1,embedded_dim))
    embedding_weights[1:,:] = pepEmbedding[0]
    
    return embedded_dim, pepEmbedding[0], n_aa_symbols

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

def load_graphprot_data(protein, train = True, path = '../GraphProt_CLIP_sequences/'):
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

def read_fasta_file(fasta_file):
    seq_dict = {}    
    fp = open(fasta_file, 'r')
    name = ''
    for line in fp:
        #let's discard the newline at the end (if any)
        line = line.rstrip()
        #distinguish header from sequence
        if line[0]=='>': #or line.startswith('>')
            #it is the header
            name = line[1:] #discarding the initial >
            seq_dict[name] = ''
        else:
            #it is sequence
            seq_dict[name] = seq_dict[name] + line.upper()
    fp.close()
    
    return seq_dict

def load_rnacomend_data(datadir = '../data/'):
    pair_file = datadir + 'interactions_HT.txt'
    #rbp_seq_file = datadir + 'rbps_HT.fa'
    rna_seq_file = datadir + 'utrs.fa'
     
    rna_seq_dict = read_fasta_file(rna_seq_file)

    inter_pair = {}
    with open(pair_file, 'r') as fp:
        for line in fp:
            values = line.rstrip().split()
            protein = values[0]
            rna = values[1]
            inter_pair.setdefault(protein, []).append(rna)
    
    return inter_pair, rna_seq_dict

def get_rnarecommend(rnas, rna_seq_dict):
    data = {}
    label = []
    rna_seqs = []
    for rna in rnas:
        rna_seq = rna_seq_dict[rna]
        rna_seq = rna_seq.replace('T', 'U')
        label.append(1)
        rna_seqs.append(rna_seq)
        label.append(0)
        shuff_seq = doublet_shuffle(rna_seq)
        rna_seqs.append(shuff_seq)
    data["seq"] = rna_seqs
    data["Y"] = np.array(label)
    
    return data
        

def get_bag_data(seqs, labels):
    bags = []
    #seqs = data["seq"]
    #labels = data["Y"]
    longlen = 0
    for seq in seqs:
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
    return padbags, labels # bags,

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

def set_cnn_model(ninstance=4, input_dim = 4, input_length = 107):
    nbfilter = 16
    model = Sequential() # #seqs * seqlen * 4
    #model.add(brnn)
    model.add(Conv2D(input_shape=(ninstance, input_length, input_dim),
                            filters=nbfilter,
                            kernel_size=(1,10),
                            padding="valid",
                            #activation="relu",
                            strides=(1,3))) # 4 33 16
    model.add(Activation('relu'))
    # model.add(Conv2D(filters=nbfilter*4, kernel_size=(1,5), padding='valid', activation='relu', strides=(1,2))) # 4 94 64
    # model.add(MaxPooling2D(pool_size=(1,3))) # 4 31 64
    # model.add(Dropout(0.25)) # will be better
    model.add(Conv2D(filters=nbfilter*2*4*2*2*2, kernel_size=(1,3), padding='valid', activation='relu', strides=(1,1))) # 4 31 128
    # model.add(Flatten())
    #model.add(Softmax4D(axis=1))

    #model.add(MaxPooling1D(pool_length=3))
    #model.add(Flatten())
    #model.add(Recalc(axis=1))
    # model.add(Flatten())
    # model.add(Dense(nbfilter*2, activation='relu'))
    # model.add(Dropout(0.25))
    model.add(Conv2D(filters=2, kernel_size=(1,31), padding='valid', activation='softmax', strides=(1,1)))
    model.add(Lambda(lambda x: x[:,:,:,1], output_shape=(1, 1, 1)))

    return model
        
def get_all_mildata(protein, dataset = 'graphprot'):
    data = load_graphprot_data(protein)
    #seqs = data["seq"]
    #labels = data["Y"]
    #pdb.set_trace()
    train_bags, label = get_bag_data(data["seq"], data["Y"])
    #pdb.set_trace()

    test_data = load_graphprot_data(protein, train = False)
    test_bags, true_y = get_bag_data(test_data["seq"], test_data["Y"]) 
    
    return train_bags, label, test_bags, true_y

def get_all_rna_mildata(rnas, seq_dict):
    data = get_rnarecommend(rnas, seq_dict)
    labels = data["Y"]
    seqs = data["seq"]
    training_val_indice, train_val_label, test_indice, test_label = split_training_validation(labels)
    
    train_seqs = []
    for val in training_val_indice:
        train_seqs.append(seqs[val])
        
    train_bags, label = get_bag_data(train_seqs, train_val_label)
    
    test_seqs = []
    for val in test_indice:
        test_seqs.append(seqs[val])        

    #test_data = load_graphprot_data(test_seqs, test_label)
    test_bags, true_y = get_bag_data(test_seqs, test_label) 
    
    return train_bags, label, test_bags, true_y

def run_network(model, total_hid, train_bags, test_bags, y_bags):
    # model.add(Dense(1)) # binary classification
    # model.add(Activation('softmax')) # #instance * 2
    model.add(GlobalMaxPooling1D()) # max pooling multi instance 

    model.summary()
    savemodelpng = 'net.png'
    #plot_model(model, to_file=savemodelpng, show_shapes=True)
    # print(len(train_bags), len(test_bags), len(y_bags), train_bags[0].shape, y_bags[0].shape, len(train_bags[0]))
    #categorical_crossentropy, binary_crossentropy, mil_squared_error
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) 
    #model.compile(loss=mil_squared_error, optimizer='rmsprop') 
    # print 'model training'
    #nb_epos= 5
    #model.fit(train_bags, y_bags, batch_size = 60, epochs=nb_epos, verbose = 0)
    
    #categorical_crossentropy, binary_crossentropy
    #sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) custom_objective
    model.compile(loss='binary_crossentropy', optimizer='rmsprop')
    #model.compile(loss=custom_objective, optimizer='rmsprop')
    print 'model training'
    nb_epos= 8
    # y_bags = to_categorical(y_bags, 2)
    # for iterate in range(nb_epos):
    print 'train epoch', nb_epos
        # for training, y in zip(train_bags, y_bags):
        #     tmp_size = len(training)
            #pdb.set_trace()
            #ys = np.array(tmp_size *[y]) # make the labels in the bag all have the same labels, maybe not correct?
            # ys = np.zeros((tmp_size,2))
            # ys[:, y] = 1  # binary class ############################################################################### one hot encoding
            # ys = y*np.ones((4,))   #  I do not understand the correspondence of ys and tarining, need to confirm  ####
            # trainingreshap = np.reshape(training, (1, training.shape[0], training.shape[1], training.shape[2]))
            # print(training.shape, y.shape)
    model.fit(train_bags, y_bags, batch_size = 100, epochs=nb_epos, verbose = 1)
        #model.reset_states()
            #ys = np_utils.to_categorical(ys)
            #model.train_on_batch(training, ys)
    print 'predicting'         
 #    predictions = []
 #    for testing in test_bags:
	# pdb.set_trace()
    pred = model.predict_proba(test_bags, verbose = 0)
        # predictions.append(pred[0][0])
    return pred

def run_graphprot_milcnn():

    data_dir = '../GraphProt_CLIP_sequences/'

    fw = open('result_micnn_mil', 'w')
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
        train_bags, train_labels, test_bags, test_labels = get_all_mildata(protein)
        train_bags_arr = np.asarray(train_bags).squeeze()
        train_labels_arr = np.array(train_labels)
        test_bags_arr = np.array(test_bags).squeeze()
        test_labels_arr = np.array(test_labels)
        print(train_bags[0].shape, train_labels[0], train_bags_arr.shape, train_labels_arr.shape, test_bags_arr.shape, test_labels_arr.shape)
        net =  set_cnn_model(ninstance=train_bags[0].shape[1])
        
        #seq_auc, seq_predict = calculate_auc(seq_net)
        hid = 16
        predict = run_network(net, hid, train_bags_arr, test_bags_arr, train_labels_arr)
        
        auc = roc_auc_score(test_labels_arr, predict)
        print 'AUC:', auc
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    
    fw.close()

def run_rnacomend_milcnn():
    inter_pair_dict, rna_seq_dict = load_rnacomend_data()
    fw = open('result_rna_micnn_mil', 'w')
    
    for protein, rnas in inter_pair_dict.iteritems():
        if len(rnas) < 2000:
            continue
        print protein
        fw.write(protein + '\t')
        train_bags, train_labels, test_bags, test_labels = get_all_rna_mildata(rnas, rna_seq_dict)
        train_bags_arr = np.asarray(train_bags).squeeze()
        train_labels_arr = np.array(train_labels)
        test_bags_arr = np.array(test_bags).squeeze()
        test_labels_arr = np.array(test_labels)
        print(train_bags[0].shape, train_labels[0], train_bags_arr.shape, train_labels_arr.shape, test_bags_arr.shape, test_labels_arr.shape)
        net =  set_cnn_model(ninstance=train_bags[0].shape[1])
        
        #seq_auc, seq_predict = calculate_auc(seq_net)
        hid = 16
        predict = run_network(net, hid, train_bags_arr, test_bags_arr, train_labels_arr)
        
        auc = roc_auc_score(test_labels_arr, predict)
        print 'AUC:', auc
        fw.write(str(auc) + '\n')
        mylabel = "\t".join(map(str, test_labels))
        myprob = "\t".join(map(str, predict))  
        fw.write(mylabel + '\n')
        fw.write(myprob + '\n')
    
    fw.close()    
    
        #run_mil_classifier(train_bags, train_labels, test_bags, test_labels)
#run_graphprot_milcnn()
run_rnacomend_milcnn()
#seq= 'TTATCTCCTAGAAGGGGAGGTTACCTCTTCAAATGAGGAGGCCCCCCAGTCCTGTTCCTCCACCAGCCCCACTACGGAATGGGAGCGCATTTTAGGGTGGTTACTCTGAAACAAGGAGGGCCTAGGAATCTAAGAGTGTGAAGAGTAGAGAGGAAGTACCTCTACCCACCAGCCCACCCGTGCGGGGGAAGATGTAGCAGCTTCTTCTCCGAACCAA'
#print len(seq)
#split_overlap_seq(seq)
