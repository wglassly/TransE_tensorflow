import tensorflow as tf
import numpy as np
import math
from util import dataset

##### HyperParam Setting####
embedding_size = 50
batch_size     = 5000
margin         = 1
learning_rate  = 0.001
epochs         = 1000
############################


####tensorflow setting####
tf_config = tf.ConfigProto()
tf_config.gpu_options.per_process_gpu_memory_fraction = 0.05        #using gpu mem
#########################

def trans_e_model( path ):

    #read dataset
    ds = dataset( path )
    entity_size = ds.entity_nums + 1                    #add 1 avoid out_of_dict
    relation_size = ds.relation_nums[0] + 1
    model_path = path + 'model/'

    #the distance of h r t
    def l1_energy(batch):
        #h = t+r
        return tf.reduce_sum(tf.abs(batch[:,1,:] - batch[:,0,:] - batch[:,2,:]) ,1)

    with tf.device('/cpu:0'):
        e_embedding_table = tf.Variable(tf.truncated_normal([entity_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)), name = 'e_embed')
        r_embedding_table = tf.Variable(tf.truncated_normal([relation_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)), name = 'r_embed')

    postive_sample = tf.placeholder(tf.int32, shape=[batch_size,3], name='p_sample')
    negtive_sample = tf.placeholder(tf.int32, shape=[batch_size,3], name='n_sample')

    pos_embed_e = tf.nn.embedding_lookup(e_embedding_table, postive_sample[:,:2])
    pos_embed_r = tf.nn.embedding_lookup(r_embedding_table, postive_sample[:,-1:])
    pos_embed = tf.concat([pos_embed_e,pos_embed_r], axis = 1)
    neg_embed_e = tf.nn.embedding_lookup(e_embedding_table, negtive_sample[:,:2])
    neg_embed_r = tf.nn.embedding_lookup(r_embedding_table, negtive_sample[:,-1:])
    neg_embed = tf.concat([neg_embed_e,neg_embed_r], axis = 1)

    p_loss, n_loss = l1_energy(pos_embed), l1_energy(neg_embed)

    loss =  tf.reduce_sum(tf.nn.relu(margin + p_loss - n_loss))                         #loss of TransE
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)                    #opt


    #session
    with tf.Session(config=tf_config) as sess:
        sess.run(tf.initialize_all_variables())
        saver = tf.train.Saver(max_to_keep=None)
        #e_emb, r_emb = [],[]
        print("start training with total {0} epochs and each batch size is{1}".format(epochs, batch_size))
        for e in range(epochs):
            for step in range(len(ds.train_pair)/batch_size):
                p, n = ds.get_next_batch(batch_size=batch_size, corpus=ds.train_pair)
                feed_dict = {postive_sample:p,negtive_sample:n}
                loss_val, _, e_emb, r_emb = sess.run([loss, optimizer, e_embedding_table, r_embedding_table], feed_dict=feed_dict)
            print(" loss_val {1} at epoch {2}".format(step, loss_val, e))
        saver.save(sess, save_path = model_path + '_TransE.model')
        np.save(model_path+"_TransE_ent.npy",e_emb)
        np.save(model_path+"_TransE_rel.npy",r_emb)
        print("Train Done!")

if __name__ == '__main__':
    trans_e_model(path='./data/')
