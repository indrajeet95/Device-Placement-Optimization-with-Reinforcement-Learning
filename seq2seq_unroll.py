import numpy as np
import tensorflow as tf
import time
import mnist_placement as mn


def load_vocab(filename):
    vocab = {}
    with open(filename) as f:
        for idx, line in enumerate(f):
            vocab[line.strip()] = idx #strip remove white spaces
    return vocab

def seq2seq_model(params):
    batch_size = params['batch_size']
    num_units = params['num_units']
    embed_size = params['embed_size']
    vocab_size = params['vocab_size']
    number_of_placement = params['number_of_placement']
    number_of_clusters = params['number_of_clusters']
    G = tf.Graph()
    with G.as_default():     
        # Tensor where we will feed the data into graph
        inputs = tf.placeholder(tf.float32, (None, number_of_clusters,embed_size), 'inputs')

        # Embedding layers
        output_embedding = tf.Variable(tf.random_uniform([params['vocab_size'], params['embed_size']], -1.0, 1.0), dtype=tf.float32)
        with tf.variable_scope("encoding") as encoding_scope:
            lstm_enc = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
            encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(lstm_enc, inputs=inputs, dtype=tf.float32)

        def decode(helper, scope, encoder_outputs,encoder_final_state, reuse=None):
            with tf.variable_scope(scope, reuse=reuse):
                attention_mechanism = tf.contrib.seq2seq.BahdanauAttention(
                num_units=num_units, memory=encoder_outputs,
                #this would mask the rest of the memory content to zero if length is not maximum
                # not valid in our case we have the fix number of clusters
                # maybe if nuber of clusters changes can cannot be dynamic we need to fix this 
                #I have no idea what it does.
                ####memory_sequence_length=input_lengths####
                )
                cell = tf.nn.rnn_cell.LSTMCell(num_units=num_units)
                attn_cell = tf.contrib.seq2seq.AttentionWrapper(cell, attention_mechanism, attention_layer_size=num_units / 2)
                out_cell = tf.contrib.rnn.OutputProjectionWrapper(attn_cell, vocab_size, reuse=reuse)
                decoder = tf.contrib.seq2seq.BasicDecoder(cell=out_cell, helper=helper,initial_state=out_cell.zero_state(dtype=tf.float32, batch_size=batch_size))
                    #initial_state=encoder_final_state)
                outputs = tf.contrib.seq2seq.dynamic_decode(decoder=decoder, output_time_major=False, impute_finished=True, maximum_iterations=number_of_placement)
                return outputs[0]
            
        with tf.variable_scope("decoding") as decoding_scope:
            start_tokens = tf.zeros([batch_size], dtype=tf.int64)
            pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
                output_embedding, start_tokens=tf.to_int32(start_tokens), end_token=10)
            pred_outputs = decode(helper=pred_helper,scope="decoding", encoder_outputs=encoder_outputs,encoder_final_state=encoder_final_state,reuse=False)
            #mn.apply_placement()
            #mn.train()
        #connect outputs to 
        with tf.name_scope("optimization"):
            
            # Loss function
            acc_prob_under_policy = tf.reduce_sum(pred_outputs.rnn_output,[0,1])
            prob_under_policy = tf.reduce_mean (acc_prob_under_policy)
            reward = tf.placeholder(tf.float32,(None),'reward')
            loss = prob_under_policy * reward# * tf.log(prob_under_policy + 1e-13)
            #loss = tf.contrib.seq2seq.sequence_loss(logits, targets, tf.ones([batch_size, y_seq_length]))
            # Optimizer
            optimizer = tf.train.AdamOptimizer().minimize(loss)
            #lstm_dec = tf.contrib.rnn.LSTMCell(nodes)    
            with tf.control_dependencies([optimizer]):
                dummy = tf.constant(0)

            #dec_outputs, _ = tf.nn.dynamic_rnn(lstm_dec, inputs=date_output_embed, initial_state=last_state)
        init_op = tf.global_variables_initializer()    

        saver = tf.train.Saver()        
    return G,init_op,loss,dummy, pred_outputs,inputs,saver,reward 

def train_seq2seq(Graph,init_op,params,loss,dummy, pred_outputs,inputs,vocab,saver,reward):
    
        sess = tf.Session(graph=Graph)
        sess.run(init_op)
        """
        try:
            saver.restore(sess, "./checkpoint/model.ckpt")
        except:
            sess.run(init_op)
        #exit()
        """
        
        for epoch_i in range(params['epochs']):
            h = sess.partial_run_setup([pred_outputs.sample_id,loss,dummy],[inputs,reward])
            start_time = time.time()
            source_batch =  batch_data(params)
            #loss tf.get_default_graph().get_tensor_by_name("loss")
            T_P = sess.partial_run(h,pred_outputs.sample_id,feed_dict = {inputs: source_batch})
            print (get_formatter(T_P[0],vocab))
            mn.apply_placement()
            runtime = mn.train()
            batch_loss,d = sess.partial_run(h,[loss,dummy],feed_dict = {reward: runtime})
            #sess.partial_run(h,dummy)
            #op, batch_loss, T_P = sess.run([optimizer, loss, pred_outputs.sample_id],feed_dict = {inputs: source_batch})

            #mn.apply_placement()
            #mn.train()
            
            #accuracy = np.mean(batch_logits.argmax(axis=-1) == target_batch[:,1:])
            #print('Epoch {:3} Loss: {:>6.3f} Epoch duration: {:>6.3f}s'.format(epoch_i, batch_loss,time.time() - start_time))
        #save_path = saver.save(sess, "./checkpoint/model.ckpt")
        #print("Model saved in path: %s" % save_path)
        #sess.close()
            
            
            
        
        #mn.train()

def batch_data(params):
    input_embed = np.random.randn(params['batch_size'], params['number_of_clusters'], params['embed_size'])
    return input_embed
def get_rev_vocab(vocab):
    return {idx: key for key, idx in vocab.items()}


def get_formatter(keys, vocab):
    rev_vocab = get_rev_vocab(vocab)
    res = []
    for key in keys:
        res.append(" "+rev_vocab[key])
    return res

  
def main():
    vocab_filename = "vocab"   
    vocab = load_vocab(vocab_filename)
    params = {
        'vocab_size': len(vocab),
        'batch_size': 1,
        'number_of_clusters': 30,
        'number_of_placement': 30,
        'embed_size': 100,
        'num_units': 256,
        'epochs': 2
    }
    mn.initilaze()
    mn.make_seesion()
    Graph,init_op,loss,dummy, pred_outputs,inputs,saver,reward  = seq2seq_model(params)

    train_seq2seq(Graph,init_op,params,loss,dummy, pred_outputs,inputs,vocab ,saver,reward)
    


if __name__ == "__main__":
    main()