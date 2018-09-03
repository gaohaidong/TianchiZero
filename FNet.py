import tensorflow as tf
slim = tf.contrib.slim
import numpy as np
import os
import vgg19
#日志文件保存路径和checkpoint文件名
train_log_dir = './log/fine_tune'
train_log_file = 'fnet_fine_tune.ckpt'

#官方imagenet已训练好的checkpoint
checkpoint_file = './log/vgg19/vgg_19.ckpt'

#TODO
batch_size = 
#TODO
learning_rate = 
#TODO
train_epochs =

if not tf.gfile.Exists(train_log_dir):
    tf.gfile.MakeDirs(train_log_dir)

#TODO 获取数据
train_images , train_labels=

#组装VGG19模型
arg_scope = vgg19.vgg_arg_scope()
with slim.arg_scope(arg_scope):
    input_images = tf.placeholder(dtype=tf.float32,shape = [None,IMAGE_SIZE,IMAGE_SIZE,3])
    input_labels = tf.placeholder(dtype=tf.float32,shape = [None,NUM_CLASSES])
    is_training = tf.placeholder(dtype = tf.bool)
    #这里根据论文要求，是否对其他的层进行训练
    logits,end_points =  vgg19.vgg19_slim(input_images,num_classes=NUM_CLASSES,is_training=is_training)
    #除了最后一层都恢复
    params = slim.get_variables_to_restore(exclude=['vgg_19/fc8'])
    restorer = tf.train.Saver(params)

    #计算损失
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=input_labels,logits=logits))
    #TODO 优化器的参数
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    #训练集的预测
    correct = tf.equal(tf.argmax(logits,1),tf.argmax(input_labels,1))
    accuracy = tf.reduce_mean(tf.cast(correct,tf.float32))
    #保存最后5个结果
    save = tf.train.Saver(max_to_keep=5)
    #恢复模型并启动训练，论文要求所有网络一起训练后面进行代码修改，这里先写上
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #最后的checkpoint文件
        ckpt = tf.train.latest_checkpoint(train_log_dir)
        if ckpt is not None:
            save.restore(sess,ckpt)
            print('From the lastest checkpoint...')
        else:
            restorer.restore(sess,checkpoint_file)
            print('From the offical pre-trained model...')

        print('Time to start training model....')
        num_batch = int(np.ceil(n_train/batch_size))
        for epoch in range(training_epochs):
            total_loss =0.0
            for i in range(num_batch):
                _ , cost = sess.run([optimizer,loss],feed_dict={input_images:train_images,input_labels:train_labels,is_training:True}) 
                total_loss += cost
            #打印信息
            if epoch % display_epoch == 0:
                print('Epoch {}/{}  average cost {:.9f}'.format(epoch+1,training_epochs,total_loss/num_batch))
            #输出准确率
            accuracy_value = sess.run([accuracy],feed_dict=input_images:train_images,input_labels:train_labels,is_training:False})
            print('准确率:',accuracy_value)
            #保存
            save.save(sess,os.path.join(train_log_dir,train_log_file),global_step=epochs)
            print('Epoch {}/{}  模型保存成功'.format(epoch+1,training_epochs))
            





