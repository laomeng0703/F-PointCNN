from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import pointfly as pf
import tensorflow as tf
from tensorflow.python import debug as tf_debug

import sys
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

import tf_util
#from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg, pointnet_fp_module
from model_util import NUM_HEADING_BIN, NUM_SIZE_CLUSTER, NUM_OBJECT_POINT
from model_util import point_cloud_masking, get_center_regression_net
from model_util import placeholder_inputs, parse_output_to_tensors, get_loss
'''
pts:输入点云数据
fts:输入features
qrs:当前层采样得到的点
tag:xcon层名（string）
N:batch size
K:近邻数
D:膨胀系数
P:代表点的数目
C:输出通道数（深度）
C_pts_fts:输入点的特征的维度
is_traing:是否训练模型
with_X_transformation:是否使用x-transform
depth_multiplier:网络中可分卷积层的参数
sorting_method:排序方法
with_global:有无global层
'''

def xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, with_X_transformation, depth_multiplier,
          sorting_method=None, with_global=False):
    # indices_dilated: batch_size 和 k个点对应的在原数据中的下标索引 （N,P,K,2）
    _, indices_dilated = pf.knn_indices_general(qrs, pts, K * D, True)
    # 每隔D个点抽取一个样本
    indices = indices_dilated[:, :, ::D, :]

    if sorting_method is not None:
        indices = pf.sort_points(pts, indices, sorting_method)
    # 从pts中选取indices中所列出的点的数据，nn_pts大小为(N,P,3)
    nn_pts = tf.gather_nd(pts, indices, name=tag + 'nn_pts')  # (N, P, K, 3)
    # qrs中的每个点作为中心点，nn_pts_center大小为(N, P, 1, 3)
    nn_pts_center = tf.expand_dims(qrs, axis=2, name=tag + 'nn_pts_center')  # (N, P, 1, 3)
    # nn_pts中的点减去qrs中心点的坐标，得到局部坐标系nn_pts_local大小为(N, P, K, 3)
    nn_pts_local = tf.subtract(nn_pts, nn_pts_center, name=tag + 'nn_pts_local')  # (N, P, K, 3)

    # Prepare features to be transformed
    nn_fts_from_pts_0 = pf.dense(nn_pts_local, C_pts_fts, tag + 'nn_fts_from_pts_0', is_training)
    nn_fts_from_pts = pf.dense(nn_fts_from_pts_0, C_pts_fts, tag + 'nn_fts_from_pts', is_training)
    if fts is None:
        nn_fts_input = nn_fts_from_pts
    else:
        nn_fts_from_prev = tf.gather_nd(fts, indices, name=tag + 'nn_fts_from_prev')
        nn_fts_input = tf.concat([nn_fts_from_pts, nn_fts_from_prev], axis=-1, name=tag + 'nn_fts_input')

    if with_X_transformation:
        ######################## X-transformation #########################
        X_0 = pf.conv2d(nn_pts_local, K * K, tag + 'X_0', is_training, (1, K))
        X_0_KK = tf.reshape(X_0, (N, P, K, K), name=tag + 'X_0_KK')
        X_1 = pf.depthwise_conv2d(X_0_KK, K, tag + 'X_1', is_training, (1, K))
        X_1_KK = tf.reshape(X_1, (N, P, K, K), name=tag + 'X_1_KK')
        X_2 = pf.depthwise_conv2d(X_1_KK, K, tag + 'X_2', is_training, (1, K), activation=None)
        X_2_KK = tf.reshape(X_2, (N, P, K, K), name=tag + 'X_2_KK')
        fts_X = tf.matmul(X_2_KK, nn_fts_input, name=tag + 'fts_X')
        ###################################################################
    else:
        fts_X = nn_fts_input

    fts_conv = pf.separable_conv2d(fts_X, C, tag + 'fts_conv', is_training, (1, K), depth_multiplier=depth_multiplier)
    fts_conv_3d = tf.squeeze(fts_conv, axis=2, name=tag + 'fts_conv_3d')

    if with_global:
        fts_global_0 = pf.dense(qrs, C // 4, tag + 'fts_global_0', is_training)
        fts_global = pf.dense(fts_global_0, C // 4, tag + 'fts_global', is_training)
        return tf.concat([fts_global, fts_conv_3d], axis=-1, name=tag + 'fts_conv_3d_with_global')
    else:
        return fts_conv_3d


num_class = 8

x = 8
xconv_param_name = ('K', 'D', 'P', 'C', 'links')
xconv_params = [dict(zip(xconv_param_name, xconv_param)) for xconv_param in
                     [(8, 1, 2048, 32 * x, []),
                      (12, 2, 768, 64 * x, []),
                      (16, 2, 384, 96 * x, []),
                      (16, 4, 128, 128 * x, [])]]
fc_param_name = ('C', 'dropout_rate')
fc_params = [dict(zip(fc_param_name, fc_param)) for fc_param in
                  [(32 * x, 0.0),
                   (32 * x, 0.5)]]
xdconv_param_name = ('K', 'D', 'pts_layer_idx', 'qrs_layer_idx')
xdconv_params = [dict(zip(xdconv_param_name, xdconv_param)) for xdconv_param in
                 [(16, 4, 3, 3),
                  (16, 2, 2, 2),
                  (12, 2, 2, 1),
                  (8, 2, 1, 0)]]



use_extra_features = True
with_normal_feature = False
with_X_transformation = True
sorting_method = None
keep_remainder = True
with_global = True



def pointcnn_seg(points, features, one_hot_vec, is_training, end_points):

    sampling = 'ids'

    optimizer = 'adam'
    epsilon = 1e-5
    
    data_dim = 6
    use_extra_features = True
    with_normal_feature = False
    with_X_transformation = True
    sorting_method = None
    keep_remainder = True
    with_global = True
    N = tf.shape(points)[0]
    from sampling import tf_sampling
    one_hot_vec = tf.expand_dims(one_hot_vec, 1)

    layer_pts = [points]
    if features is None:
        layer_fts = [features]
    else:
        features = one_hot_vec
        features = tf.reshape(features, (N, -1, data_dim - 3), name='features_reshape')
        C_fts = xconv_params[0]['C'] // 2
        features_hd = pf.dense(features, C_fts, 'features_hd', is_training)
        layer_fts = [features_hd]
        #features = None
        #layer_fts = [features]
    print('features', features)

    for layer_idx, layer_param in enumerate(xconv_params):
        tag = 'xconv_' + str(layer_idx + 1) + '_'
        K = layer_param['K']
        D = layer_param['D']
        P = layer_param['P']
        C = layer_param['C']
        links = layer_param['links']
        if sampling != 'random' and links:
            print('Error: flexible links are supported only when random sampling is used!')
            exit()

        # get k-nearest points
        pts = layer_pts[-1]
        fts = layer_fts[-1]
        if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
            qrs = layer_pts[-1]
        else:
            if sampling == 'fps':
                fps_indices = tf_sampling.farthest_point_sample(P, pts)
                batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                indices = tf.concat([batch_indices, tf.expand_dims(fps_indices,-1)], axis=-1)
                qrs = tf.gather_nd(pts, indices, name= tag + 'qrs') # (N, P, 3)
            elif sampling == 'ids':
                indices = pf.inverse_density_sampling(pts, K, P)
                qrs = tf.gather_nd(pts, indices)
            elif sampling == 'random':
                qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            else:
                print('Unknown sampling method!')
                exit()
        layer_pts.append(qrs)
        #print('layer_pts',layer_pts)

        if layer_idx == 0:
            C_pts_fts = C // 2 if fts is None else C // 4
            depth_multiplier = 4
        else:
            C_prev = xconv_params[layer_idx - 1]['C']
            C_pts_fts = C_prev // 4
            depth_multiplier = math.ceil(C / C_prev)
        with_global = (with_global and layer_idx == len(xconv_params) - 1)
        fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, 
                          with_X_transformation,
                          depth_multiplier, sorting_method, with_global)
        #print('fts_xconv:', fts_xconv)


        #one_hot_vec = tf.tile(one_hot_vec, [1, P, 1])
        #print('one_hot_vec:', one_hot_vec)
        #fts_xconv = tf.concat([fts_xconv, one_hot_vec], axis=2)
        #one_hot_vec = tf.slice(one_hot_vec, [0,0,0], [-1,1,-1])
        #print('one_hot_vec_slice:', one_hot_vec)

        fts_list = []
        for link in links:
            fts_from_link = layer_fts[link]
            if fts_from_link is not None:
                fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1), 
                                     name=tag + 'fts_slice_' + str(-link))
                fts_list.append(fts_slice)
        if fts_list:
            fts_list.append(fts_xconv)
            layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
        else:
            layer_fts.append(fts_xconv)
        print('layer_fts',layer_fts)



    for layer_idx, layer_param in enumerate(xdconv_params):
        tag = 'xdconv_' + str(layer_idx + 1) + '_'
        K = layer_param['K']
        D = layer_param['D']
        pts_layer_idx = layer_param['pts_layer_idx']
        qrs_layer_idx = layer_param['qrs_layer_idx']
                    
        pts = layer_pts[pts_layer_idx + 1]
        fts = layer_fts[pts_layer_idx + 1] if layer_idx == 0 else layer_fts[-1]
        qrs = layer_pts[qrs_layer_idx + 1]
        fts_qrs = layer_fts[qrs_layer_idx + 1]
        P = xconv_params[qrs_layer_idx]['P']
        C = xconv_params[qrs_layer_idx]['C']
        C_prev = xconv_params[pts_layer_idx]['C']
        C_pts_fts = C_prev // 4
        depth_multiplier = 1

        #print('fts: ',fts)

        '''one_hot_vec = tf.tile(one_hot_vec, [1, P, 1])
        print('one_hot_vec:', one_hot_vec)
        fts = tf.concat([fts, one_hot_vec], axis=2)
        print('fts: ', fts)
        one_hot_vec = tf.slice(one_hot_vec, [0,0,0], [-1,1,-1])
        print('one_hot_vec_slice:', one_hot_vec)'''

        fts_xdconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training, 
                           with_X_transformation,
                           depth_multiplier, sorting_method)
        fts_concat = tf.concat([fts_xdconv, fts_qrs], axis=-1, name=tag + 'fts_concat')
        fts_fuse = pf.dense(fts_concat, C, tag + 'fts_fuse', is_training)
        layer_pts.append(qrs)
        layer_fts.append(fts_fuse)

    fc_layers = [layer_fts[-1]]
    for layer_idx, layer_param in enumerate(fc_params):
        C = layer_param['C']
        dropout_rate = layer_param['dropout_rate']
        fc = pf.dense(fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
        fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training, 
                                    name='fc{:d}_drop'.format(layer_idx))
        fc_layers.append(fc_drop)

    print('fc_layers:',fc_layers)
    logits = pf.dense(fc_layers[-1], num_class, 'logits',
                      is_training, with_bn=False, activation=None)
    print('logits:',logits)
    logits = tf_util.conv1d(logits, 2, 1, padding='VALID', activation_fn=None, scope='conv1d-fc')

    return logits, end_points


'''def boundingbox_3d(object_point_cloud, one_hot_vec, is_training, end_points):
    sampling = 'ids'
    with_X_transformation = True
    sorting_method = None

    with_global = True
    N = tf.shape(object_point_cloud)[0]
    from sampling import tf_sampling
    one_hot_vec = tf.expand_dims(one_hot_vec, 1)

    layer_pts = [object_point_cloud]
    layer_fts = [None]

    for layer_idx, layer_param in enumerate(xconv_params):
        tag = 'xconv_' + str(layer_idx + 1) + '_'
        K = layer_param['K']
        D = layer_param['D']
        P = layer_param['P']
        C = layer_param['C']
        links = layer_param['links']
        if sampling != 'random' and links:
            print('Error: flexible links are supported only when random sampling is used!')
            exit()

        # get k-nearest points
        pts = layer_pts[-1]
        fts = layer_fts[-1]
        if P == -1 or (layer_idx > 0 and P == xconv_params[layer_idx - 1]['P']):
            qrs = layer_pts[-1]
        else:
            if sampling == 'fps':
                fps_indices = tf_sampling.farthest_point_sample(P, pts)
                batch_indices = tf.tile(tf.reshape(tf.range(N), (-1, 1, 1)), (1, P, 1))
                indices = tf.concat([batch_indices, tf.expand_dims(fps_indices, -1)], axis=-1)
                qrs = tf.gather_nd(pts, indices, name=tag + 'qrs')  # (N, P, 3)
            elif sampling == 'ids':
                indices = pf.inverse_density_sampling(pts, K, P)
                qrs = tf.gather_nd(pts, indices)
            elif sampling == 'random':
                qrs = tf.slice(pts, (0, 0, 0), (-1, P, -1), name=tag + 'qrs')  # (N, P, 3)
            else:
                print('Unknown sampling method!')
                exit()
        layer_pts.append(qrs)
        # print('layer_pts',layer_pts)

        if layer_idx == 0:
            C_pts_fts = C // 2 if fts is None else C // 4
            depth_multiplier = 4
        else:
            C_prev = xconv_params[layer_idx - 1]['C']
            C_pts_fts = C_prev // 4
            depth_multiplier = math.ceil(C / C_prev)
        with_global = (with_global and layer_idx == len(xconv_params) - 1)
        fts_xconv = xconv(pts, fts, qrs, tag, N, K, D, P, C, C_pts_fts, is_training,
                          with_X_transformation,
                          depth_multiplier, sorting_method, with_global)
        #print('fts_xconv:', fts_xconv)

        one_hot_vec = tf.tile(one_hot_vec, [1, P, 1])
        # print('one_hot_vec:', one_hot_vec)
        fts_xconv = tf.concat([fts_xconv, one_hot_vec], axis=2)
        one_hot_vec = tf.slice(one_hot_vec, [0, 0, 0], [-1, 1, -1])
        # print('one_hot_vec_slice:', one_hot_vec)

        fts_list = []
        for link in links:
            fts_from_link = layer_fts[link]
            if fts_from_link is not None:
                fts_slice = tf.slice(fts_from_link, (0, 0, 0), (-1, P, -1),
                                     name=tag + 'fts_slice_' + str(-link))
                fts_list.append(fts_slice)
        if fts_list:
            fts_list.append(fts_xconv)
            layer_fts.append(tf.concat(fts_list, axis=-1, name=tag + 'fts_list_concat'))
        else:
            layer_fts.append(fts_xconv)
        print('layer_fts', layer_fts)

    fc_layers = [layer_fts[-1]]
    for layer_idx, layer_param in enumerate(fc_params):
        C = layer_param['C']
        dropout_rate = layer_param['dropout_rate']
        fc = pf.dense(fc_layers[-1], C, 'fc{:d}'.format(layer_idx), is_training)
        fc_drop = tf.layers.dropout(fc, dropout_rate, training=is_training,
                                    name='fc{:d}_drop'.format(layer_idx))
        fc_layers.append(fc_drop)

    # print('fc_layers:',fc_layers)
    fc_mean = tf.reduce_mean(fc_layers[-1], axis=1, keepdims=True, name='fc_mean')
    fc_layers[-1] = tf.cond(is_training, lambda: fc_layers[-1], lambda: fc_mean)
    logits = pf.dense(fc_layers[-1], num_class, 'logits',
                      is_training, with_bn=False, activation=None)
    print('logits:', logits)
    logits = tf_util.conv1d(logits, 2, 1, padding='VALID', activation_fn=None, scope='conv1d-fc')

    net = tf_util.fully_connected(logits, 512, bn=True,
                                  is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
                                  is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net, 3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4,
                                     activation_fn=None, scope='fc3')
    return output, end_points'''

'''def boundingbox_3d(object_point_cloud, one_hot_vec, is_training, bn_decay, end_points):
    
    # Gather object points
    batch_size = object_point_cloud.get_shape()[0].value

    l0_xyz = object_point_cloud
    l0_points = None
    # Set abstraction layers
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
        npoint=128, radius=0.2, nsample=64, mlp=[64,64,128],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer1')
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
        npoint=32, radius=0.4, nsample=64, mlp=[128,128,256],
        mlp2=None, group_all=False,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
        npoint=None, radius=None, nsample=None, mlp=[256,256,512],
        mlp2=None, group_all=True,
        is_training=is_training, bn_decay=bn_decay, scope='ssg-layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf.concat([net, one_hot_vec], axis=1)
    net = tf_util.fully_connected(net, 512, bn=True,
        is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.fully_connected(net, 256, bn=True,
        is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    return output, end_points'''

def boundingbox_3d(object_point_cloud, one_hot_vec, is_training, bn_decay, end_points):
    num_point = object_point_cloud.get_shape()[1].value
    net1 = tf.expand_dims(object_point_cloud, 2)
    net1 = tf_util.conv2d(net1, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg1', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 128, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg2', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 256, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg3', bn_decay=bn_decay)
    net1 = tf_util.conv2d(net1, 512, [1,1],
                         padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training,
                         scope='conv-reg4', bn_decay=bn_decay)
    net1 = tf_util.max_pool2d(net1, [num_point,1],
        padding='VALID', scope='maxpool2')
    net1 = tf.squeeze(net1, axis=[1, 2])
    net1 = tf.concat([net1, one_hot_vec], axis=1)
    net1 = tf_util.fully_connected(net1, 512, scope='fc1', bn=True,
        is_training=is_training, bn_decay=bn_decay)
    #print(net1)
    net1 = tf_util.fully_connected(net1, 256, scope='fc2', bn=True,
        is_training=is_training, bn_decay=bn_decay)

    # The first 3 numbers: box center coordinates (cx,cy,cz),
    # the next NUM_HEADING_BIN*2:  heading bin class scores and bin residuals
    # next NUM_SIZE_CLUSTER*4: box cluster scores and residuals
    output = tf_util.fully_connected(net1,
        3+NUM_HEADING_BIN*2+NUM_SIZE_CLUSTER*4, activation_fn=None, scope='fc3')
    print('output: ',output)
    return output, end_points
        
def get_model(points, features, one_hot_vec, is_training, bn_decay=None):

    end_points = {}
    
    # 3D Instance Segmentation 
    logits, end_points = pointcnn_seg(points, features, one_hot_vec, is_training, end_points)
    end_points['mask_logits'] = logits

    # Masking
    # select masked points and translate to masked points' centroid
    object_point_cloud_xyz, mask_xyz_mean, end_points = point_cloud_masking(points, logits, end_points)

    # T-Net and coordinate translation
    center_delta, end_points = get_center_regression_net(object_point_cloud_xyz, one_hot_vec,
                                                         is_training, bn_decay, end_points)
    stage1_center = center_delta + mask_xyz_mean # Bx3
    end_points['stage1_center'] = stage1_center
    # Get object point cloud in object coordinate
    object_point_cloud_xyz_new = object_point_cloud_xyz - tf.expand_dims(center_delta, 1)

    # Amodel Box Estimation PointNet
    output, end_points = boundingbox_3d(object_point_cloud_xyz_new, one_hot_vec,
                                        is_training, bn_decay, end_points)

    # Parse output to 3D box parameters
    end_points = parse_output_to_tensors(output, end_points)
    end_points['center'] = end_points['center_boxnet'] + stage1_center # Bx3

    return end_points

if __name__=='__main__':
    with tf.Graph().as_default():
        sess = tf.Session()
        inputs = tf.ones((32,2048,4))
        features = tf.ones((32,3))
        outputs = get_model(inputs, features, tf.ones((32,3)), tf.constant(True))
        #sess = tf_debug.LocalCLIDebugWrapperSession(sess=sess)
        #sess.run(outputs)
        
        for key in outputs:
            print((key, outputs[key]))
        loss = get_loss(tf.zeros((32,1024),dtype=tf.int32),
                        tf.zeros((32,3)), tf.zeros((32,),dtype=tf.int32),
                        tf.zeros((32,)), tf.zeros((32,),dtype=tf.int32),
                        tf.zeros((32,3)), outputs)
        print(loss)
