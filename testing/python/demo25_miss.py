#!/usr/bin/env python
# coding: utf-8

# In[103]:

import glob
import os
import cv2 as cv
import numpy as np
import scipy
import PIL.Image
import math
import caffe
import time
from config_reader import config_reader
import util
import copy
import csv
import matplotlib
import pandas as pd
#get_ipython().magic(u'matplotlib inline')
import pylab as plt

eps = 1e-5

# In[104]:

#folder_name = '/home/tianlab/hengheng/pose/DukeMTMC-reID/pose/train_256'
#folder_image_suffix = '.png'
folder_name = '/home/hmhm/Pose/MuVS_ori/Data/cuhk03/detected/image'
folder_image_suffix = '.jpg'

save_path = '/home/hmhm/Pose/MuVS_ori/Data/cuhk03/detected/pose'

bad_pose_csv = os.path.join('/home/hmhm/Pose/MuVS_ori/Data/cuhk03/detected', 'bad_pose.csv')

if not os.path.exists(save_path):
    os.mkdir(save_path)

img_df = pd.read_csv('/home/hmhm/Pose/MuVS_ori/Data/cuhk03/detected/3d_miss_v1.csv', header=None)
image_list = img_df[0].tolist()

# image_list = sorted(glob.glob(os.path.join(folder_name, '*' + folder_image_suffix)))

#%%

for test_image in image_list:

    #test_image = '../sample_image/ski.jpg'
    # test_image = '../sample_image/686_0.png'
    #test_image = '../sample_image/upper.jpg'
    #test_image = '../sample_image/upper2.jpg'
    oriImg = cv.imread(test_image) # B,G,R order
    f = plt.imshow(oriImg[:,:,[2,1,0]]) # reorder it before displaying


    # In[105]:


    param, model = config_reader()
    multiplier = [x * model['boxsize'] / oriImg.shape[0] for x in param['scale_search']]


    # In[106]:


    if param['use_gpu']:
        caffe.set_mode_gpu()
        caffe.set_device(param['GPUdeviceNumber']) # set to your device!
    else:
        caffe.set_mode_cpu()
    net = caffe.Net(model['deployFile'], model['caffemodel'], caffe.TEST)


    # In[107]:


    heatmap_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 26))
    paf_avg = np.zeros((oriImg.shape[0], oriImg.shape[1], 52))
    # first figure shows padded images

    # f, axarr = plt.subplots(1, len(multiplier))
    # f.set_size_inches((20, 5))
    # # second figure shows heatmaps
    # f2, axarr2 = plt.subplots(1, len(multiplier))
    # f2.set_size_inches((20, 5))
    # # third figure shows PAFs
    # f3, axarr3 = plt.subplots(2, len(multiplier))
    # f3.set_size_inches((20, 10))


    for m in range(len(multiplier)):
        scale = multiplier[m]
        imageToTest = cv.resize(oriImg, (0,0), fx=scale, fy=scale, interpolation=cv.INTER_CUBIC)
        imageToTest_padded, pad = util.padRightDownCorner(imageToTest, model['stride'], model['padValue'])
        print(imageToTest_padded.shape)

#        axarr[m].imshow(imageToTest_padded[:,:,[2,1,0]])
#        axarr[m].set_title('Input image: scale %d' % m)

        # print(net.blobs)

        # net.blobs['data'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        net.blobs['image'].reshape(*(1, 3, imageToTest_padded.shape[0], imageToTest_padded.shape[1]))
        #net.forward() # dry run
        # net.blobs['data'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        net.blobs['image'].data[...] = np.transpose(np.float32(imageToTest_padded[:,:,:,np.newaxis]), (3,2,0,1))/256 - 0.5;
        start_time = time.time()
        output_blobs = net.forward()
        print('At scale %d, The CNN took %.2f ms.' % (m, 1000 * (time.time() - start_time)))

        # print(list(output_blobs))

        # extract outputs, resize, and remove padding
        heatmap = np.transpose(np.squeeze(net.blobs[list(output_blobs)[0]].data[:,0:26,:,:]), (1,2,0)) # output 1 is heatmaps
        heatmap = cv.resize(heatmap, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        heatmap = heatmap[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        heatmap = cv.resize(heatmap, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)

        paf = np.transpose(np.squeeze(net.blobs[list(output_blobs)[0]].data[:,26:78,:,:]), (1,2,0)) # output 0 is PAFs
        paf = cv.resize(paf, (0,0), fx=model['stride'], fy=model['stride'], interpolation=cv.INTER_CUBIC)
        paf = paf[:imageToTest_padded.shape[0]-pad[2], :imageToTest_padded.shape[1]-pad[3], :]
        paf = cv.resize(paf, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv.INTER_CUBIC)


#        # visualization
#        axarr2[m].imshow(oriImg[:,:,[2,1,0]])
#        ax2 = axarr2[m].imshow(heatmap[:,:,4], alpha=.5) # right wrist
#        axarr2[m].set_title('Heatmaps (Rwri): scale %d' % m)
#
#        axarr3.flat[m].imshow(oriImg[:,:,[2,1,0]])
#        ax3x = axarr3.flat[m].imshow(paf[:,:,18], alpha=.5) # right elbow
#        axarr3.flat[m].set_title('PAFs (x comp. of Rwri to Relb): scale %d' % m)
#        axarr3.flat[len(multiplier) + m].imshow(oriImg[:,:,[2,1,0]])
#        ax3y = axarr3.flat[len(multiplier) + m].imshow(paf[:,:,19], alpha=.5) # right wrist
#        axarr3.flat[len(multiplier) + m].set_title('PAFs (y comp. of Relb to Rwri): scale %d' % m)


        # print(heatmap_avg.shape)
        heatmap_avg = heatmap_avg + heatmap / len(multiplier)
        paf_avg = paf_avg + paf / len(multiplier)


#    f2.subplots_adjust(right=0.93)
#    cbar_ax = f2.add_axes([0.95, 0.15, 0.01, 0.7])
#    _ = f2.colorbar(ax2, cax=cbar_ax)
#
#    f3.subplots_adjust(right=0.93)
#    cbar_axx = f3.add_axes([0.95, 0.57, 0.01, 0.3])
#    _ = f3.colorbar(ax3x, cax=cbar_axx)
#    cbar_axy = f3.add_axes([0.95, 0.15, 0.01, 0.3])
#    _ = f3.colorbar(ax3y, cax=cbar_axy)



    # Let's have a closer look on those averaged heatmaps and PAFs!

    # In[102]:



    # plt.imshow(oriImg[:,:,[2,1,0]])
    # plt.imshow(heatmap_avg[:,:,7], alpha=.5)
    # fig = matplotlib.pyplot.gcf()
    # cax = matplotlib.pyplot.gca()
    # fig.set_size_inches(20, 20)
    # fig.subplots_adjust(right=0.93)
    # cbar_ax = fig.add_axes([0.95, 0.15, 0.01, 0.7])
    # _ = fig.colorbar(ax2, cax=cbar_ax)



    # In[68]:


#    from numpy import ma
#    U = paf_avg[:,:,16] * -1
#    V = paf_avg[:,:,17]
#    X, Y = np.meshgrid(np.arange(U.shape[1]), np.arange(U.shape[0]))
#    M = np.zeros(U.shape, dtype='bool')
#    M[U**2 + V**2 < 0.5 * 0.5] = True
#    U = ma.masked_array(U, mask=M)
#    V = ma.masked_array(V, mask=M)
#
#    # 1
#    plt.figure()
#    plt.imshow(oriImg[:,:,[2,1,0]], alpha = .5)
#    s = 5
#    Q = plt.quiver(X[::s,::s], Y[::s,::s], U[::s,::s], V[::s,::s],
#                    scale=50, headaxislength=4, alpha=.5, width=0.001, color='r')
#
#    fig = matplotlib.pyplot.gcf()
#    fig.set_size_inches(20, 20)



    # In[69]:


    import scipy
    print(heatmap_avg.shape)

    #plt.imshow(heatmap_avg[:,:,2])
    from scipy.ndimage.filters import gaussian_filter
    all_peaks = []
    peak_counter = 0

    for part in range(26-1):
        x_list = []
        y_list = []
        map_ori = heatmap_avg[:,:,part]
        map = gaussian_filter(map_ori, sigma=3)

        map_left = np.zeros(map.shape)
        map_left[1:,:] = map[:-1,:]
        map_right = np.zeros(map.shape)
        map_right[:-1,:] = map[1:,:]
        map_up = np.zeros(map.shape)
        map_up[:,1:] = map[:,:-1]
        map_down = np.zeros(map.shape)
        map_down[:,:-1] = map[:,1:]

        peaks_binary = np.logical_and.reduce((map>=map_left, map>=map_right, map>=map_up, map>=map_down, map > param['thre1']))
        peaks = zip(np.nonzero(peaks_binary)[1], np.nonzero(peaks_binary)[0]) # note reverse
        peaks_with_score = [x + (map_ori[x[1],x[0]],) for x in peaks]
        id = range(peak_counter, peak_counter + len(peaks))
        peaks_with_score_and_id = [peaks_with_score[i] + (id[i],) for i in range(len(id))]

        all_peaks.append(peaks_with_score_and_id)
        peak_counter += len(peaks)


    # In[70]:


    # find connection in the specified sequence, center 29 is in the position 15
    # index from matlab (start from 1)
    #limbSeq = [[2,3], [2,6], [3,4], [4,5], [6,7], [7,8], [2,9], [9,10], \
    #           [10,11], [2,12], [12,13], [13,14], [2,1], [1,15], [15,17], \
    #           [1,16], [16,18], [3,17], [6,18]]

    # body 25
    limbSeq = [[1,8], [1,2], [1,5], [2,3], [3,4], [5,6], [6,7], [8,9], [9,10], [10,11], \
               [8,12], [12,13], [13,14], [1,0], [0,15], [15,17], [0,16], [16,18],  \
               [14,19], [19,20], [14,21], [11,22], [22,23], [11,24], [2,17], [5,18]]

    # index from matlab (start from 1)
    #mapIdx = [[31,32], [39,40], [33,34], [35,36], [41,42], [43,44], [19,20], [21,22], \
    #          [23,24], [25,26], [27,28], [29,30], [47,48], [49,50], [53,54], [51,52], \
    #          [55,56], [37,38], [45,46]]

    mapIdx = [[0,1], [14,15], [22,23], [16,17], [18,19], [24,25], [26,27], [6,7], [2,3], [4,5], \
              [8,9], [10,11], [12,13], [30,31], [32,33], [36,37], [34,35], [38,39], \
              [40,41], [42,43], [44,45], [46,47], [48,49], [50,51], [20,21], [28,29]]
    # In[71]:


    connection_all = []
    special_k = []
    mid_num = 10

    for k in range(len(mapIdx)):
        score_mid = paf_avg[:,:,[x for x in mapIdx[k]]]
        candA = all_peaks[limbSeq[k][0]]
        candB = all_peaks[limbSeq[k][1]]
        nA = len(candA)
        nB = len(candB)
        indexA, indexB = limbSeq[k]
        if(nA != 0 and nB != 0):
            connection_candidate = []
            for i in range(nA):
                for j in range(nB):
                    vec = np.subtract(candB[j][:2], candA[i][:2])
                    norm = math.sqrt(vec[0]*vec[0] + vec[1]*vec[1])
                    vec = np.divide(vec, norm + eps)

                    startend = zip(np.linspace(candA[i][0], candB[j][0], num=mid_num), np.linspace(candA[i][1], candB[j][1], num=mid_num))

                    vec_x = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 0] for I in range(len(startend))])
                    vec_y = np.array([score_mid[int(round(startend[I][1])), int(round(startend[I][0])), 1] for I in range(len(startend))])

                    score_midpts = np.multiply(vec_x, vec[0]) + np.multiply(vec_y, vec[1])
                    score_with_dist_prior = sum(score_midpts)/len(score_midpts) + min(0.5*oriImg.shape[0]/(norm+eps)-1, 0)
                    criterion1 = len(np.nonzero(score_midpts > param['thre2'])[0]) > 0.8 * len(score_midpts)
                    criterion2 = score_with_dist_prior > 0
                    if criterion1 and criterion2:
                        connection_candidate.append([i, j, score_with_dist_prior, score_with_dist_prior+candA[i][2]+candB[j][2]])

            connection_candidate = sorted(connection_candidate, key=lambda x: x[2], reverse=True)
            connection = np.zeros((0,5))
            for c in range(len(connection_candidate)):
                i,j,s = connection_candidate[c][0:3]
                if(i not in connection[:,3] and j not in connection[:,4]):
                    connection = np.vstack([connection, [candA[i][3], candB[j][3], s, i, j]])
                    if(len(connection) >= min(nA, nB)):
                        break

            connection_all.append(connection)
        else:
            special_k.append(k)
            connection_all.append([])


    # In[72]:


    # last number in each row is the total parts number of that person
    # the second last number in each row is the score of the overall configuration
    subset = -1 * np.ones((0, 27))
    candidate = np.array([item for sublist in all_peaks for item in sublist])

    for k in range(len(mapIdx)):
        if k not in special_k:
            partAs = connection_all[k][:,0]
            partBs = connection_all[k][:,1]
            indexA, indexB = np.array(limbSeq[k])

            for i in range(len(connection_all[k])): #= 1:size(temp,1)
                found = 0
                subset_idx = [-1, -1]
                for j in range(len(subset)): #1:size(subset,1):
                    if subset[j][indexA] == partAs[i] or subset[j][indexB] == partBs[i]:
                        subset_idx[found] = j
                        found += 1

                if found == 1:
                    j = subset_idx[0]
                    if(subset[j][indexB] != partBs[i]):
                        subset[j][indexB] = partBs[i]
                        subset[j][-1] += 1
                        subset[j][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]
                elif found == 2: # if found 2 and disjoint, merge them
                    j1, j2 = subset_idx
                    print("found = 2")
                    membership = ((subset[j1]>=0).astype(int) + (subset[j2]>=0).astype(int))[:-2]
                    if len(np.nonzero(membership == 2)[0]) == 0: #merge
                        subset[j1][:-2] += (subset[j2][:-2] + 1)
                        subset[j1][-2:] += subset[j2][-2:]
                        subset[j1][-2] += connection_all[k][i][2]
                        subset = np.delete(subset, j2, 0)
                    else: # as like found == 1
                        subset[j1][indexB] = partBs[i]
                        subset[j1][-1] += 1
                        subset[j1][-2] += candidate[partBs[i].astype(int), 2] + connection_all[k][i][2]

                # if find no partA in the subset, create a new subset
                elif not found and k < 25:
                    row = -1 * np.ones(27)
                    row[indexA] = partAs[i]
                    row[indexB] = partBs[i]
                    row[-1] = 2
                    row[-2] = sum(candidate[connection_all[k][i,:2].astype(int), 2]) + connection_all[k][i][2]
                    subset = np.vstack([subset, row])


    # In[73]:


    # delete some rows of subset which has few parts occur
    deleteIdx = [];
    for i in range(len(subset)):
        if subset[i][-1] < 4 or subset[i][-2]/subset[i][-1] < 0.4:
            deleteIdx.append(i)
    subset = np.delete(subset, deleteIdx, axis=0)


    # In[74]:



 # visualize
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]]
    cmap = matplotlib.cm.get_cmap('hsv')

    canvas = cv.imread(test_image) # B,G,R order

    #toe_pos = [0, 0]

#    for i in range(25):
#        rgba = np.array(cmap(1 - i/18. - 1./36))
#        rgba[0:3] *= 255
#        for j in range(len(all_peaks[i])):
#    #        toe_pos[0] += all_peaks[i][j][0]
#    #        toe_pos[1] += all_peaks[i][j][1]
#            cv.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
#    #toe_pos[0] = toe_pos[0] / 2
#    #toe_pos[1] = toe_pos[1] / 2
#    #toe_pos = tuple(toe_pos)
#    #cv.circle(canvas, toe_pos, 4, colors[i], thickness=-1)
##    num_parts = 0
##    choice = 0
##    for i in range(len(subset)):
##        num_parts_max = subset[i][-1]
##        if num_parts_max > num_parts:
##            choice = i
##            num_parts = num_parts_max
##
##    for i, idx in enumerate(subset[choice][0:25]):
##        rgba = np.array(cmap(1 - i/18. - 1./36))
##        rgba[0:3] *= 255
##        if idx == -1:
##            continue
##        for j in range(len(all_peaks[i])):
##            if all_peaks[i][j][3] == idx:
##                cv.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
#

    for i in range(25):
        rgba = np.array(cmap(1 - i/18. - 1./36))
        rgba[0:3] *= 255
        for j in range(len(all_peaks[i])):
     #        toe_pos[0] += all_peaks[i][j][0]
     #        toe_pos[1] += all_peaks[i][j][1]
            cv.circle(canvas, all_peaks[i][j][0:2], 4, colors[i], thickness=-1)
     #toe_pos[0] = toe_pos[0] / 2
     #toe_pos[1] = toe_pos[1] / 2
     #toe_pos = tuple(toe_pos)
     #cv.circle(canvas, toe_pos, 4, colors[i], thickness=-1)


#    to_plot = cv.addWeighted(oriImg, 0.3, canvas, 0.7, 0)
#    plt.imshow(to_plot[:,:,[2,1,0]])
#    fig = matplotlib.pyplot.gcf()
#    fig.set_size_inches(12, 12)



    # In[75]:

    canvas = cv.imread(test_image) # B,G,R order
    colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255], \
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85], \
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255]]
    # visualize 2
    stickwidth = 2

    for i in range(24):
        for n in range(len(subset)):
            index = subset[n][np.array(limbSeq[i])]
            if -1 in index:
                continue
            cur_canvas = canvas.copy()
            Y = candidate[index.astype(int), 0]
            X = candidate[index.astype(int), 1]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv.ellipse2Poly((int(mY),int(mX)), (int(length/2), stickwidth), int(angle), 0, 360, 1)
            cv.fillConvexPoly(cur_canvas, polygon, colors[i])
            canvas = cv.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)


    # plt.imshow(canvas[:,:,[2,1,0]])
    # fig = matplotlib.pyplot.gcf()
    # fig.set_size_inches(12, 12)



    # In[ ]:

	# idx_subset = np.array(range(25))
    # idx_sub0 = idx_subset[subset[0][0:25]!=-1]
    # idx_sub1 = idx_subset[subset[1][0:25]!=-1]
	# # idx_union = np.union1d(idx_sub0, idx_sub1)
    # print(subset)
    # idx_intersect = np.intersect1d(idx_sub0, idx_sub1)
    # print(idx_intersect)
    # if len(idx_intersect) < 2:
    #     subset_new = -1 * np.ones((1,27))
    #     subset_new[0][idx_sub0] = subset[0][idx_sub0]
    #     subset_new[0][idx_sub1] = subset[1][idx_sub1]
    #     print(subset_new)

    if len(subset) == 0:
        with open(bad_pose_csv, 'a') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([os.path.basename(test_image)])

    else:
        pose = -1 * np.ones((3,25))
        num_parts_max = 0
        choice = 0
        if len(subset) >= 2:
            idx_subset = np.array(range(25))
            idx_sub1 = idx_subset[subset[1][0:25]!=-1]
            idx_sub0 = idx_subset[subset[0][0:25]!=-1]
            idx_intersect = np.intersect1d(idx_sub0, idx_sub1)
            if len(idx_intersect) < 3:
                subset_new = -1 * np.ones((1,27))
                subset_new[0][idx_sub0] = subset[0][idx_sub0]
                subset_new[0][idx_sub1] = subset[1][idx_sub1]
                choice = 0
                subset = subset_new
            else:
                for i in range(len(subset)):
                    num_parts = subset[i][-1]
                    if num_parts > num_parts_max:
                        choice = i
                        num_parts_max = num_parts

        for i, idx in enumerate(subset[choice][0:25]):
            if subset[choice][i] == -1:
                continue
            for j in range(len(all_peaks[i])):
                if all_peaks[i][j][3] == idx:
                    pose[0, i] = all_peaks[i][j][0]
                    pose[1, i] = all_peaks[i][j][1]
                    pose[2, i] = all_peaks[i][j][2]

        vis_name = os.path.join(save_path, os.path.basename(test_image) + '_vis.png')
        scipy.misc.imsave(vis_name, canvas[:,:,[2,1,0]])

        out_name = os.path.join(save_path, os.path.basename(test_image) + '_pose.npz')
        np.savez_compressed(out_name, pose=pose)



# In[ ]:
