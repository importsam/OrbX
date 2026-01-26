16/01
Need to add density estimation to clusters, do not pre-process.
Need to make clusters stricter, somehow this must be measured.

One of the main problems in the tsne space is that separated clusters are not being considered their own, should change with strictness.
Need to write the section on clustering with what is here, change later per new findings and ideas.

On stability
In unsupervised clustering, a lot of the validation is qualitative, so in this case quantitative measures are relative.
Some foundational issues that have been addressed with stability include an estimate of confidence to an item's membership to a cluster, an estimate of confidence to cluster, and an overall estimate of confidence for a clustering of a dataset. Synonymous with cluster stability is its utility in the selection of the optimal number of clusters, which herein we refer to as model selection. 

One clustering method: https://cran.r-project.org/web/packages/fpc/index.html
another: https://cran.r-project.org/web/packages/bootcluster/index.html

For the clustering algs with tunable parameters, we can find a range of parameters which yield clusters within the assigned range. We then find the one with the highest score in that range and 

OPTICS:
(orbs) admin@ip-172-31-28-54:~/SatelliteClustering$ python main.py
Parsing TLE data from Space-Track...
Loaded 2857 satellites in range - inc: (80, 100), apogee: (300, 700)
Calculating orbits
Calculating distances
Distance matrix is valid
Min Samples: 2, DBCV Score: 0.5026247543013831, Clusters Found: 803
Min Samples: 3, DBCV Score: 0.5224568603364995, Clusters Found: 403
Min Samples: 4, DBCV Score: 0.4890044092259751, Clusters Found: 283
Min Samples: 5, DBCV Score: 0.47548129165564673, Clusters Found: 206
Min Samples: 6, DBCV Score: 0.46514568389853966, Clusters Found: 175
Min Samples: 7, DBCV Score: 0.4283606784060988, Clusters Found: 134
Min Samples: 8, DBCV Score: 0.41018743629087745, Clusters Found: 118
Min Samples: 9, DBCV Score: 0.40576176856742124, Clusters Found: 105

so we have done this, now we need to look further on min_samples


how do you justify the range of 200-500

we look at the ratio between number of clusters and noise points or maybe the ratio of number of clusters to average cluster size.
We try to strike a balance here, not too strict, not too loose. Then within this middle range, we pick the one with the highest DBCV score.


OPTICS parameter grid search:
(orbs) admin@ip-172-31-28-54:~/SatelliteClustering$ python main.py
Parsing TLE data from Space-Track...
Loaded 2857 satellites in range - inc: (80, 100), apogee: (300, 700)
Calculating orbits
Calculating distances
Distance matrix is valid
min_samples= 2, xi=0.0050, clusters= 821, noise= 463, DBCV=0.5038
min_samples= 2, xi=0.0075, clusters= 819, noise= 469, DBCV=0.5042
min_samples= 2, xi=0.0113, clusters= 818, noise= 474, DBCV=0.5053
min_samples= 2, xi=0.0171, clusters= 818, noise= 480, DBCV=0.5038
min_samples= 2, xi=0.0258, clusters= 810, noise= 499, DBCV=0.5032
min_samples= 2, xi=0.0388, clusters= 806, noise= 515, DBCV=0.5019
min_samples= 2, xi=0.0585, clusters= 800, noise= 530, DBCV=0.5019
min_samples= 2, xi=0.0881, clusters= 783, noise= 581, DBCV=0.5049
min_samples= 2, xi=0.1327, clusters= 767, noise= 641, DBCV=0.5038
min_samples= 2, xi=0.2000, clusters= 738, noise= 731, DBCV=0.5038
min_samples= 3, xi=0.0050, clusters= 412, noise= 770, DBCV=0.5283
min_samples= 3, xi=0.0075, clusters= 411, noise= 774, DBCV=0.5278
min_samples= 3, xi=0.0113, clusters= 411, noise= 777, DBCV=0.5271
min_samples= 3, xi=0.0171, clusters= 411, noise= 777, DBCV=0.5271
min_samples= 3, xi=0.0258, clusters= 408, noise= 792, DBCV=0.5261
min_samples= 3, xi=0.0388, clusters= 405, noise= 810, DBCV=0.5232
min_samples= 3, xi=0.0585, clusters= 401, noise= 835, DBCV=0.5227
min_samples= 3, xi=0.0881, clusters= 395, noise= 863, DBCV=0.5199
min_samples= 3, xi=0.1327, clusters= 385, noise= 927, DBCV=0.5130
min_samples= 3, xi=0.2000, clusters= 356, noise=1089, DBCV=0.4936
min_samples= 4, xi=0.0050, clusters= 289, noise= 958, DBCV=0.4917
min_samples= 4, xi=0.0075, clusters= 288, noise= 960, DBCV=0.4941
min_samples= 4, xi=0.0113, clusters= 287, noise= 965, DBCV=0.4933
min_samples= 4, xi=0.0171, clusters= 287, noise= 969, DBCV=0.4924
min_samples= 4, xi=0.0258, clusters= 285, noise= 983, DBCV=0.4920
min_samples= 4, xi=0.0388, clusters= 283, noise=1002, DBCV=0.4896
min_samples= 4, xi=0.0585, clusters= 282, noise=1018, DBCV=0.4882
min_samples= 4, xi=0.0881, clusters= 271, noise=1086, DBCV=0.4857
min_samples= 4, xi=0.1327, clusters= 258, noise=1159, DBCV=0.4836
min_samples= 4, xi=0.2000, clusters= 238, noise=1297, DBCV=0.4553
min_samples= 5, xi=0.0050, clusters= 214, noise=1021, DBCV=0.4810
min_samples= 5, xi=0.0075, clusters= 214, noise=1021, DBCV=0.4810
min_samples= 5, xi=0.0113, clusters= 214, noise=1024, DBCV=0.4805
min_samples= 5, xi=0.0171, clusters= 212, noise=1044, DBCV=0.4779
min_samples= 5, xi=0.0258, clusters= 209, noise=1065, DBCV=0.4743
min_samples= 5, xi=0.0388, clusters= 206, noise=1095, DBCV=0.4762
min_samples= 5, xi=0.0585, clusters= 205, noise=1113, DBCV=0.4747
min_samples= 5, xi=0.0881, clusters= 199, noise=1177, DBCV=0.4622
min_samples= 5, xi=0.1327, clusters= 195, noise=1230, DBCV=0.4624
min_samples= 5, xi=0.2000, clusters= 180, noise=1355, DBCV=0.4372
min_samples= 6, xi=0.0050, clusters= 179, noise=1017, DBCV=0.4563
min_samples= 6, xi=0.0075, clusters= 179, noise=1017, DBCV=0.4563
min_samples= 6, xi=0.0113, clusters= 178, noise=1024, DBCV=0.4550
min_samples= 6, xi=0.0171, clusters= 178, noise=1025, DBCV=0.4548
min_samples= 6, xi=0.0258, clusters= 177, noise=1026, DBCV=0.4598
min_samples= 6, xi=0.0388, clusters= 175, noise=1057, DBCV=0.4636
min_samples= 6, xi=0.0585, clusters= 174, noise=1097, DBCV=0.4627
min_samples= 6, xi=0.0881, clusters= 167, noise=1136, DBCV=0.4571
min_samples= 6, xi=0.1327, clusters= 157, noise=1247, DBCV=0.4549
min_samples= 6, xi=0.2000, clusters= 144, noise=1390, DBCV=0.4325
min_samples= 7, xi=0.0050, clusters= 136, noise=1143, DBCV=0.4197
min_samples= 7, xi=0.0075, clusters= 136, noise=1145, DBCV=0.4192
min_samples= 7, xi=0.0113, clusters= 136, noise=1145, DBCV=0.4192
min_samples= 7, xi=0.0171, clusters= 136, noise=1149, DBCV=0.4193
min_samples= 7, xi=0.0258, clusters= 135, noise=1165, DBCV=0.4176
min_samples= 7, xi=0.0388, clusters= 134, noise=1189, DBCV=0.4265
min_samples= 7, xi=0.0585, clusters= 132, noise=1223, DBCV=0.4288
min_samples= 7, xi=0.0881, clusters= 126, noise=1282, DBCV=0.4179
min_samples= 7, xi=0.1327, clusters= 119, noise=1365, DBCV=0.4285
min_samples= 7, xi=0.2000, clusters= 102, noise=1560, DBCV=0.3849
min_samples= 8, xi=0.0050, clusters= 120, noise=1095, DBCV=0.4016
min_samples= 8, xi=0.0075, clusters= 119, noise=1106, DBCV=0.4025
min_samples= 8, xi=0.0113, clusters= 119, noise=1106, DBCV=0.4025
min_samples= 8, xi=0.0171, clusters= 119, noise=1112, DBCV=0.4010
min_samples= 8, xi=0.0258, clusters= 118, noise=1119, DBCV=0.4107
min_samples= 8, xi=0.0388, clusters= 117, noise=1136, DBCV=0.4092
min_samples= 8, xi=0.0585, clusters= 116, noise=1156, DBCV=0.4186
min_samples= 8, xi=0.0881, clusters= 107, noise=1261, DBCV=0.4217
min_samples= 8, xi=0.1327, clusters= 104, noise=1355, DBCV=0.4209
min_samples= 8, xi=0.2000, clusters=  90, noise=1538, DBCV=0.3943
min_samples= 9, xi=0.0050, clusters= 106, noise= 998, DBCV=0.3808
min_samples= 9, xi=0.0075, clusters= 106, noise=1003, DBCV=0.3819
min_samples= 9, xi=0.0113, clusters= 106, noise=1007, DBCV=0.3834
min_samples= 9, xi=0.0171, clusters= 106, noise=1018, DBCV=0.3835
min_samples= 9, xi=0.0258, clusters= 105, noise=1032, DBCV=0.3803
min_samples= 9, xi=0.0388, clusters= 105, noise=1039, DBCV=0.3940
min_samples= 9, xi=0.0585, clusters= 102, noise=1096, DBCV=0.4145
min_samples= 9, xi=0.0881, clusters=  99, noise=1165, DBCV=0.4065
min_samples= 9, xi=0.1327, clusters=  88, noise=1333, DBCV=0.3970
min_samples= 9, xi=0.2000, clusters=  66, noise=1714, DBCV=0.3486

Best OPTICS params → min_samples=3, xi=0.0050, DBCV=0.5283
OPTICS found 412 clusters (noise points: 770)
OPTICS found 413 clusters
Clustering Quality Metrics:
Primary - DBCV Score: 0.5283002690246926
Secondary - S_Dbw Score: 0.00855908969079414

Sanity - Viasckde Score: 0.06010184563304567

Calinski-Harabasz Score: 17.01147444690398
Silhouette Score: 0.23966249876747667
Davies-Bouldin Score: 1.0757829034722448
(orbs) admin@ip-172-31-28-54:~/SatelliteClustering$